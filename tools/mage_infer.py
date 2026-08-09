#!/usr/bin/env python
"""Clean Mage-Flow text-to-image inference (no content screening, no watermark).

Loads a Mage-Flow HF-style repo (model_index.json + transformer/ vae/
text_encoder/ scheduler/), optionally merges one or more LoRAs into the
transformer, and generates images. Reuses the upstream sampling internals from
``Mage/mage_flow/pipeline.py`` (packing, CFG, scheduler) but never calls the
mandatory ``screen_text``/``screen_edit`` gate or the Gaussian-Shading
``encode_noise`` watermark — the initial latent is plain Gaussian noise and
every prompt is generated.

Attention runs on the SDPA backend by default, so flash-attn is not required.

Usage (CLI):
    python tools/mage_infer.py --model ./mage-flow \
        --prompt "a red fox in snow, cinematic" \
        --negative "blurry, lowres" \
        --steps 30 --cfg 5.0 --width 1024 --height 1024 --seed 42 \
        --lora ./output/mage-flow-lora/epochXX/adapter_model.safetensors:0.8 \
        --out ./out

Usage (import):
    from tools.mage_infer import MageInfer
    m = MageInfer("./mage-flow", loras=[("adapter.safetensors", 0.8)])
    imgs = m.generate(["a cat"], steps=30, cfg=5.0)
"""

import argparse
import glob
import json
import os
import random
import sys

import torch
from einops import rearrange
from tqdm.auto import tqdm

# Vendored Mage package (Mage/mage_flow -> `import mage_flow`).
_MAGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Mage')
if _MAGE_ROOT not in sys.path:
    sys.path.insert(0, _MAGE_ROOT)

from mage_flow.models.mage_flow import MageFlowModel, ModelConfig  # noqa: E402
from mage_flow.models.modules._attn_backend import set_attn_backend  # noqa: E402
from mage_flow.models.utils import get_noise  # noqa: E402
from mage_flow.pipeline import (  # noqa: E402
    _as_list,
    _build_pack_ctx,
    _decode_one,
    _encode_texts_packed,
    _get_scheduler,
    _lens_to_cu,
    _make_divisible_by_16,
    _slice_packed,
    _template_info,
    _velocity,
)

# Config keys that are NOT MageFlowParams constructor args (mirrors
# pipeline.load_from_repo); everything else in transformer/config.json becomes
# the DiT model_structure.
_META_KEYS = {
    "_class_name", "txt_max_length", "max_sequence_length", "param_dtype",
    "packing", "schedule_mode", "static_shift", "use_time_shift",
    "rope_type", "apply_text_rotary_emb", "mlp_ratio", "depth_single_blocks",
    "theta", "qkv_bias", "guidance_embed", "vec_in_dim", "vec_type",
    "time_type", "double_block_type",
}

# LoRA key substrings that indicate a format this simple merger does not handle.
_UNSUPPORTED_LORA_MARKERS = ("hada_", "lokr_", "lora_w1", "lora_w2", ".oft_", ".boft_")


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def build_model(repo_dir, device="cuda", dtype=torch.bfloat16, attn_type=None):
    """Build a MageFlowModel from a repo dir, on the chosen attention backend.

    Bypasses ``pipeline.load_from_repo`` only to inject ``attn_type`` (default:
    'sdpa' unless flash-attn is importable). No behavior change otherwise.
    """
    from safetensors.torch import load_file
    from diffusers import FlowMatchEulerDiscreteScheduler

    if attn_type is None:
        attn_type = "flash2" if _flash_attn_available() else "sdpa"

    repo_dir = os.path.realpath(repo_dir)
    mi = json.load(open(os.path.join(repo_dir, "model_index.json")))
    tcfg = json.load(open(os.path.join(repo_dir, "transformer", "config.json")))
    structure = {k: v for k, v in tcfg.items() if k not in _META_KEYS}

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(repo_dir, p)

    cfg = ModelConfig(
        vae_path=_resolve(mi["_vae_source"]),
        txt_enc_path=_resolve(mi["_text_encoder_path"]),
        model_structure=structure,
        txt_max_length=tcfg.get("txt_max_length", 2048),
        packing=tcfg.get("packing", True),
        static_shift=tcfg.get("static_shift", 6.0),
        attn_type=attn_type,
    )
    set_attn_backend(attn_type)

    model = MageFlowModel(cfg)
    sd = load_file(os.path.join(repo_dir, "transformer", "diffusion_pytorch_model.safetensors"), device="cpu")
    model.transformer.load_state_dict(sd, strict=False, assign=True)
    model.to(device)
    model.transformer.to(dtype)
    model.txt_enc.to(dtype)
    if model.vae is not None:
        model.vae.to(dtype)
    model.eval()
    model.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(repo_dir, "scheduler"))
    return model


def _find_lora_files(lora_path):
    """Return (state_dict_path, config_path_or_None) for a file or a dir."""
    if os.path.isdir(lora_path):
        sts = sorted(glob.glob(os.path.join(lora_path, "*.safetensors")))
        if not sts:
            raise FileNotFoundError(f"No .safetensors in {lora_path}")
        st = sts[0]
        cfg = os.path.join(lora_path, "adapter_config.json")
        return st, (cfg if os.path.exists(cfg) else None)
    cfg = os.path.join(os.path.dirname(lora_path), "adapter_config.json")
    return lora_path, (cfg if os.path.exists(cfg) else None)


@torch.no_grad()
def merge_lora(transformer, lora_path, scale=1.0):
    """Merge a diffusion-pipe PEFT LoRA into the transformer weights in place.

    Expects the ComfyUI-style adapter saved by MageFlowPipeline.save_adapter:
    ``diffusion_model.<module>.lora_A.weight`` / ``lora_B.weight`` pairs, with
    rank + alpha from the sibling ``adapter_config.json``. Effective delta per
    module is ``scale * (alpha / r) * (B @ A)``. LyCORIS (LoHa/LoKr/OFT) formats
    are detected and rejected (not supported by this simple merger).
    """
    from safetensors.torch import load_file

    st_path, cfg_path = _find_lora_files(lora_path)
    sd = load_file(st_path, device="cpu")

    r_cfg = alpha_cfg = None
    if cfg_path:
        c = json.load(open(cfg_path))
        r_cfg = c.get("r")
        alpha_cfg = c.get("lora_alpha")

    modules = dict(transformer.named_modules())

    def _strip(key):
        for pre in ("diffusion_model.", "transformer.", "base_model.model.", "model."):
            if key.startswith(pre):
                key = key[len(pre):]
        return key.replace(".default.", ".").replace(".default", "")

    pairs = {}
    for k, v in sd.items():
        if any(marker in k for marker in _UNSUPPORTED_LORA_MARKERS):
            raise NotImplementedError(
                f"{os.path.basename(st_path)} looks like a LyCORIS/OFT adapter "
                f"(key '{k}'); this merger only supports plain PEFT LoRA. Merge it "
                f"with the lycoris library instead, or train a LoRA-type adapter."
            )
        key = _strip(k)
        if ".lora_A" in key or ".lora_down" in key:
            mod = key.split(".lora_")[0]
            pairs.setdefault(mod, {})["A"] = v
        elif ".lora_B" in key or ".lora_up" in key:
            mod = key.split(".lora_")[0]
            pairs.setdefault(mod, {})["B"] = v

    if not pairs:
        raise RuntimeError(f"No LoRA A/B pairs found in {st_path}")

    merged, skipped = 0, []
    for mod_name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            skipped.append(mod_name)
            continue
        module = modules.get(mod_name)
        if module is None or not hasattr(module, "weight"):
            skipped.append(mod_name)
            continue
        A = ab["A"].float()  # [r, in]
        B = ab["B"].float()  # [out, r]
        r = r_cfg if r_cfg else A.shape[0]
        alpha = alpha_cfg if alpha_cfg is not None else r
        delta = (scale * alpha / r) * (B @ A)
        w = module.weight
        module.weight.data = (w.float() + delta.to(w.device)).to(w.dtype)
        merged += 1

    print(f"Merged LoRA {os.path.basename(st_path)}: {merged} modules "
          f"(scale={scale}, r={r_cfg}, alpha={alpha_cfg})"
          + (f", skipped {len(skipped)}" if skipped else ""))
    return merged


@torch.no_grad()
def generate(model, prompts, neg_prompts=None, seeds=None, steps=30, cfg=5.0,
             heights=None, widths=None, device="cuda", prompt_template="mage-flow",
             static_shift=None, renormalization=False, batch_cfg=True, progress=True):
    """Clean packed multi-resolution t2i. Returns a list of PIL images.

    Identical sampling to pipeline.generate_images minus the mandatory content
    gate and the Gaussian-Shading watermark: no prompt is screened, no refusal
    placeholder is produced, and the initial latent is plain Gaussian noise.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    n = len(prompts)
    neg_prompts = _as_list(neg_prompts, " ", n)
    seeds = _as_list(seeds, 42, n)
    heights = _as_list(heights, 1024, n)
    widths = _as_list(widths, 1024, n)
    info = _template_info(prompt_template)
    template = info.get("template", "{}")
    drop_idx = int(info.get("start_idx", 0))
    dev = torch.device(device)

    ch = model.vae.latent_channels
    img_list, ids_list, lens, shapes, hw = [], [], [], [], []
    for i in range(n):
        if seeds[i] == -1:
            seeds[i] = random.randint(0, 2**32 - 1)
        h_, w_ = _make_divisible_by_16(heights[i]), _make_divisible_by_16(widths[i])
        # Plain Gaussian noise (NO watermark injection).
        x = get_noise(num_samples=1, channel=ch, height=h_, width=w_,
                      device=dev, dtype=torch.bfloat16, seed=seeds[i])
        _, _, gh, gw = x.shape
        img_list.append(rearrange(x, "b c h w -> b (h w) c")[0])
        ids = torch.zeros(gh, gw, 3, device=dev)
        ids[..., 1] = ids[..., 1] + torch.arange(gh, device=dev)[:, None]
        ids[..., 2] = ids[..., 2] + torch.arange(gw, device=dev)[None, :]
        ids_list.append(rearrange(ids, "h w c -> (h w) c"))
        lens.append(gh * gw); shapes.append((1, gh, gw)); hw.append((h_, w_))
    img = torch.cat(img_list, 0).unsqueeze(0)
    img_ids = torch.cat(ids_list, 0).unsqueeze(0)
    img_cu = _lens_to_cu(lens, dev)
    img_shapes = [shapes]

    na = n
    use_neg = cfg > 1.0 and any(neg_prompts[i] for i in range(n))
    if use_neg:
        neg_list = [neg_prompts[i] or " " for i in range(n)]
        txt_flat, vec_all, lens_t = _encode_texts_packed(
            model, list(prompts) + neg_list, template, drop_idx, dev)
        txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, na, dev)
        neg_txt, neg_cu, neg_mask, neg_vec = _slice_packed(txt_flat, vec_all, lens_t, na, na, dev)
    else:
        txt_flat, vec_all, lens_t = _encode_texts_packed(model, list(prompts), template, drop_idx, dev)
        txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, na, dev)
        neg_txt = neg_cu = neg_mask = neg_vec = None

    ctx = _build_pack_ctx(img_ids, img_cu, img_shapes, lens, txt, txt_cu, txt_mask, vec,
                          neg_txt, neg_cu, neg_mask, neg_vec, cfg, renormalization, batch_cfg, dev)
    scheduler = _get_scheduler(model, steps, device, static_shift)
    timesteps = list(enumerate(scheduler.timesteps))
    for si, t in tqdm(timesteps, desc="sampling", unit="step", disable=not progress):
        pred = _velocity(model.transformer, img, ctx, scheduler.sigmas[si].item())
        img = scheduler.step(pred, t, img, return_dict=False)[0]

    results = []
    off = 0
    for k in range(n):
        L = lens[k]
        h_, w_ = hw[k]
        results.append(_decode_one(model, img[:, off:off + L, :], h_, w_, dev))
        off += L
    return results


class MageInfer:
    """Convenience wrapper: build once, optionally merge LoRAs, generate many."""

    def __init__(self, repo_dir, device="cuda", dtype=torch.bfloat16,
                 attn_type=None, loras=None):
        self.device = device
        self.model = build_model(repo_dir, device=device, dtype=dtype, attn_type=attn_type)
        for lora_path, scale in (loras or []):
            merge_lora(self.model.transformer, lora_path, scale=scale)

    def generate(self, prompts, **kw):
        kw.setdefault("device", self.device)
        return generate(self.model, prompts, **kw)


def _parse_lora_arg(s):
    """'path:scale' -> (path, float). Trailing float after ':' is the scale."""
    if ":" in s:
        head, tail = s.rsplit(":", 1)
        try:
            return head, float(tail)
        except ValueError:
            return s, 1.0
    return s, 1.0


def main():
    ap = argparse.ArgumentParser(description="Clean Mage-Flow t2i inference (no safety, optional LoRAs).")
    ap.add_argument("--model", required=True, help="Path to Mage-Flow repo dir")
    ap.add_argument("--prompt", action="append", required=True, help="Prompt (repeatable)")
    ap.add_argument("--negative", default="", help="Negative prompt (applies to all)")
    ap.add_argument("--lora", action="append", default=[], help="LoRA as path or path:scale (repeatable)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42, help="-1 for random per image")
    ap.add_argument("--shift", type=float, default=None, help="Override static shift (default from repo/6.0)")
    ap.add_argument("--renorm", action="store_true", help="CFG renormalization")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default=None, choices=[None, "sdpa", "flash2", "flash4"], help="Attention backend")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default="./mage_out", help="Output directory")
    ap.add_argument("--quiet", action="store_true", help="Disable the sampling progress bar")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    loras = [_parse_lora_arg(s) for s in args.lora]

    m = MageInfer(args.model, device=args.device, dtype=dtype, attn_type=args.attn, loras=loras)

    prompts = args.prompt
    seeds = [args.seed] * len(prompts) if args.seed != -1 else [-1] * len(prompts)
    imgs = m.generate(
        prompts,
        neg_prompts=args.negative or None,
        seeds=seeds,
        steps=args.steps,
        cfg=args.cfg,
        heights=[args.height] * len(prompts),
        widths=[args.width] * len(prompts),
        static_shift=args.shift,
        renormalization=args.renorm,
        progress=not args.quiet,
    )

    os.makedirs(args.out, exist_ok=True)
    for i, im in enumerate(imgs):
        path = os.path.join(args.out, f"mage_{i:03d}.png")
        im.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()

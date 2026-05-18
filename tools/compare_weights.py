"""
Compare L1 and L2 weight similarity along the Anima training progression:
  Cosmos2 → Preview1 → Preview2 → Preview3 → Base v1.0

Cosmos2 only contains DiT weights; the Anima checkpoints additionally carry
an LLMAdapter. cosmos2.safetensors uses the original Cosmos2 key naming
scheme; Anima checkpoints use internal naming (net./diffusion_model. prefix
stripped before compare). A converter normalises cosmos2 keys to Anima
internal naming before comparison.
"""

import re
import torch
import safetensors.torch as st
from pathlib import Path


COSMOS2  = Path('/home/bluvoll/diffusion-pipe/cosmos2.safetensors')
PREVIEW1 = Path('/home/bluvoll/ComfyUI/models/diffusion_models/anima-preview.safetensors')
PREVIEW2 = Path('/home/bluvoll/ComfyUI/models/diffusion_models/anima-preview2.safetensors')
PREVIEW3 = Path('/home/bluvoll/ComfyUI/models/diffusion_models/anima-preview3-base.safetensors')
BASE     = Path('/home/bluvoll/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors')


# ---------------------------------------------------------------------------
# Key conversion: Cosmos2 format → Anima internal format
# ---------------------------------------------------------------------------

_GLOBAL_MAP = {
    'patch_embed.proj.weight':              'x_embedder.proj.1.weight',
    'proj_out.weight':                       'final_layer.linear.weight',
    'norm_out.linear_1.weight':              'final_layer.adaln_modulation.1.weight',
    'norm_out.linear_2.weight':              'final_layer.adaln_modulation.2.weight',
    'time_embed.norm.weight':                't_embedding_norm.weight',
    'time_embed.t_embedder.linear_1.weight': 't_embedder.1.linear_1.weight',
    'time_embed.t_embedder.linear_2.weight': 't_embedder.1.linear_2.weight',
}

_ATTN1_MAP = {
    'to_q.weight':     'self_attn.q_proj.weight',
    'to_k.weight':     'self_attn.k_proj.weight',
    'to_v.weight':     'self_attn.v_proj.weight',
    'to_out.0.weight': 'self_attn.output_proj.weight',
    'norm_q.weight':   'self_attn.q_norm.weight',
    'norm_k.weight':   'self_attn.k_norm.weight',
}

_ATTN2_MAP = {
    'to_q.weight':     'cross_attn.q_proj.weight',
    'to_k.weight':     'cross_attn.k_proj.weight',
    'to_v.weight':     'cross_attn.v_proj.weight',
    'to_out.0.weight': 'cross_attn.output_proj.weight',
    'norm_q.weight':   'cross_attn.q_norm.weight',
    'norm_k.weight':   'cross_attn.k_norm.weight',
}

_ADALN_MAP = {
    'norm1': 'adaln_modulation_self_attn',
    'norm2': 'adaln_modulation_cross_attn',
    'norm3': 'adaln_modulation_mlp',
}

_FF_MAP = {
    'ff.net.0.proj.weight': 'mlp.layer1.weight',
    'ff.net.2.weight':      'mlp.layer2.weight',
}

_BLOCK_RE = re.compile(r'^transformer_blocks\.(\d+)\.(.*)')


def _convert_cosmos2_key(key: str) -> str | None:
    """Convert a cosmos2 key to Anima internal naming. Returns None if unknown."""
    if key in _GLOBAL_MAP:
        return _GLOBAL_MAP[key]

    m = _BLOCK_RE.match(key)
    if not m:
        return None
    n, rest = m.group(1), m.group(2)
    prefix = f'blocks.{n}.'

    # attn1 (self-attention)
    if rest.startswith('attn1.'):
        sub = rest[len('attn1.'):]
        if sub in _ATTN1_MAP:
            return prefix + _ATTN1_MAP[sub]

    # attn2 (cross-attention)
    if rest.startswith('attn2.'):
        sub = rest[len('attn2.'):]
        if sub in _ATTN2_MAP:
            return prefix + _ATTN2_MAP[sub]

    # MLP
    if rest in _FF_MAP:
        return prefix + _FF_MAP[rest]

    # AdaLN modulation (norm1/norm2/norm3)
    for norm_key, adaln_name in _ADALN_MAP.items():
        if rest.startswith(norm_key + '.'):
            sub = rest[len(norm_key) + 1:]          # e.g. "linear_1.weight"
            sub = sub.replace('linear_1', '1').replace('linear_2', '2')
            return prefix + f'{adaln_name}.{sub}'

    return None


def cosmos2_to_anima(sd: dict) -> dict:
    """Convert a cosmos2 state dict to Anima internal naming."""
    out = {}
    unknown = []
    for k, v in sd.items():
        new_k = _convert_cosmos2_key(k)
        if new_k is None:
            unknown.append(k)
        else:
            out[new_k] = v
    if unknown:
        print(f'  [WARN] {len(unknown)} cosmos2 keys had no mapping: {unknown[:5]}')
    return out


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_anima(path: Path) -> dict:
    """Load an Anima checkpoint, stripping net./diffusion_model. prefixes."""
    sd = st.load_file(str(path), device='cpu')
    out = {}
    for k, v in sd.items():
        if k.startswith('net.'):
            k = k[4:]
        elif k.startswith('diffusion_model.'):
            k = k[len('diffusion_model.'):]
        out[k] = v
    return out


def split_llm(sd: dict) -> tuple[dict, dict]:
    dit = {k: v for k, v in sd.items() if not k.startswith('llm_adapter.')}
    llm = {k: v for k, v in sd.items() if k.startswith('llm_adapter.')}
    return dit, llm


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

_BLOCK_KEY_RE = re.compile(r'^blocks\.(\d+)\.')
_LLMADAPTER_KEY_RE = re.compile(r'^llm_adapter\.blocks\.(\d+)\.')


def _bucket_by_layer(common, sd_a, sd_b):
    """Return per-layer stats: {layer_idx: (total_l1_diff, total_abs_a, n_elements, sum_l2)}."""
    layers = {}
    for key in common:
        m = _BLOCK_KEY_RE.match(key) or _LLMADAPTER_KEY_RE.match(key)
        if m is None:
            idx = -1  # global / non-block tensors
        else:
            idx = int(m.group(1))
        a = sd_a[key].float()
        b = sd_b[key].float()
        if a.shape != b.shape:
            continue
        diff = (a - b).abs()
        l1   = diff.sum().item()
        l2   = diff.pow(2).sum().sqrt().item()
        n    = a.numel()
        abs_a = a.abs().sum().item()
        if idx not in layers:
            layers[idx] = [0.0, 0.0, 0, 0.0]
        layers[idx][0] += l1
        layers[idx][1] += abs_a
        layers[idx][2] += n
        layers[idx][3] += l2
    return layers


def compare(sd_a: dict, sd_b: dict, label: str):
    keys_a, keys_b = set(sd_a), set(sd_b)
    common = sorted(keys_a & keys_b)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Common keys : {len(common)}")
    if only_a:
        print(f"  Only in A   : {len(only_a)}  e.g. {next(iter(sorted(only_a)))}")
    if only_b:
        print(f"  Only in B   : {len(only_b)}  e.g. {next(iter(sorted(only_b)))}")

    if not common:
        print("  (nothing to compare)")
        return

    total_l1 = 0.0
    total_abs_a = 0.0
    total_elements = 0
    per_tensor = []

    for key in common:
        a = sd_a[key].float()
        b = sd_b[key].float()
        if a.shape != b.shape:
            print(f"  [SHAPE MISMATCH] {key}: {a.shape} vs {b.shape}")
            continue
        diff = (a - b).abs()
        l1 = diff.sum().item()
        l2 = diff.pow(2).sum().sqrt().item()
        n  = a.numel()
        total_l1 += l1
        total_abs_a += a.abs().sum().item()
        total_elements += n
        per_tensor.append((l2, key))

    mean_l1 = total_l1 / total_elements if total_elements else 0
    mean_l2 = sum(x[0] for x in per_tensor) / len(per_tensor) if per_tensor else 0
    rel_l1  = total_l1 / total_abs_a * 100 if total_abs_a else 0

    print(f"\n  Tensors compared : {len(per_tensor):,}  ({total_elements:,} elements)")
    print(f"  Mean |diff| per element (L1) : {mean_l1:.6f}")
    print(f"  Mean L2 norm per tensor      : {mean_l2:.4f}")
    print(f"  Total L1 sum                 : {total_l1:,.1f}")
    print(f"  Relative L1 change           : {rel_l1:.2f}%")

    per_tensor.sort(reverse=True)
    top_k = per_tensor[0]
    print(f"  Most diverged (L2)           : {top_k[1]}  ({top_k[0]:.4f})")

    print(f"\n  Top 10 most changed tensors (L2):")
    for l2, key in per_tensor[:10]:
        print(f"    {l2:10.4f}  {key}")

    # Per-layer breakdown
    layers = _bucket_by_layer(common, sd_a, sd_b)
    block_idxs = sorted(k for k in layers if k >= 0)
    if block_idxs:
        print(f"\n  Per-layer breakdown  (layer | rel L1% | sum L2 | mean L1/elem):")
        print(f"  {'Layer':>6}  {'Rel L1%':>8}  {'Sum L2':>10}  {'Mean L1':>10}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}")
        for idx in block_idxs:
            l1_diff, abs_a, n, sum_l2 = layers[idx]
            rel = l1_diff / abs_a * 100 if abs_a else 0
            mean = l1_diff / n if n else 0
            print(f"  {idx:>6}  {rel:>7.2f}%  {sum_l2:>10.2f}  {mean:>10.6f}")
        if -1 in layers:
            l1_diff, abs_a, n, sum_l2 = layers[-1]
            rel = l1_diff / abs_a * 100 if abs_a else 0
            mean = l1_diff / n if n else 0
            print(f"  {'global':>6}  {rel:>7.2f}%  {sum_l2:>10.2f}  {mean:>10.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading weights...")
    cosmos_raw = st.load_file(str(COSMOS2), device='cpu')
    cosmos     = cosmos2_to_anima(cosmos_raw)

    p1   = load_anima(PREVIEW1)
    p2   = load_anima(PREVIEW2)
    p3   = load_anima(PREVIEW3)
    base = load_anima(BASE)

    p1_dit,   p1_llm   = split_llm(p1)
    p2_dit,   p2_llm   = split_llm(p2)
    p3_dit,   p3_llm   = split_llm(p3)
    base_dit, base_llm = split_llm(base)

    print(f"  Cosmos2  (converted) : {len(cosmos)} keys")
    print(f"  Preview1             : {len(p1_dit)} DiT + {len(p1_llm)} LLMAdapter")
    print(f"  Preview2             : {len(p2_dit)} DiT + {len(p2_llm)} LLMAdapter")
    print(f"  Preview3             : {len(p3_dit)} DiT + {len(p3_llm)} LLMAdapter")
    print(f"  Base v1.0            : {len(base_dit)} DiT + {len(base_llm)} LLMAdapter")

    # Drift from Cosmos2 baseline (DiT only)
    compare(cosmos,   p1_dit,   "Cosmos2  →  Anima Preview 1  (DiT only)")
    compare(cosmos,   p2_dit,   "Cosmos2  →  Anima Preview 2  (DiT only)")
    compare(cosmos,   p3_dit,   "Cosmos2  →  Anima Preview 3  (DiT only)")
    compare(cosmos,   base_dit, "Cosmos2  →  Anima Base v1.0  (DiT only)")

    # Sequential progression — DiT
    compare(p1_dit,   p2_dit,   "Preview1 →  Preview 2        (DiT only)")
    compare(p2_dit,   p3_dit,   "Preview2 →  Preview 3        (DiT only)")
    compare(p3_dit,   base_dit, "Preview3 →  Base v1.0        (DiT only)")

    # Sequential progression — LLMAdapter
    compare(p1_llm,   p2_llm,   "Preview1 →  Preview 2        (LLMAdapter only)")
    compare(p2_llm,   p3_llm,   "Preview2 →  Preview 3        (LLMAdapter only)")
    compare(p3_llm,   base_llm, "Preview3 →  Base v1.0        (LLMAdapter only)")


if __name__ == '__main__':
    main()

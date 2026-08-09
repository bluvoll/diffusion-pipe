"""diffusion-pipe training pipeline for MageFlow (Mage-Flow).

Mage-Flow = modified Flux2-style VAE (MageVAE, 16x downsample, 128 latent
channels) + a 12-block double-stream MMDiT (MageFlow) + Qwen3-VL text
conditioning. This module wires the pretrained inference model (vendored under
``Mage/mage_flow``) into diffusion-pipe for LoRA / LyCORIS and full fine-tuning
of the transformer (VAE and text encoder are frozen).

Design notes
------------
The upstream MageFlow transformer runs a *packed varlen* forward
(``flash_attn_varlen_func`` + ``cu_seqlens``, batch dim forced to 1). diffusion-pipe
feeds genuine batched, single-resolution tensors (one size bucket per batch) and
splits the model across DeepSpeed pipeline stages, so we drive the same
pretrained weights through a *batched SDPA* double-stream forward instead. Since
every image in a size-bucket batch shares a resolution, image RoPE / QK-norm /
adaLN modulation reproduce exactly under SDPA — the only difference from the
varlen kernel is padding + a key-padding mask on the variable-length text stream
(identical to how models/qwen_image.py handles its Qwen double-stream blocks).
This also means training carries no hard flash-attn dependency.

Only text-to-image is implemented here (LoRA + full finetune). The upstream
image-edit / multi-reference path and the inference-time content screening +
watermarking are intentionally not part of the training graph.
"""

import os
import sys
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from safetensors.torch import load_file
from einops import rearrange
import transformers

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE
from utils.offloading import ModelOffloader
from utils import caption_processing as capproc


# Make the vendored Mage package importable (Mage/mage_flow -> `import mage_flow`).
# We import the bare nn.Module / VAE directly and avoid mage_flow/__init__.py,
# which pulls the full inference pipeline (content screening, watermarking,
# gradio) that training does not need.
_MAGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Mage')
if _MAGE_ROOT not in sys.path:
    sys.path.insert(0, _MAGE_ROOT)

from mage_flow.models.mage_flow import MageFlow, MageFlowParams  # noqa: E402
from mage_flow.models.modules.mage_vae import MageVAE  # noqa: E402


# Keep the projection / embedding / norm layers in high precision; only the
# 12 transformer blocks run in the (optionally) lower transformer_dtype.
KEEP_IN_HIGH_PRECISION = ['time_text_embed', 'img_in', 'txt_in', 'txt_norm', 'norm_out', 'proj_out', 'pos_embed']

# Qwen3-VL text-conditioning template (mage-flow t2i). start_idx = number of
# leading system-prompt tokens to drop from the encoded sequence.
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34


def _apply_rope_batched(x, freqs_complex):
    """Apply MageFlow 2D multi-scale RoPE to a batched tensor.

    Args:
        x: [B, H, L, Dh]
        freqs_complex: [L, Dh//2] complex
    """
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # [B, H, L, Dh/2]
    freqs = freqs_complex.view(1, 1, freqs_complex.shape[0], freqs_complex.shape[1])
    x_out = torch.view_as_real(x_c * freqs).flatten(-2)  # [B, H, L, Dh]
    return x_out.type_as(x)


def _modulate(x, mod):
    """adaLN modulation for a [B, L, D] tensor from [B, 3*D] params.

    Returns (modulated_x, gate) with gate shaped [B, 1, D] for the residual.
    """
    shift, scale, gate = mod.chunk(3, dim=-1)  # each [B, D]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)


def _double_stream_block_forward(block, hidden_states, encoder_hidden_states, temb,
                                 img_freqs, attn_mask, num_heads):
    """Batched (SDPA) reimplementation of MageFlowTransformerBlock.forward.

    Reuses the block's pretrained submodules but drives them with real batched
    [B, L, D] tensors and a joint [text, image] SDPA (padded text + key mask)
    instead of the upstream varlen path. Returns (encoder_hidden_states,
    hidden_states) to match the upstream (txt, img) return order.
    """
    attn = block.attn

    img_mod1, img_mod2 = block.img_mod(temb).chunk(2, dim=-1)  # each [B, 3*dim]
    txt_mod1, txt_mod2 = block.txt_mod(temb).chunk(2, dim=-1)

    # --- norm1 + modulation ---
    img_modulated, img_gate1 = _modulate(block.img_norm1(hidden_states), img_mod1)
    txt_modulated, txt_gate1 = _modulate(block.txt_norm1(encoder_hidden_states), txt_mod1)

    # --- joint attention (order: [text, image]) ---
    B, Li, _ = img_modulated.shape
    Lt = txt_modulated.shape[1]

    def _proj(x, q_proj, k_proj, v_proj, nq, nk):
        q = q_proj(x).unflatten(-1, (num_heads, -1))
        k = k_proj(x).unflatten(-1, (num_heads, -1))
        v = v_proj(x).unflatten(-1, (num_heads, -1))
        if nq is not None:
            q = nq(q)
        if nk is not None:
            k = nk(k)
        # [B, L, H, Dh] -> [B, H, L, Dh]
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    iq, ik, iv = _proj(img_modulated, attn.to_q, attn.to_k, attn.to_v, attn.norm_q, attn.norm_k)
    tq, tk, tv = _proj(txt_modulated, attn.add_q_proj, attn.add_k_proj, attn.add_v_proj,
                       attn.norm_added_q, attn.norm_added_k)

    # RoPE on image tokens only (text is not rotated in MageFlow).
    iq = _apply_rope_batched(iq, img_freqs)
    ik = _apply_rope_batched(ik, img_freqs)

    q = torch.cat([tq, iq], dim=2)
    k = torch.cat([tk, ik], dim=2)
    v = torch.cat([tv, iv], dim=2)

    # softmax_scale=None upstream -> flash default 1/sqrt(head_dim); SDPA default matches.
    joint = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    joint = joint.transpose(1, 2).flatten(2)  # [B, Lt+Li, dim]

    txt_attn_output = attn.to_add_out(joint[:, :Lt])
    img_attn_output = attn.to_out[0](joint[:, Lt:])
    img_attn_output = attn.to_out[1](img_attn_output)  # dropout (no-op in eval/train p=0)

    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

    # --- norm2 + MLP ---
    img_modulated2, img_gate2 = _modulate(block.img_norm2(hidden_states), img_mod2)
    hidden_states = hidden_states + img_gate2 * block.img_mlp(img_modulated2)

    txt_modulated2, txt_gate2 = _modulate(block.txt_norm2(encoder_hidden_states), txt_mod2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.txt_mlp(txt_modulated2)

    return encoder_hidden_states, hidden_states


class MageFlowPipeline(BasePipeline):
    name = 'mage_flow'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['MageFlowTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        dtype = self.model_config['dtype']

        self.preprocess_media_file_fn = PreprocessMediaFile(
            self.config, support_video=False, round_height=16, round_width=16)

        self.max_text_tokens = self.model_config.get('max_text_tokens', 512)

        # Text-embedding caching vs on-the-fly encoding. Caching is fast and
        # low-VRAM but freezes each caption's embedding, so per-step caption
        # augmentation (tag/sentence shuffle, dropout, tags/NL mixing) requires
        # cache_text_embeddings=false (Qwen3-VL stays resident and re-encodes
        # every step; costs ~the text encoder's size in extra VRAM).
        self.cache_text_embeddings = self.model_config.get('cache_text_embeddings', True)
        self.caption_config = capproc.build_caption_config(self.model_config)
        capproc.validate_caption_config(self.caption_config)
        self.protected_tags = capproc.load_protected_tags(self.model_config.get('protected_tags_file', None))
        if self.protected_tags:
            print(f"Loaded {len(self.protected_tags)} protected tags")
        self.caption_debug_state = {}
        self.caption_sample_idx = 0

        if self.cache_text_embeddings and capproc.caption_config_needs_on_the_fly(self.caption_config):
            print("WARNING: per-step caption augmentation (tag/sentence shuffle, tag dropout, "
                  "caption dropout) requires cache_text_embeddings=false. With caching on, these "
                  "options have no effect. (Tags/NL mixing via caption_mode DOES work cached.)")
        if self.cache_text_embeddings and self._uses_nl_variants():
            print(f"Cached tags/NL mixing enabled (caption_mode='{self.caption_config['caption_mode']}'): "
                  "each variant is encoded once and weighted-selected per step.")

        diffusers_path = self.model_config.get('diffusers_path', None)

        def _sub(key, *rel):
            if key in self.model_config:
                return self.model_config[key]
            if diffusers_path is None:
                raise ValueError(f"Config needs '{key}' or 'diffusers_path'")
            return str(Path(diffusers_path).joinpath(*rel))

        text_encoder_path = _sub('text_encoder_path', 'text_encoder')
        vae_path = _sub('vae_path', 'vae', 'diffusion_pytorch_model.safetensors')

        # Frozen text encoder (Qwen3-VL) + tokenizer.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(text_encoder_path)
        self.text_encoder = self._load_text_encoder(text_encoder_path, dtype)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # Frozen VAE (MageVAE). Deterministic (posterior mean) by default so
        # cached latents are stable across epochs.
        sample_posterior = self.model_config.get('vae_sample_posterior', False)
        self.vae = MageVAE(ckpt_path=vae_path, sample_posterior=sample_posterior)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(dtype)

    def _resolve_te_quant(self):
        """Resolve the text-encoder quantization request to '', 'fp8' or 'nf4'.

        Quantization only helps in on-the-fly mode (encoder stays resident
        alongside the DiT). In cache mode the encoder is transient, so a
        requested quant is ignored (with a note).
        """
        quant = (self.model_config.get('text_encoder_quant') or '').lower()
        if self.model_config.get('text_encoder_nf4', False):
            quant = 'nf4'
        elif self.model_config.get('text_encoder_fp8', False):
            quant = 'fp8'
        if quant and quant not in ('fp8', 'nf4'):
            raise ValueError(f"text_encoder_quant must be 'fp8' or 'nf4', got {quant!r}")
        if quant and self.cache_text_embeddings:
            print(f"Note: text_encoder_quant='{quant}' ignored with cache_text_embeddings=true "
                  "(the encoder is only transient during caching).")
            return ''
        return quant

    @staticmethod
    def _repo_is_prequantized(path):
        """True if the HF repo's config.json already declares a quantization
        (bnb / GPTQ / AWQ / fp8). Such repos load low-bit directly via
        from_pretrained; we must not layer our own quantization on top."""
        cfg_path = os.path.join(path, 'config.json') if os.path.isdir(path) else None
        if not cfg_path or not os.path.exists(cfg_path):
            return False
        try:
            import json
            return 'quantization_config' in json.load(open(cfg_path))
        except Exception:
            return False

    def _load_text_encoder(self, path, dtype):
        quant = self._resolve_te_quant()

        # If the checkpoint is already quantized, load it as-is (no flags).
        if self._repo_is_prequantized(path):
            if quant:
                print(f"Note: text_encoder is a pre-quantized repo; ignoring "
                      f"text_encoder_quant='{quant}' and loading its native quantization.")
            te = transformers.Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype)
            print("text_encoder: loaded pre-quantized checkpoint (native quantization_config)")
            return te

        quantization_config = None
        if quant == 'nf4':
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        te = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=dtype, quantization_config=quantization_config)

        # fp8: store nn.Linear weights as float8_e4m3fn to ~halve resident
        # memory. Autocast upcasts a Linear's fp8 weight to the compute dtype
        # for the matmul, so the frozen forward is numerically unaffected.
        # Embeddings are deliberately left in dtype: autocast does NOT upcast
        # F.embedding output, so an fp8 embed_tokens would emit fp8 activations
        # and start the whole text representation from fp8-rounded values.
        if quantization_config is None and quant == 'fp8':
            n = 0
            for _, mod in te.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    mod.weight.data = mod.weight.data.to(torch.float8_e4m3fn)
                    n += 1
            print(f"text_encoder fp8: cast {n} Linear weights to float8_e4m3fn")
        elif quant == 'nf4':
            print("text_encoder nf4: loaded 4-bit (bitsandbytes, double-quant)")
        return te

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        transformer_path = self.model_config.get('transformer_path', None)
        if transformer_path is None:
            diffusers_path = self.model_config['diffusers_path']
            transformer_path = str(Path(diffusers_path) / 'transformer' / 'diffusion_pytorch_model.safetensors')

        params = self._transformer_params()
        transformer = MageFlow(params)

        sd = load_file(transformer_path, device='cpu')
        missing, unexpected = transformer.load_state_dict(sd, strict=False, assign=True)
        if missing:
            print(f'MageFlow load: {len(missing)} missing keys (e.g. {missing[:3]})')
        if unexpected:
            print(f'MageFlow load: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})')

        # Cast: blocks -> transformer_dtype, everything else / 1D -> dtype.
        for name, p in transformer.named_parameters():
            keep = any(k in name for k in KEEP_IN_HIGH_PRECISION) or p.ndim == 1
            p.data = p.data.to(dtype if keep else transformer_dtype)

        self.transformer = transformer
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def _transformer_params(self):
        import json
        transformer_path = self.model_config.get('transformer_path', None)
        if transformer_path is not None:
            cfg_path = Path(transformer_path).parent / 'config.json'
        else:
            cfg_path = Path(self.model_config['diffusers_path']) / 'transformer' / 'config.json'
        with open(cfg_path) as f:
            c = json.load(f)
        return MageFlowParams(
            in_channels=c['in_channels'],
            out_channels=c['out_channels'],
            context_in_dim=c['context_in_dim'],
            hidden_size=c['hidden_size'],
            num_heads=c['num_heads'],
            depth=c['depth'],
            axes_dim=list(c['axes_dim']),
            checkpoint=False,  # diffusion-pipe drives its own activation checkpointing
            patch_size=c.get('patch_size', 1),
        )

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        # Empty in on-the-fly mode so the framework does NOT cache text
        # embeddings; the encoder is kept resident and run in InitialLayer.
        return [self.text_encoder] if self.cache_text_embeddings else []

    def get_preprocess_media_file_fn(self):
        return self.preprocess_media_file_fn

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI-style key prefix.
        peft_state_dict = {'diffusion_model.' + k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'diffusion_pytorch_model.safetensors', metadata={'format': 'pt'})

    def get_call_vae_fn(self, vae):
        def fn(*args):
            image = args[0]
            if image.ndim == 5:
                # [B, C, F, H, W] -> [B, C, H, W] (image model, F == 1)
                image = image.squeeze(2)
            latents = vae.encode(image.to(vae.device, vae.dtype))  # [B, 128, h, w]
            return {'latents': latents}
        return fn

    def _uses_nl_variants(self):
        return self.caption_config.get('caption_mode', 'tags') in ('mixed', 'nl')

    # Keys under which each cached caption variant's embedding is stored.
    _VARIANT_KEYS = {
        'tags': 'prompt_embeds',
        'nl': 'prompt_embeds_nl',
        'tags_nl': 'prompt_embeds_tags_nl',
        'nl_tags': 'prompt_embeds_nl_tags',
    }

    def get_call_text_encoder_fn(self, text_encoder):
        # Cached tags/NL mixing: encode each discrete variant once so a variant
        # can be weighted-selected per step from cache (no resident encoder).
        # `image_spec` in the signature signals the framework to pass it here.
        if self.cache_text_embeddings and self._uses_nl_variants():
            def fn(caption, is_video, image_spec):
                assert not any(is_video)
                if isinstance(caption, str):
                    caption, image_spec = [caption], [image_spec]
                tags_l, nl_l, tagsnl_l, nltags_l = [], [], [], []
                for cap, spec in zip(caption, image_spec):
                    tags = cap or ''
                    spec_t = tuple(spec) if spec is not None else None
                    nl = capproc._load_nl_caption(spec_t) or ''
                    has = bool(nl.strip())
                    tags_l.append(tags)
                    # Fall back to tags when no NL, so all variant columns stay
                    # populated and any selection degrades gracefully to tags.
                    nl_l.append(nl if has else tags)
                    tagsnl_l.append(f'{tags}. {nl}' if has else tags)
                    nltags_l.append(f'{nl}. {tags}' if has else tags)
                dev = text_encoder.device
                return {
                    'prompt_embeds': self._encode_prompts(tags_l, device=dev),
                    'prompt_embeds_nl': self._encode_prompts(nl_l, device=dev),
                    'prompt_embeds_tags_nl': self._encode_prompts(tagsnl_l, device=dev),
                    'prompt_embeds_nl_tags': self._encode_prompts(nltags_l, device=dev),
                }
            return fn

        def fn(caption, is_video):
            assert not any(is_video)
            prompt_embeds = self._encode_prompts(caption, device=text_encoder.device)
            return {'prompt_embeds': prompt_embeds}
        return fn

    @torch.no_grad()
    def _encode_prompts(self, prompts, device=None):
        device = device or self.text_encoder.device
        if isinstance(prompts, str):
            prompts = [prompts]
        drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
        txt = [PROMPT_TEMPLATE_ENCODE.format(p) for p in prompts]
        tokens = self.tokenizer(
            txt,
            max_length=self.max_text_tokens + drop_idx,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(device)
        outputs = self.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, L, 2560]
        # Drop system-prompt tokens and any right padding, per sample.
        embeds = []
        for h, m in zip(hidden, tokens.attention_mask):
            valid = int(m.sum().item())
            embeds.append(h[drop_idx:valid])
        return embeds

    def _pad_cached_embeds(self, prompt_embeds, device):
        """Cache mode: pad the list of per-caption embeds + build a key mask."""
        bs = len(prompt_embeds)
        seq_lens = [e.size(0) for e in prompt_embeds]
        max_len = max(seq_lens)
        dim_txt = prompt_embeds[0].size(1)
        txt = torch.zeros(bs, max_len, dim_txt, device=device, dtype=prompt_embeds[0].dtype)
        txt_mask = torch.zeros(bs, max_len, dtype=torch.bool, device=device)
        for i, e in enumerate(prompt_embeds):
            txt[i, :e.size(0)] = e.to(device)
            txt_mask[i, :e.size(0)] = True
        return txt, txt_mask

    def _select_mixed_embeds(self, inputs):
        """Cached mixing: per sample, weighted-pick a tags/NL variant's cached
        embedding using mixed_weights. Returns a list of [Li, 2560] tensors."""
        tags = inputs['prompt_embeds']
        bs = len(tags)
        mode = self.caption_config.get('caption_mode', 'mixed')
        weights = self.caption_config.get('mixed_weights', capproc.DEFAULT_MIXED_WEIGHTS)
        chosen = []
        for i in range(bs):
            # has_nl_caption=True: no-NL samples fall back to tags at cache time,
            # so any variant already resolves to tags for them.
            variant = capproc._select_variant(mode, weights, has_nl_caption=True)
            key = self._VARIANT_KEYS.get(variant, 'prompt_embeds')
            seq = inputs.get(key)
            chosen.append((seq[i] if seq is not None else tags[i]))
            vk = f'variant_{variant}'
            self.caption_debug_state[vk] = self.caption_debug_state.get(vk, 0) + 1
        self.caption_sample_idx += bs
        capproc.log_caption_stats(self.caption_debug_state, self.caption_sample_idx)
        return chosen

    def _tokenize_on_the_fly(self, inputs, device):
        """On-the-fly mode: augment captions per step, template + tokenize.

        Returns (input_ids [B, Lt] long, attention_mask [B, Lt] long). The
        Qwen3-VL forward + system-prompt drop happen later, in InitialLayer.
        """
        captions = inputs['caption']
        if isinstance(captions, str):
            captions = [captions]
        image_specs = inputs.get('image_spec', None)
        aug = []
        for i, cap in enumerate(captions):
            spec = image_specs[i] if image_specs else (None, None)
            aug.append(capproc.process_caption(
                cap, spec, self.caption_config, self.protected_tags,
                self.caption_sample_idx, self.caption_debug_state))
            self.caption_sample_idx += 1
        capproc.log_caption_stats(self.caption_debug_state, self.caption_sample_idx)

        drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
        txts = [PROMPT_TEMPLATE_ENCODE.format(c) for c in aug]
        tokens = self.tokenizer(
            txts, max_length=self.max_text_tokens + drop_idx,
            padding=True, truncation=True, return_tensors='pt')
        return tokens.input_ids.to(device), tokens.attention_mask.to(device)

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()          # [B, 128, h, w]
        mask = inputs['mask']
        device = latents.device
        bs, channels, h, w = latents.shape

        # Text stream: cached embeds (optionally weighted tags/NL mixing), or
        # token ids to encode on-the-fly (enables per-step shuffle/dropout).
        if self.cache_text_embeddings and self._uses_nl_variants():
            embeds = self._select_mixed_embeds(inputs)
            text0, text1 = self._pad_cached_embeds(embeds, device)
        elif self.cache_text_embeddings:
            text0, text1 = self._pad_cached_embeds(inputs['prompt_embeds'], device)
        else:
            text0, text1 = self._tokenize_on_the_fly(inputs, device)

        # Flow-matching sampling (matches models/qwen_image.py conventions).
        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')
        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError(timestep_sample_method)

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = torch.sigmoid(t * sigmoid_scale)

        # MageFlow uses the z-image static-shift schedule (default shift 6.0).
        shift = self.model_config.get('shift', 6.0)
        if shift and shift != 1.0:
            t = (t * shift) / (1 + (shift - 1) * t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        # Cosine optimal transport: reorder noise so each latent pairs with its
        # most-similar noise sample (straighter flow trajectories). No-op at bs=1.
        if self.model_config.get('flow_use_ot', False) and bs > 1:
            from utils.optimal_transport import ot_reorder_noise
            x_0 = ot_reorder_noise(x_1, x_0)
        t_exp = t.view(-1, 1, 1, 1)
        x_t = (1 - t_exp) * x_1 + t_exp * x_0
        target = x_0 - x_1

        # To token sequences: MageVAE has patch_size 1, so tokens == spatial pixels.
        img = rearrange(x_t, 'b c h w -> b (h w) c')          # [B, L, 128]
        target = rearrange(target, 'b c h w -> b (h w) c')    # [B, L, 128]
        img_hw = torch.tensor([[h, w]], dtype=torch.int32, device=device).repeat(bs, 1)

        if mask is not None:
            mask = mask.to(device)
            if mask.ndim == 5:
                mask = mask.squeeze(2)  # [B,1,F,h,w] -> [B,1,h,w]
            mask = F.interpolate(mask.float(), size=(h, w), mode='nearest-exact')
            mask = rearrange(mask, 'b 1 h w -> b (h w) 1')
        else:
            mask = torch.empty(0, device=device)

        return (
            (img, text0, text1, t, img_hw),
            (target, mask),
        )

    def to_layers(self):
        transformer = self.transformer
        te = None if self.cache_text_embeddings else self.text_encoder
        layers = [InitialLayer(transformer, text_encoder=te,
                               drop_idx=PROMPT_TEMPLATE_ENCODE_START_IDX)]
        for i, block in enumerate(transformer.transformer_blocks):
            layers.append(TransformerLayer(block, i, transformer.num_attention_heads, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.transformer_blocks
        num_blocks = len(blocks)
        assert blocks_to_swap <= num_blocks - 2, (
            f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap}.')
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True,
            torch.device('cuda'), self.config['reentrant_activation_checkpointing'])
        transformer.transformer_blocks = None
        transformer.to('cuda')
        transformer.transformer_blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model, text_encoder=None, drop_idx=0):
        super().__init__()
        self.img_in = model.img_in
        self.txt_norm = model.txt_norm
        self.txt_in = model.txt_in
        self.time_text_embed = model.time_text_embed
        self.pos_embed = model.pos_embed
        # Resident (frozen) text encoder for on-the-fly mode; None when text
        # embeddings are cached. Registered as a submodule so the pipeline
        # moves it to the stage-0 device.
        self.text_encoder = text_encoder
        self.drop_idx = drop_idx

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        img, text0, text1, timestep, img_hw = inputs

        if self.text_encoder is not None:
            # On-the-fly: text0=input_ids, text1=attention_mask.
            input_ids = text0.long()
            attn = text1.long()
            with torch.no_grad():
                out = self.text_encoder(input_ids=input_ids, attention_mask=attn,
                                        output_hidden_states=True)
                hidden = out.hidden_states[-1]
            # Drop the system-prompt prefix; keep padding (masked out in attn).
            # detach()+requires_grad_ so the stage-boundary activation is a leaf
            # requiring grad (parallels the cached-embeds path), without
            # backprop into the frozen text encoder.
            txt = hidden[:, self.drop_idx:, :].detach().requires_grad_(True)
            txt_mask = attn[:, self.drop_idx:].bool()
        else:
            # Cached: text0=padded embeds, text1=key mask.
            txt = text0
            txt_mask = text1

        hidden_states = self.img_in(img)
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_in(self.txt_norm(txt))
        temb = self.time_text_embed(timestep, hidden_states)

        h = int(img_hw[0, 0].item())
        w = int(img_hw[0, 1].item())
        # Image RoPE freqs [L, Dh//2] (complex); same for every sample in the bucket.
        # Kept COMPLEX (not view_as_real) so it crosses the pipeline boundary as a
        # non-floating-point tensor: DeepSpeed's pipe engine only runs backward
        # through floating-point stage outputs, and this constant has no grad_fn.
        # A real [.,.,2] tensor here would trip "element N ... does not require grad".
        img_freqs = self.pos_embed([(1, h, w)], device=hidden_states.device)

        # Joint [text, image] key-padding mask -> [B, 1, 1, Lt+Li] bool.
        B, Li, _ = hidden_states.shape
        img_keep = torch.ones(B, Li, dtype=torch.bool, device=hidden_states.device)
        key_mask = torch.cat([txt_mask, img_keep], dim=1).view(B, 1, 1, -1)

        return make_contiguous(hidden_states, encoder_hidden_states, key_mask, temb, img_freqs)


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, num_heads, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, key_mask, temb, img_freqs = inputs

        self.offloader.wait_for_block(self.block_idx)
        encoder_hidden_states, hidden_states = _double_stream_block_forward(
            self.block, hidden_states, encoder_hidden_states, temb,
            img_freqs, key_mask, self.num_heads)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, encoder_hidden_states, key_mask, temb, img_freqs)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, key_mask, temb, img_freqs = inputs
        # Batched AdaLayerNormContinuous (upstream cu_seqlens=None branch assumes
        # a flattened layout; do the [B, L, D] modulation explicitly here).
        emb = self.norm_out.linear(self.norm_out.silu(temb).to(hidden_states.dtype))
        scale, shift = emb.chunk(2, dim=-1)
        hidden_states = self.norm_out.norm(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output = self.proj_out(hidden_states)
        return output

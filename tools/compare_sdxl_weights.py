"""
Compare L1 and L2 weight similarity between SDXL single-file checkpoints
(webui/reForge format), split by component:

  UNet    model.diffusion_model.*
  CLIP-L  conditioner.embedders.0.transformer.*
  CLIP-G  conditioner.embedders.1.model.*
  VAE     first_stage_model.*

Usage:
  python tools/compare_sdxl_weights.py A.safetensors B.safetensors [C.safetensors ...]

Consecutive pairs are compared (A→B, B→C, ...). Pass --baseline to instead
compare the first checkpoint against every other one (A→B, A→C, ...).

Components are loaded one at a time with safe_open so peak RAM stays at
roughly 2x the largest component instead of 2x the full checkpoint.
"""

import argparse
import re
import safetensors
from pathlib import Path


COMPONENTS = {
    'UNet':   'model.diffusion_model.',
    'CLIP-L': 'conditioner.embedders.0.transformer.',
    'CLIP-G': 'conditioner.embedders.1.model.',
    'VAE':    'first_stage_model.',
}


# ---------------------------------------------------------------------------
# Per-layer bucketing
# ---------------------------------------------------------------------------

_UNET_BLOCK_RE = re.compile(r'^(input_blocks\.\d+|middle_block|output_blocks\.\d+)\.')
_CLIP_L_RE     = re.compile(r'^text_model\.encoder\.layers\.(\d+)\.')
_CLIP_G_RE     = re.compile(r'^transformer\.resblocks\.(\d+)\.')
_VAE_RE        = re.compile(r'^(encoder\.down\.\d+|encoder\.mid|decoder\.up\.\d+|decoder\.mid)\.')


def _layer_label(component: str, key: str) -> str:
    """Bucket a (prefix-stripped) key into a layer label for the breakdown table."""
    if component == 'UNet':
        m = _UNET_BLOCK_RE.match(key)
        return m.group(1) if m else 'global'
    if component == 'CLIP-L':
        m = _CLIP_L_RE.match(key)
        return f'layer.{int(m.group(1)):02d}' if m else 'global'
    if component == 'CLIP-G':
        m = _CLIP_G_RE.match(key)
        return f'resblock.{int(m.group(1)):02d}' if m else 'global'
    if component == 'VAE':
        m = _VAE_RE.match(key)
        return m.group(1) if m else 'global'
    return 'global'


def _layer_sort_key(label: str):
    """Sort UNet blocks input→middle→output, numbered labels numerically."""
    order = {'input_blocks': 0, 'middle_block': 1, 'output_blocks': 2,
             'encoder': 0, 'decoder': 1}
    parts = label.split('.')
    head = order.get(parts[0], 9)
    nums = tuple(int(p) for p in parts if p.isdigit())
    return (head, label if not nums else '', nums)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_component(path: Path, prefix: str) -> dict:
    """Load only the tensors under `prefix`, with the prefix stripped."""
    out = {}
    with safetensors.safe_open(str(path), framework='pt', device='cpu') as f:
        for k in f.keys():
            if k.startswith(prefix):
                out[k[len(prefix):]] = f.get_tensor(k)
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(sd_a: dict, sd_b: dict, component: str, label: str):
    keys_a, keys_b = set(sd_a), set(sd_b)
    common = sorted(keys_a & keys_b)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    print(f"\n{'='*70}")
    print(f"  [{component}]  {label}")
    print(f"{'='*70}")
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
    layers = {}

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
        abs_a = a.abs().sum().item()

        total_l1 += l1
        total_abs_a += abs_a
        total_elements += n
        per_tensor.append((l2, key))

        lbl = _layer_label(component, key)
        if lbl not in layers:
            layers[lbl] = [0.0, 0.0, 0, 0.0]
        layers[lbl][0] += l1
        layers[lbl][1] += abs_a
        layers[lbl][2] += n
        layers[lbl][3] += l2

    if not per_tensor:
        print("  (no comparable tensors)")
        return

    mean_l1 = total_l1 / total_elements if total_elements else 0
    mean_l2 = sum(x[0] for x in per_tensor) / len(per_tensor)
    rel_l1  = total_l1 / total_abs_a * 100 if total_abs_a else 0

    print(f"\n  Tensors compared : {len(per_tensor):,}  ({total_elements:,} elements)")
    print(f"  Mean |diff| per element (L1) : {mean_l1:.6f}")
    print(f"  Mean L2 norm per tensor      : {mean_l2:.4f}")
    print(f"  Total L1 sum                 : {total_l1:,.1f}")
    print(f"  Relative L1 change           : {rel_l1:.2f}%")

    per_tensor.sort(reverse=True)
    print(f"\n  Top 10 most changed tensors (L2):")
    for l2, key in per_tensor[:10]:
        print(f"    {l2:10.4f}  {key}")

    print(f"\n  Per-layer breakdown  (layer | rel L1% | sum L2 | mean L1/elem):")
    print(f"  {'Layer':>18}  {'Rel L1%':>8}  {'Sum L2':>10}  {'Mean L1':>10}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*10}  {'-'*10}")
    labels = sorted((l for l in layers if l != 'global'), key=_layer_sort_key)
    if 'global' in layers:
        labels.append('global')
    for lbl in labels:
        l1_diff, abs_a, n, sum_l2 = layers[lbl]
        rel = l1_diff / abs_a * 100 if abs_a else 0
        mean = l1_diff / n if n else 0
        print(f"  {lbl:>18}  {rel:>7.2f}%  {sum_l2:>10.2f}  {mean:>10.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare SDXL checkpoints per component.')
    parser.add_argument('checkpoints', nargs='+', type=Path, help='2+ SDXL .safetensors files')
    parser.add_argument('--baseline', action='store_true',
                        help='compare the first checkpoint against every other one '
                             'instead of consecutive pairs')
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        parser.error('need at least two checkpoints')
    for p in args.checkpoints:
        if not p.exists():
            parser.error(f'not found: {p}')

    if args.baseline:
        pairs = [(args.checkpoints[0], p) for p in args.checkpoints[1:]]
    else:
        pairs = list(zip(args.checkpoints, args.checkpoints[1:]))

    for component, prefix in COMPONENTS.items():
        for path_a, path_b in pairs:
            sd_a = load_component(path_a, prefix)
            sd_b = load_component(path_b, prefix)
            compare(sd_a, sd_b, component, f'{path_a.stem}  →  {path_b.stem}')
            del sd_a, sd_b


if __name__ == '__main__':
    main()

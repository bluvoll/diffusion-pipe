"""Convert Anima DiT safetensors between the native `net.` prefix and ComfyUI's
`model.diffusion_model.` prefix.

The weights are identical across anima-base (native) and ComfyUI exports like
Slivani / waiANIMA; only the state-dict key prefix differs. This converts in
either direction, auto-detecting the source format unless one is forced.

Examples:
    # ComfyUI -> native (auto-detected), write a new file
    python tools/convert_anima_prefix.py Slivani-2.0-1280px.safetensors -o slivani-native.safetensors

    # native -> ComfyUI explicitly
    python tools/convert_anima_prefix.py anima-base-v1.0.safetensors --to comfy -o anima-comfy.safetensors

    # in-place conversion of several files (auto-detect each)
    python tools/convert_anima_prefix.py a.safetensors b.safetensors
"""

import argparse
import json
import struct
from pathlib import Path

import safetensors.torch as st

NATIVE_PREFIX = 'net.'
COMFY_PREFIX = 'model.diffusion_model.'


def read_keys(path: Path):
    """Read tensor key names from a safetensors header without loading weights."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop('__metadata__', None)
    return list(hdr.keys())


def detect_format(keys):
    """Return 'comfy' or 'native' based on the key prefixes present."""
    has_comfy = any(k.startswith(COMFY_PREFIX) for k in keys)
    has_native = any(k.startswith(NATIVE_PREFIX) for k in keys) and not has_comfy
    if has_comfy:
        return 'comfy'
    if has_native:
        return 'native'
    return None


def remap_key(key, target):
    if target == 'native':
        if key.startswith(COMFY_PREFIX):
            return NATIVE_PREFIX + key[len(COMFY_PREFIX):]
        return key
    else:  # comfy
        if key.startswith(NATIVE_PREFIX):
            return COMFY_PREFIX + key[len(NATIVE_PREFIX):]
        return key


def convert(path: Path, target: str | None, output: Path | None, keep_metadata: bool):
    keys = read_keys(path)
    src = detect_format(keys)
    if src is None:
        raise SystemExit(f'{path}: could not detect prefix format (no `{NATIVE_PREFIX}` '
                         f'or `{COMFY_PREFIX}` keys found)')

    # default target is the opposite of the detected source
    if target is None:
        target = 'native' if src == 'comfy' else 'comfy'

    if src == target:
        print(f'{path}: already in `{target}` format, nothing to do')
        return

    if output is None:
        output = path

    sd = st.load_file(str(path), device='cpu')
    new_sd = {remap_key(k, target): v for k, v in sd.items()}
    changed = sum(1 for k in sd if remap_key(k, target) != k)

    metadata = {'format': 'pt'}
    if keep_metadata:
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            orig = json.loads(f.read(n)).get('__metadata__')
        if orig:
            metadata = orig

    st.save_file(new_sd, str(output), metadata=metadata)
    print(f'{path}: {src} -> {target} ({changed}/{len(sd)} keys remapped) -> {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Anima DiT safetensors between native (`net.`) and '
                    'ComfyUI (`model.diffusion_model.`) prefixes.')
    parser.add_argument('files', nargs='+', type=Path, help='Safetensors file(s) to convert')
    parser.add_argument('--to', dest='target', choices=['native', 'comfy'], default=None,
                        help='Target format. Default: opposite of the auto-detected source.')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output path (single input only; overwrites input by default).')
    parser.add_argument('--keep-metadata', action='store_true',
                        help='Preserve the original __metadata__ instead of resetting to {"format": "pt"}.')
    args = parser.parse_args()

    if args.output and len(args.files) > 1:
        parser.error('-o/--output can only be used with a single input file')

    for f in args.files:
        convert(f, args.target, args.output, args.keep_metadata)

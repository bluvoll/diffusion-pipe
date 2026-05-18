#!/usr/bin/env python3
"""Strip specific layer types from an Anima LoRA safetensors file.

Usage:
    python3 strip_lora_layers.py input.safetensors output.safetensors --strip mlp
    python3 strip_lora_layers.py input.safetensors output.safetensors --strip mlp adaln_modulation
    python3 strip_lora_layers.py input.safetensors output.safetensors --strip llm_adapter cross_attn
"""

import argparse
import safetensors.torch


def main():
    parser = argparse.ArgumentParser(description='Strip layer types from Anima LoRA')
    parser.add_argument('--input', help='Input safetensors file')
    parser.add_argument('--output', help='Output safetensors file')
    parser.add_argument('--strip', nargs='+', required=True,
                        help='Layer types to strip (e.g. mlp, self_attn, cross_attn, adaln_modulation, llm_adapter)')
    args = parser.parse_args()

    state_dict = safetensors.torch.load_file(args.input, device='cpu')
    total = len(state_dict)

    filtered = {k: v for k, v in state_dict.items()
                if not any(s in k for s in args.strip)}

    stripped = total - len(filtered)
    print(f'Total keys: {total}')
    print(f'Stripped: {stripped} (matching: {", ".join(args.strip)})')
    print(f'Remaining: {len(filtered)}')

    safetensors.torch.save_file(filtered, args.output, metadata={'format': 'pt'})
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()

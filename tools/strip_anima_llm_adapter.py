#!/usr/bin/env python3
"""
Strip LLMAdapter weights from an Anima checkpoint.

This is useful when you want a Cosmos-Predict2-style diffusion checkpoint
without any `llm_adapter.*` tensors.

Supported key styles:
- llm_adapter.*
- diffusion_model.llm_adapter.*
- net.llm_adapter.*
- net.diffusion_model.llm_adapter.*
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import safetensors.torch
import torch
from safetensors import safe_open


def _normalize_model_key(key: str) -> str:
    """Normalize common training/inference prefixes before matching."""
    out = key
    if out.startswith("net."):
        out = out[len("net.") :]
    if out.startswith("diffusion_model."):
        out = out[len("diffusion_model.") :]
    return out


def _is_llm_adapter_key(key: str) -> bool:
    return _normalize_model_key(key).startswith("llm_adapter.")


def _load_safetensors(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors, metadata


def _load_state_dict(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    if path.suffix == ".safetensors":
        return _load_safetensors(path)

    # Non-safetensors fallback
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict dictionary, got {type(state_dict)}")
    return state_dict, {}


def _split_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Iterable[str]]:
    kept: Dict[str, torch.Tensor] = {}
    removed = []
    for key, tensor in state_dict.items():
        if _is_llm_adapter_key(key):
            removed.append(key)
        else:
            kept[key] = tensor
    return kept, removed


def _default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".safetensors"
    return input_path.with_name(f"{input_path.stem}_no_llm_adapter{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove llm_adapter tensors from an Anima checkpoint."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input checkpoint (.safetensors or .pt/.bin)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output checkpoint. Defaults to <input>_no_llm_adapter.safetensors",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be removed; do not write output.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not fail if no llm_adapter keys are found.",
    )
    args = parser.parse_args()

    input_path: Path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    output_path: Path = (
        args.output.expanduser().resolve() if args.output else _default_output_path(input_path)
    )

    if output_path.exists() and not args.force and not args.dry_run:
        raise FileExistsError(
            f"Output exists: {output_path}\nUse --force to overwrite."
        )

    print(f"Loading: {input_path}")
    state_dict, metadata = _load_state_dict(input_path)
    print(f"Loaded tensors: {len(state_dict)}")

    kept, removed = _split_state_dict(state_dict)
    removed = list(removed)

    print(f"LLMAdapter tensors found: {len(removed)}")
    if removed:
        print("First 10 removed keys:")
        for k in removed[:10]:
            print(f"  - {k}")
        if len(removed) > 10:
            print(f"  ... and {len(removed) - 10} more")

    if len(removed) == 0 and not args.allow_empty:
        raise RuntimeError(
            "No llm_adapter keys found. If this is expected, run with --allow-empty."
        )

    if args.dry_run:
        print("Dry run complete. No file written.")
        return

    if output_path.suffix == ".safetensors":
        out_metadata = dict(metadata or {})
        out_metadata["converted_by"] = "strip_anima_llm_adapter.py"
        out_metadata["llm_adapter_removed"] = str(len(removed))
        out_metadata["source_checkpoint"] = str(input_path)
        safetensors.torch.save_file(kept, str(output_path), metadata=out_metadata)
    else:
        torch.save(kept, str(output_path))

    print(f"Saved: {output_path}")
    print(f"Kept tensors: {len(kept)}")


if __name__ == "__main__":
    main()

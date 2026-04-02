#!/usr/bin/env python3
"""
Attach LLMAdapter weights from a donor checkpoint to a Cosmos/Anima-compatible base checkpoint.

Typical use:
- base: cosmos2.safetensors (without llm_adapter)
- donor: anima checkpoint that contains llm_adapter.*
- output: merged checkpoint ready for Anima-style inference path
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import safetensors.torch
import torch
from safetensors import safe_open


def _normalize_model_key(key: str) -> str:
    """Normalize common prefixes to model-internal naming."""
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

    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict dictionary, got {type(state_dict)}")
    return state_dict, {}


def _default_output_path(base_path: Path) -> Path:
    suffix = base_path.suffix or ".safetensors"
    return base_path.with_name(f"{base_path.stem}_with_llm_adapter{suffix}")


def _detect_prefix_style(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("net.diffusion_model.") for k in keys):
        return "net.diffusion_model."
    if any(k.startswith("diffusion_model.") for k in keys):
        return "diffusion_model."
    if any(k.startswith("net.") for k in keys):
        return "net."
    return ""


def _collect_donor_llm_adapter(
    donor_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in donor_state.items():
        norm = _normalize_model_key(key)
        if norm.startswith("llm_adapter."):
            out[norm] = tensor
    return out


def _make_prefixed_key(prefix_style: str, normalized_key: str) -> str:
    return f"{prefix_style}{normalized_key}" if prefix_style else normalized_key


def _count_llm_keys(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(1 for k in state_dict if _is_llm_adapter_key(k))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach llm_adapter tensors from a donor checkpoint to a base checkpoint."
    )
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Path to base checkpoint (e.g. cosmos2.safetensors).",
    )
    parser.add_argument(
        "--donor",
        required=True,
        type=Path,
        help="Path to donor checkpoint that contains llm_adapter weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint path. Defaults to <base>_with_llm_adapter.safetensors",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace llm_adapter keys already present in base checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report operations, do not write output file.",
    )
    args = parser.parse_args()

    base_path = args.base.expanduser().resolve()
    donor_path = args.donor.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve() if args.output else _default_output_path(base_path)
    )

    if not base_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_path}")
    if not donor_path.exists():
        raise FileNotFoundError(f"Donor checkpoint not found: {donor_path}")
    if output_path.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Output exists: {output_path}\nUse --force to overwrite.")

    print(f"Loading base:  {base_path}")
    base_state, base_meta = _load_state_dict(base_path)
    print(f"Loading donor: {donor_path}")
    donor_state, _ = _load_state_dict(donor_path)

    base_llm_before = _count_llm_keys(base_state)
    donor_llm = _collect_donor_llm_adapter(donor_state)
    if len(donor_llm) == 0:
        raise RuntimeError("Donor checkpoint has no llm_adapter keys.")

    prefix_style = _detect_prefix_style(base_state)
    print(f"Detected base key prefix style: '{prefix_style}'")
    print(f"Base llm_adapter keys before merge: {base_llm_before}")
    print(f"Donor llm_adapter keys available:   {len(donor_llm)}")

    merged = dict(base_state)
    added = 0
    replaced = 0
    skipped = 0

    for normalized_key, tensor in donor_llm.items():
        target_key = _make_prefixed_key(prefix_style, normalized_key)
        if target_key in merged and not args.replace_existing:
            skipped += 1
            continue
        if target_key in merged and args.replace_existing:
            replaced += 1
        else:
            added += 1
        merged[target_key] = tensor

    llm_after = _count_llm_keys(merged)
    print(f"Merged results: added={added}, replaced={replaced}, skipped={skipped}")
    print(f"Base llm_adapter keys after merge: {llm_after}")

    if args.dry_run:
        print("Dry run complete. No file written.")
        return

    if output_path.suffix == ".safetensors":
        out_meta = dict(base_meta or {})
        out_meta["converted_by"] = "attach_anima_llm_adapter.py"
        out_meta["llm_adapter_source"] = str(donor_path)
        out_meta["llm_adapter_added"] = str(added)
        out_meta["llm_adapter_replaced"] = str(replaced)
        out_meta["llm_adapter_skipped"] = str(skipped)
        safetensors.torch.save_file(merged, str(output_path), metadata=out_meta)
    else:
        torch.save(merged, str(output_path))

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

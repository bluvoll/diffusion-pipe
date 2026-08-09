import inspect

from lycoris.modules.locon import LoConModule as _LoConModule
from lycoris.modules.loha import LohaModule as _LohaModule
from lycoris.modules.lokr import LokrModule as _LokrModule
from lycoris.modules.glora import GLoRAModule as _GLoRAModule
from lycoris.modules.dylora import DyLoraModule as _DyLoraModule
from lycoris.modules.ia3 import IA3Module as _IA3Module
from lycoris.modules.diag_oft import DiagOFTModule as _DiagOFTModule
from lycoris.modules.boft import ButterflyOFTModule as _BOFTModule

import torch
from torch import nn

# Note: lycoris's FullModule is intentionally omitted. It rewrites org_module.weight
# in place, which breaks our injection pattern (we register alongside, not replace).
# Full fine-tuning is already supported via per-component train_*=true toggles
# in the model config.
LYCORIS_MODULE_CLASSES = {
    'locon':    _LoConModule,
    'loha':     _LohaModule,
    'lokr':     _LokrModule,
    'glora':    _GLoRAModule,
    'dylora':   _DyLoraModule,
    'ia3':      _IA3Module,
    'diag_oft': _DiagOFTModule,
    'boft':     _BOFTModule,
}

LYCORIS_ADAPTER_TYPES = frozenset(LYCORIS_MODULE_CLASSES.keys())

# Adapter config keys consumed by train.py / configure_lycoris() directly.
# Everything else under [adapter] is forwarded to the LyCORIS module ctor when the
# module accepts that kwarg (filtered out otherwise).
_RESERVED_CONFIG_KEYS = {'type', 'rank', 'alpha', 'dropout', 'dtype', 'init_from_existing', 'loraplus_lr_ratio'}

# Known parameter-name patterns per algorithm. Used by is_lycoris_state_dict_key
# for sniffing externally-loaded state_dicts.
LYCORIS_WEIGHT_KEYS = {
    'locon':    ['lora_up.weight', 'lora_down.weight', 'alpha', 'dora_scale'],
    'loha':     ['hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b', 'hada_t1', 'hada_t2', 'alpha', 'dora_scale'],
    'lokr':     ['lokr_w1', 'lokr_w1_a', 'lokr_w1_b', 'lokr_w2', 'lokr_w2_a', 'lokr_w2_b', 'lokr_t1', 'lokr_t2', 'alpha', 'dora_scale'],
    'glora':    ['a1.weight', 'a2.weight', 'b1.weight', 'b2.weight', 'alpha'],
    'dylora':   ['lora_up.weight', 'lora_down.weight', 'alpha'],
    'ia3':      ['weight', 'on_input'],
    'diag_oft': ['oft_blocks', 'oft_diag', 'rescale', 'alpha'],
    'boft':     ['oft_blocks', 'oft_diag', 'rescale', 'boft_b', 'boft_m', 'alpha'],
}


def _filter_extra_kwargs(module_cls, adapter_config):
    sig = inspect.signature(module_cls.__init__)
    valid = set(sig.parameters.keys()) - {'self', 'kwargs'}
    extras = {}
    for k, v in adapter_config.items():
        if k in _RESERVED_CONFIG_KEYS:
            continue
        if k in valid:
            extras[k] = v
    return extras


def configure_lycoris(transformer, adapter_target_modules, adapter_config):
    algo = adapter_config['type']
    if algo not in LYCORIS_MODULE_CLASSES:
        raise ValueError(
            f"Unknown LyCORIS algorithm '{algo}'. Available: {sorted(LYCORIS_MODULE_CLASSES)}"
        )
    module_cls = LYCORIS_MODULE_CLASSES[algo]

    rank = adapter_config['rank']
    alpha = adapter_config['alpha']
    dropout = adapter_config.get('dropout', 0.0)
    extra_kwargs = _filter_extra_kwargs(module_cls, adapter_config)

    targets = []
    for name, module in transformer.named_modules():
        if module.__class__.__name__ not in adapter_target_modules:
            continue
        for full_submodule_name, submodule in module.named_modules(prefix=name):
            if isinstance(submodule, nn.Linear):
                targets.append((full_submodule_name, submodule))

    lycoris_modules = []
    for full_submodule_name, submodule in targets:
        parts = full_submodule_name.split(".")
        child_name = parts[-1]
        parent_path = ".".join(parts[:-1])
        parent = transformer.get_submodule(parent_path)

        lora_name = full_submodule_name.replace(".", "_")

        kwargs = dict(
            lora_name=lora_name,
            org_module=submodule,
            multiplier=1.0,
            lora_dim=rank,
            alpha=alpha,
            dropout=dropout,
            **extra_kwargs,
        )

        lycoris_mod = module_cls(**kwargs)
        lycoris_mod.apply_to()

        register_parent = parent
        suffix_parts = [child_name]
        current_path = parent_path
        # Never register the tracking module into a container that executes its
        # children by iteration (nn.Sequential OR nn.ModuleList) — e.g. diffusers
        # FeedForward.net is a ModuleList, and appending here would run the
        # LyCORIS module as a spurious extra layer. Walk up to a non-container
        # ancestor and register there instead.
        while isinstance(register_parent, (nn.Sequential, nn.ModuleList)):
            if current_path:
                suffix_parts.insert(0, current_path.split(".")[-1])
                current_path = ".".join(current_path.split(".")[:-1])
                register_parent = transformer.get_submodule(current_path) if current_path else transformer
            else:
                register_parent = transformer
                break
        register_name = "_lycoris_" + "_".join(suffix_parts)
        register_parent.add_module(register_name, lycoris_mod)

        for param_name, p in lycoris_mod.named_parameters():
            p.original_name = f"{full_submodule_name}.{param_name}"

        lycoris_modules.append(lycoris_mod)

    return lycoris_modules


def build_lycoris_name_map(transformer):
    name_map = {}
    for name, p in transformer.named_parameters():
        if hasattr(p, 'original_name'):
            name_map[p.original_name] = name
    return name_map


def is_lycoris_state_dict_key(key):
    key_after_prefix = key.split('.', 1)[1] if '.' in key else key
    for suffixes in LYCORIS_WEIGHT_KEYS.values():
        for suffix in suffixes:
            if key_after_prefix.endswith(suffix) or f'.{suffix}' in key:
                return True
    return False

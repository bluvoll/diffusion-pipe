"""Training-time caption augmentation (dependency-free).

Tag shuffle, sentence shuffle, tag dropout, full-caption (CFG) dropout, and
tags / natural-language mixing with protected tags. Ported from the Anima
pipeline so any model can reuse it. Pure stdlib + ``random`` — no torch/model
imports — so it's cheap to import from a data pipeline.

These augmentations re-roll the caption *every step*, so they only take effect
when text embeddings are NOT cached (the model must encode on-the-fly). Call
``process_caption`` per sample inside ``prepare_inputs``.
"""

import random
from pathlib import Path

# Minimum number of tags that must survive dropout.
MIN_SURVIVING_TAGS = 3

# Default sampling weights for caption_mode='mixed'.
DEFAULT_MIXED_WEIGHTS = {'tags': 50, 'nl': 10, 'tags_nl': 20, 'nl_tags': 20}


def build_caption_config(model_config):
    """Extract caption-processing options from a model config dict."""
    return {
        'shuffle_tags': model_config.get('shuffle_tags', False),
        'tag_delimiter': model_config.get('tag_delimiter', ', '),
        'shuffle_keep_first_n': model_config.get('shuffle_keep_first_n', 0),
        'tag_dropout_percent': model_config.get('tag_dropout_percent', 0.0),
        'nl_shuffle_sentences': model_config.get('nl_shuffle_sentences', False),
        'nl_keep_first_sentence': model_config.get('nl_keep_first_sentence', False),
        'caption_dropout_percent': model_config.get('caption_dropout_percent', 0.0),
        'caption_mode': model_config.get('caption_mode', 'tags'),
        'mixed_weights': model_config.get('mixed_weights', DEFAULT_MIXED_WEIGHTS),
        'debug_caption_processing': model_config.get('debug_caption_processing', False),
        'debug_caption_interval': model_config.get('debug_caption_interval', 100),
    }


def validate_caption_config(config):
    caption_mode = config.get('caption_mode', 'tags')
    valid_modes = ['tags', 'nl', 'mixed']
    if caption_mode not in valid_modes:
        raise ValueError(f"caption_mode must be one of {valid_modes}, got '{caption_mode}'")
    dropout = config.get('tag_dropout_percent', 0.0)
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"tag_dropout_percent must be in [0,1], got {dropout}")
    caption_dropout = config.get('caption_dropout_percent', 0.0)
    if not 0.0 <= caption_dropout <= 1.0:
        raise ValueError(f"caption_dropout_percent must be in [0,1], got {caption_dropout}")
    if caption_mode in ['nl', 'mixed']:
        print(f"Note: caption_mode='{caption_mode}' expects {{name}}_nl.txt files. "
              "Samples without NL captions fall back to tags.")


def caption_config_needs_on_the_fly(config):
    """True if any option re-rolls the caption combinatorially per step and so
    cannot be cached. Tags/NL *mixing* (caption_mode) is NOT here: its variants
    form a finite discrete set that can be cached and weighted-selected per step.
    """
    return bool(
        config.get('shuffle_tags')
        or config.get('tag_dropout_percent')
        or config.get('caption_dropout_percent')
        or config.get('nl_shuffle_sentences')
    )


def load_protected_tags(filepath):
    """Load protected tags (one per line, '#' comments allowed) into a set."""
    if not filepath:
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tags = set()
            for line in f:
                tag = line.strip()
                if tag and not tag.startswith('#'):
                    tags.add(tag)
            return tags
    except FileNotFoundError:
        print(f"Warning: protected_tags_file not found: {filepath}")
        return set()
    except Exception as e:
        print(f"Warning: Error loading protected_tags_file: {e}")
        return set()


def _apply_tag_dropout(tags, dropout_percent, protected_indices, protected_tags):
    """Drop a fraction of tags, keeping protected ones and a minimum count."""
    if dropout_percent <= 0 or len(tags) == 0:
        return tags, []

    droppable_indices = []
    for i, tag in enumerate(tags):
        if i in protected_indices:
            continue
        if tag.strip() in protected_tags:
            continue
        droppable_indices.append(i)

    if len(droppable_indices) == 0:
        return tags, []

    num_to_drop = round(len(droppable_indices) * dropout_percent)
    max_droppable = len(tags) - MIN_SURVIVING_TAGS
    num_to_drop = min(num_to_drop, max(0, max_droppable))
    if num_to_drop == 0:
        return tags, []

    drop_indices = set(random.sample(droppable_indices, num_to_drop))
    surviving, dropped = [], []
    for i, tag in enumerate(tags):
        (dropped if i in drop_indices else surviving).append(tag)
    return surviving, dropped


def _process_nl_caption(nl_caption, shuffle_sentences, keep_first_sentence):
    """Optionally shuffle sentences of an NL caption."""
    if not shuffle_sentences or not nl_caption:
        return nl_caption
    sentences = [s.strip() for s in nl_caption.split('. ') if s.strip()]
    if len(sentences) <= 1:
        return nl_caption
    if keep_first_sentence:
        first, rest = sentences[0], sentences[1:]
        random.shuffle(rest)
        sentences = [first] + rest
    else:
        random.shuffle(sentences)
    result = '. '.join(s.rstrip('.') for s in sentences)
    if not result.endswith('.'):
        result += '.'
    return result


def _select_variant(caption_mode, mixed_weights, has_nl_caption):
    """Pick a caption variant ('tags' | 'nl' | 'tags_nl' | 'nl_tags')."""
    if caption_mode == "tags":
        return "tags"
    if caption_mode == "nl":
        if has_nl_caption:
            return "nl"
        if not hasattr(_select_variant, '_nl_fallback_count'):
            _select_variant._nl_fallback_count = 0
        _select_variant._nl_fallback_count += 1
        if _select_variant._nl_fallback_count <= 5:
            print("Warning: caption_mode='nl' but no *_nl.txt found for sample, "
                  f"falling back to tags (warning {_select_variant._nl_fallback_count}/5)")
        elif _select_variant._nl_fallback_count == 6:
            print("Warning: Suppressing further NL fallback warnings.")
        return "tags"
    if caption_mode == "mixed":
        available = {"tags": mixed_weights.get("tags", 50)}
        if has_nl_caption:
            available["nl"] = mixed_weights.get("nl", 10)
            available["tags_nl"] = mixed_weights.get("tags_nl", 20)
            available["nl_tags"] = mixed_weights.get("nl_tags", 20)
        total = sum(available.values())
        if total == 0:
            return "tags"
        r = random.random() * total
        cumulative = 0
        for variant, weight in available.items():
            cumulative += weight
            if r < cumulative:
                return variant
        return variant
    return "tags"


def _construct_caption(variant, processed_tags, processed_nl):
    """Combine tag/NL components per the chosen variant, handling empties."""
    tags = processed_tags.strip() if processed_tags else ""
    nl = processed_nl.strip() if processed_nl else ""
    if variant == "tags":
        return tags if tags else nl
    if variant == "nl":
        return nl if nl else tags
    if variant == "tags_nl":
        return f"{tags}. {nl}" if (tags and nl) else (tags or nl)
    if variant == "nl_tags":
        return f"{nl}. {tags}" if (tags and nl) else (nl or tags)
    return tags or nl


def _load_nl_caption(image_spec):
    """Load '{basename}_nl.txt' next to the image, or None."""
    if image_spec is None:
        return None
    tar_file, image_path = image_spec
    if tar_file is not None or not image_path:
        return None
    image_path = Path(image_path)
    nl_path = image_path.parent / f"{image_path.stem}_nl.txt"
    if not nl_path.exists():
        return None
    try:
        with open(nl_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content or None
    except Exception:
        return None


def _should_debug_sample(sample_idx, interval):
    if interval == 0:
        return True
    if interval == -1:
        return sample_idx < 10
    return sample_idx % interval == 0


def _print_debug(sample_idx, info, full_dropout):
    print(f"\n[Caption Debug | Sample {sample_idx}]")
    if full_dropout:
        print("├─ Full caption dropout: YES (CFG training)")
        print("├─ Final caption: \"\"")
        print("└─ (all other processing skipped)")
        return
    print(f"├─ Original tags: \"{info.get('original_tags', '')}\"")
    nl = info.get('original_nl')
    print(f"├─ Original NL: \"{(nl[:100] + '...') if nl and len(nl) > 100 else (nl or '(none)')}\"")
    if info.get('dropped_tags'):
        print(f"├─ Dropped tags: {info['dropped_tags']}")
    print(f"├─ Surviving tags: \"{info.get('surviving_tags', '')}\"")
    print(f"├─ Variant selected: {info.get('variant', 'unknown')}")
    final = info.get('final_caption', '')
    print(f"└─ Final caption: \"{(final[:150] + '...') if len(final) > 150 else final}\"")


def log_caption_stats(debug_state, step, interval=1000):
    if step % interval != 0 or step == 0:
        return
    variants = ['tags', 'nl', 'tags_nl', 'nl_tags']
    counts = [debug_state.get(f'variant_{v}', 0) for v in variants]
    total = sum(counts)
    if total == 0:
        return
    pcts = [f"{v}={c}({100 * c // total}%)" for v, c in zip(variants, counts)]
    print(f"Step {step} | Variants: {', '.join(pcts)} | "
          f"Tag drops: {debug_state.get('tag_dropout_count', 0)} | "
          f"CFG drops: {debug_state.get('full_dropout_count', 0)}")


def process_caption(tags_str, image_spec, config, protected_tags, sample_idx, debug_state):
    """Full per-sample caption pipeline. Returns the final caption string.

    Steps: full-caption (CFG) dropout -> load NL if needed -> tag shuffle ->
    tag dropout -> NL sentence shuffle -> variant selection -> construct.
    """
    debug_info = {}
    debug_enabled = config.get('debug_caption_processing', False)
    debug_interval = config.get('debug_caption_interval', 100)
    should_debug = debug_enabled and _should_debug_sample(sample_idx, debug_interval)

    # Step 1: full caption dropout (unconditional / CFG training).
    caption_dropout = config.get('caption_dropout_percent', 0.0)
    if caption_dropout > 0 and random.random() < caption_dropout:
        debug_state['full_dropout_count'] = debug_state.get('full_dropout_count', 0) + 1
        if should_debug:
            _print_debug(sample_idx, debug_info, full_dropout=True)
        return ""

    if should_debug:
        debug_info['original_tags'] = tags_str

    # Step 2: load NL caption if the mode uses it.
    caption_mode = config.get('caption_mode', 'tags')
    nl_caption = _load_nl_caption(image_spec) if caption_mode in ['nl', 'mixed'] else None
    if should_debug:
        debug_info['original_nl'] = nl_caption

    # Step 3: parse + shuffle + drop tags.
    delimiter = config.get('tag_delimiter', ', ')
    tags = [t.strip() for t in tags_str.split(delimiter) if t.strip()]

    if config.get('shuffle_tags', False):
        keep_first_n = config.get('shuffle_keep_first_n', 0)
        if 0 < keep_first_n < len(tags):
            prefix, suffix = tags[:keep_first_n], tags[keep_first_n:]
            random.shuffle(suffix)
            tags = prefix + suffix
        else:
            random.shuffle(tags)

    dropout_percent = config.get('tag_dropout_percent', 0.0)
    keep_first_n = config.get('shuffle_keep_first_n', 0)
    protected_indices = set(range(min(keep_first_n, len(tags))))
    dropped_tags = []
    if dropout_percent > 0:
        tags, dropped_tags = _apply_tag_dropout(tags, dropout_percent, protected_indices, protected_tags)
        debug_state['tag_dropout_count'] = debug_state.get('tag_dropout_count', 0) + len(dropped_tags)
    processed_tags = delimiter.join(tags)

    if should_debug:
        debug_info['dropped_tags'] = dropped_tags
        debug_info['surviving_tags'] = processed_tags

    # Step 4: process NL caption.
    has_nl = bool(nl_caption and nl_caption.strip())
    processed_nl = ""
    if has_nl:
        processed_nl = _process_nl_caption(
            nl_caption,
            config.get('nl_shuffle_sentences', False),
            config.get('nl_keep_first_sentence', False),
        )

    # Step 5: pick a variant and construct.
    mixed_weights = config.get('mixed_weights', DEFAULT_MIXED_WEIGHTS)
    variant = _select_variant(caption_mode, mixed_weights, has_nl)
    debug_state[f'variant_{variant}'] = debug_state.get(f'variant_{variant}', 0) + 1
    if should_debug:
        debug_info['variant'] = variant

    final_caption = _construct_caption(variant, processed_tags, processed_nl)

    # Step 6: never return empty (unless it was intentional CFG dropout above).
    if not final_caption or not final_caption.strip():
        final_caption = tags_str

    if should_debug:
        debug_info['final_caption'] = final_caption
        _print_debug(sample_idx, debug_info, full_dropout=False)
    return final_caption

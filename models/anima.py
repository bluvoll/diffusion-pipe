# Anima Pipeline for diffusion-pipe
# Based on Cosmos-Predict2 but with dual text encoders (Qwen3-0.6B + T5)
# Uses Qwen Image VAE (same architecture/normalization as Wan VAE)

import math
import random
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
import transformers
from transformers import T5TokenizerFast, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from models.anima_modeling import Anima
from models.cosmos_predict2 import get_dit_config, time_shift, get_lin_function, WanVAE, vae_encode
from utils.common import load_state_dict, AUTOCAST_DTYPE
from utils.offloading import ModelOffloader


KEEP_IN_HIGH_PRECISION = ['x_embedder', 't_embedder', 't_embedding_norm', 'final_layer', 'llm_adapter']

# Minimum number of tags that must survive dropout
MIN_SURVIVING_TAGS = 3

# Default weights for mixed caption mode
DEFAULT_MIXED_WEIGHTS = {'tags': 50, 'nl': 10, 'tags_nl': 20, 'nl_tags': 20}


def _load_protected_tags(filepath):
    """
    Load protected tags from file.

    Args:
        filepath: Path to protected_tags.txt (one tag per line)

    Returns:
        Set of protected tag strings
    """
    if not filepath:
        return set()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tags = set()
            for line in f:
                tag = line.strip()
                if tag and not tag.startswith('#'):  # Allow comments
                    tags.add(tag)
            return tags
    except FileNotFoundError:
        print(f"Warning: protected_tags_file not found: {filepath}")
        return set()
    except Exception as e:
        print(f"Warning: Error loading protected_tags_file: {e}")
        return set()


def _apply_tag_dropout(tags, dropout_percent, protected_indices, protected_tags):
    """
    Drop a percentage of tags, respecting protections.

    Args:
        tags: List of tag strings
        dropout_percent: Fraction of tags to drop (0.0 to 1.0)
        protected_indices: Indices protected by keep_first_n
        protected_tags: Tag strings protected by protected_tags_file

    Returns:
        (surviving_tags, dropped_tags) tuple
    """
    if dropout_percent <= 0 or len(tags) == 0:
        return tags, []

    # Identify which tags can be dropped
    droppable_indices = []
    for i, tag in enumerate(tags):
        if i in protected_indices:
            continue
        if tag.strip() in protected_tags:
            continue
        droppable_indices.append(i)

    if len(droppable_indices) == 0:
        return tags, []

    # Calculate how many to drop (use round for unbiased rounding)
    num_to_drop = round(len(droppable_indices) * dropout_percent)

    # Ensure minimum survivors
    max_droppable = len(tags) - MIN_SURVIVING_TAGS
    num_to_drop = min(num_to_drop, max(0, max_droppable))

    if num_to_drop == 0:
        return tags, []

    # Randomly select indices to drop
    drop_indices = set(random.sample(droppable_indices, num_to_drop))

    surviving = []
    dropped = []
    for i, tag in enumerate(tags):
        if i in drop_indices:
            dropped.append(tag)
        else:
            surviving.append(tag)

    return surviving, dropped


def _process_nl_caption(nl_caption, shuffle_sentences, keep_first_sentence):
    """
    Process NL caption with optional sentence shuffling.

    Args:
        nl_caption: Raw NL caption string
        shuffle_sentences: Whether to shuffle sentences
        keep_first_sentence: Keep first sentence in place when shuffling

    Returns:
        Processed NL caption string
    """
    if not shuffle_sentences or not nl_caption:
        return nl_caption

    # Split by ". " (period + space)
    # Handle edge cases: trailing period, multiple spaces
    sentences = []
    for s in nl_caption.split('. '):
        s = s.strip()
        if s:
            sentences.append(s)

    if len(sentences) <= 1:
        return nl_caption

    if keep_first_sentence:
        first = sentences[0]
        rest = sentences[1:]
        random.shuffle(rest)
        sentences = [first] + rest
    else:
        random.shuffle(sentences)

    # Rejoin, ensuring proper period spacing
    result = '. '.join(s.rstrip('.') for s in sentences)
    if not result.endswith('.'):
        result += '.'

    return result


def _select_variant(caption_mode, mixed_weights, has_nl_caption):
    """
    Select which caption variant to use for this sample.

    Args:
        caption_mode: "tags", "nl", or "mixed"
        mixed_weights: Weight dict for mixed mode
        has_nl_caption: Whether NL caption is available

    Returns:
        Variant string: "tags", "nl", "tags_nl", or "nl_tags"
    """
    if caption_mode == "tags":
        return "tags"
    elif caption_mode == "nl":
        if has_nl_caption:
            return "nl"
        else:
            # Warn user about fallback (rate-limited to avoid console spam)
            if not hasattr(_select_variant, '_nl_fallback_count'):
                _select_variant._nl_fallback_count = 0
            _select_variant._nl_fallback_count += 1
            if _select_variant._nl_fallback_count <= 5:
                print(f"Warning: caption_mode='nl' but no *_nl.txt found for sample, "
                      f"falling back to tags (warning {_select_variant._nl_fallback_count}/5)")
            elif _select_variant._nl_fallback_count == 6:
                print("Warning: Suppressing further NL fallback warnings. "
                      "Check that your NL caption files have the '_nl.txt' suffix.")
            return "tags"
    elif caption_mode == "mixed":
        # Build available variants with weights
        available = {"tags": mixed_weights.get("tags", 50)}
        if has_nl_caption:
            available["nl"] = mixed_weights.get("nl", 10)
            available["tags_nl"] = mixed_weights.get("tags_nl", 20)
            available["nl_tags"] = mixed_weights.get("nl_tags", 20)

        # Normalize and select
        total = sum(available.values())
        if total == 0:
            return "tags"

        r = random.random() * total
        cumulative = 0
        for variant, weight in available.items():
            cumulative += weight
            if r < cumulative:
                return variant

        # Fallback should never execute with correct math, but guard against
        # floating-point edge cases by returning last variant instead of biasing to tags
        return variant
    else:
        return "tags"  # Unknown mode fallback


def _construct_caption(variant, processed_tags, processed_nl):
    """
    Construct final caption string based on selected variant.

    Handles empty strings gracefully to avoid malformed captions like ". text"
    or "text. " when one component is empty.

    Args:
        variant: "tags", "nl", "tags_nl", or "nl_tags"
        processed_tags: Processed tag string (may be empty)
        processed_nl: Processed NL caption string (may be empty)

    Returns:
        Final caption string
    """
    # Normalize empty/whitespace-only strings to empty
    tags = processed_tags.strip() if processed_tags else ""
    nl = processed_nl.strip() if processed_nl else ""

    if variant == "tags":
        return tags if tags else nl  # Fallback to NL if tags empty
    elif variant == "nl":
        return nl if nl else tags  # Fallback to tags if NL empty
    elif variant == "tags_nl":
        if tags and nl:
            return f"{tags}. {nl}"
        return tags or nl  # Return whichever is non-empty
    elif variant == "nl_tags":
        if tags and nl:
            return f"{nl}. {tags}"
        return nl or tags  # Return whichever is non-empty
    else:
        return tags or nl  # Fallback: return any non-empty component


def _load_nl_caption(image_spec):
    """
    Load NL caption from {basename}_nl.txt file.

    Args:
        image_spec: Tuple of (tar_file, image_path)

    Returns:
        NL caption string or None if not found
    """
    tar_file, image_path = image_spec
    if tar_file is not None:
        # Tar files not supported for NL captions yet
        return None

    image_path = Path(image_path)
    nl_path = image_path.parent / f"{image_path.stem}_nl.txt"

    if not nl_path.exists():
        return None

    try:
        with open(nl_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content if content else None
    except Exception:
        return None


def _should_debug_sample(sample_idx, interval):
    """Determine if we should print debug info for this sample."""
    if interval == 0:
        return True  # Every sample
    elif interval == -1:
        return sample_idx < 10  # First 10 only
    else:
        return sample_idx % interval == 0


def _print_debug(sample_idx, info, full_dropout):
    """Print debug information for a sample."""
    print(f"\n[Caption Debug | Sample {sample_idx}]")

    if full_dropout:
        print(f"├─ Full caption dropout: YES (CFG training)")
        print(f"├─ Final caption: \"\"")
        print(f"└─ (all other processing skipped)")
        return

    print(f"├─ Original tags: \"{info.get('original_tags', '')}\"")

    nl = info.get('original_nl')
    if nl:
        # Truncate long NL captions for display
        display_nl = nl[:100] + "..." if len(nl) > 100 else nl
        print(f"├─ Original NL: \"{display_nl}\"")
    else:
        print(f"├─ Original NL: (none)")

    if info.get('protected_indices'):
        print(f"├─ Protected indices (keep_first_n): {info['protected_indices']}")

    if info.get('protected_tags_matched'):
        print(f"├─ Protected tags (from file): {info['protected_tags_matched']}")

    dropped = info.get('dropped_tags', [])
    if dropped:
        print(f"├─ Dropped tags: {dropped}")

    print(f"├─ Surviving tags: \"{info.get('surviving_tags', '')}\"")

    if info.get('nl_shuffled') and info.get('nl_shuffled') != info.get('original_nl'):
        display_nl = info['nl_shuffled'][:100] + "..." if len(info['nl_shuffled']) > 100 else info['nl_shuffled']
        print(f"├─ NL shuffled: \"{display_nl}\"")

    print(f"├─ Variant selected: {info.get('variant', 'unknown')}")

    final = info.get('final_caption', '')
    display_final = final[:150] + "..." if len(final) > 150 else final
    print(f"├─ Final caption: \"{display_final}\"")
    print(f"└─ Full caption dropout: No")


def _log_caption_stats(debug_state, step, interval=1000):
    """Log caption processing statistics periodically."""
    if step % interval != 0 or step == 0:
        return

    variants = ['tags', 'nl', 'tags_nl', 'nl_tags']
    variant_counts = [debug_state.get(f'variant_{v}', 0) for v in variants]
    total = sum(variant_counts)

    if total == 0:
        return

    percentages = [f"{v}={c}({100*c//total}%)" for v, c in zip(variants, variant_counts)]

    tag_dropout = debug_state.get('tag_dropout_count', 0)
    full_dropout = debug_state.get('full_dropout_count', 0)

    print(f"Step {step} | Variants: {', '.join(percentages)} | "
          f"Tag drops: {tag_dropout} | CFG drops: {full_dropout}")


def _process_caption_full(
    tags_str,
    image_spec,
    config,
    protected_tags,
    sample_idx,
    debug_state
):
    """
    Full caption processing pipeline.

    Args:
        tags_str: Raw tags string from caption file
        image_spec: Tuple of (tar_file, image_path) for loading NL caption
        config: Dict with caption processing config options
        protected_tags: Set of protected tag strings
        sample_idx: Current sample index (for debug logging)
        debug_state: Mutable dict for tracking stats

    Returns:
        Final caption string for training
    """
    debug_info = {}
    debug_enabled = config.get('debug_caption_processing', False)
    debug_interval = config.get('debug_caption_interval', 100)

    should_debug = debug_enabled and _should_debug_sample(sample_idx, debug_interval)

    # Step 1: Full caption dropout (CFG training)
    caption_dropout = config.get('caption_dropout_percent', 0.0)
    if caption_dropout > 0 and random.random() < caption_dropout:
        debug_state['full_dropout_count'] = debug_state.get('full_dropout_count', 0) + 1
        if should_debug:
            _print_debug(sample_idx, debug_info, full_dropout=True)
        return ""

    if should_debug:
        debug_info['original_tags'] = tags_str

    # Step 2: Load NL caption if needed
    caption_mode = config.get('caption_mode', 'tags')
    nl_caption = None
    if caption_mode in ['nl', 'mixed']:
        nl_caption = _load_nl_caption(image_spec)

    if should_debug:
        debug_info['original_nl'] = nl_caption

    # Step 3: Parse and process tags
    delimiter = config.get('tag_delimiter', ', ')
    tags = [t.strip() for t in tags_str.split(delimiter) if t.strip()]

    # Shuffle tags
    if config.get('shuffle_tags', False):
        keep_first_n = config.get('shuffle_keep_first_n', 0)
        if keep_first_n > 0 and keep_first_n < len(tags):
            prefix = tags[:keep_first_n]
            suffix = tags[keep_first_n:]
            random.shuffle(suffix)
            tags = prefix + suffix
        else:
            random.shuffle(tags)

    # Apply tag dropout
    dropout_percent = config.get('tag_dropout_percent', 0.0)
    keep_first_n = config.get('shuffle_keep_first_n', 0)
    protected_indices = set(range(min(keep_first_n, len(tags))))

    dropped_tags = []
    if dropout_percent > 0:
        tags, dropped_tags = _apply_tag_dropout(
            tags, dropout_percent, protected_indices, protected_tags
        )
        debug_state['tag_dropout_count'] = debug_state.get('tag_dropout_count', 0) + len(dropped_tags)

    processed_tags = delimiter.join(tags)

    if should_debug:
        debug_info['protected_indices'] = protected_indices if protected_indices else None
        debug_info['protected_tags_matched'] = [t for t in tags if t in protected_tags]
        debug_info['dropped_tags'] = dropped_tags
        debug_info['surviving_tags'] = processed_tags

    # Step 4: Process NL caption
    has_nl = nl_caption is not None and len(nl_caption.strip()) > 0
    processed_nl = ""

    if has_nl:
        processed_nl = _process_nl_caption(
            nl_caption,
            config.get('nl_shuffle_sentences', False),
            config.get('nl_keep_first_sentence', False)
        )
        if should_debug:
            debug_info['nl_shuffled'] = processed_nl

    # Step 5: Select variant
    mixed_weights = config.get('mixed_weights', DEFAULT_MIXED_WEIGHTS)
    variant = _select_variant(caption_mode, mixed_weights, has_nl)

    # Track variant distribution
    variant_key = f'variant_{variant}'
    debug_state[variant_key] = debug_state.get(variant_key, 0) + 1

    if should_debug:
        debug_info['variant'] = variant

    # Step 6: Construct final caption
    final_caption = _construct_caption(variant, processed_tags, processed_nl)

    # Step 7: Validate final caption is not empty
    if not final_caption or not final_caption.strip():
        # Both tags and NL were empty - fall back to original caption
        if not hasattr(_process_caption_full, '_empty_caption_count'):
            _process_caption_full._empty_caption_count = 0
        _process_caption_full._empty_caption_count += 1
        if _process_caption_full._empty_caption_count <= 5:
            print(f"Warning: Caption processing produced empty result for sample {sample_idx}, "
                  f"using original caption (warning {_process_caption_full._empty_caption_count}/5)")
        elif _process_caption_full._empty_caption_count == 6:
            print("Warning: Suppressing further empty caption warnings. "
                  "Check your caption files for empty or whitespace-only content.")
        final_caption = tags_str  # Use original caption as fallback

    if should_debug:
        debug_info['final_caption'] = final_caption
        _print_debug(sample_idx, debug_info, full_dropout=False)

    return final_caption


def _shuffle_tags(caption, delimiter=', ', keep_first_n=0):
    """
    Shuffle tags in a caption string at training time.

    Args:
        caption: Caption string with tags separated by delimiter
        delimiter: Tag separator (default ", " for danbooru-style tags)
        keep_first_n: Keep the first N tags in place, shuffle the rest
                     (useful for keeping trigger words at the start)

    Returns:
        Caption with tags shuffled
    """
    if not caption or delimiter not in caption:
        return caption

    tags = caption.split(delimiter)
    if len(tags) <= 1:
        return caption

    # Keep first N tags in place, shuffle the rest
    if keep_first_n > 0 and keep_first_n < len(tags):
        prefix = tags[:keep_first_n]
        suffix = tags[keep_first_n:]
        random.shuffle(suffix)
        tags = prefix + suffix
    else:
        random.shuffle(tags)

    return delimiter.join(tags)


def _tokenize_t5(tokenizer, prompts):
    """Tokenize prompts using T5 tokenizer."""
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def _tokenize_qwen(tokenizer, prompts):
    """Tokenize prompts using Qwen tokenizer."""
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def _compute_qwen_embeddings(qwen_model, input_ids, attention_mask):
    """Compute Qwen3 hidden states for use as cross-attention context."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(qwen_model.device, dtype=torch.long)
    attention_mask = attention_mask.to(qwen_model.device, dtype=torch.long)

    with torch.no_grad():
        outputs = qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # Use the last hidden state
    hidden_states = outputs.hidden_states[-1]

    # Zero out padding positions
    lengths = attention_mask.sum(dim=1).cpu()
    for batch_id in range(hidden_states.shape[0]):
        length = lengths[batch_id]
        if length == 1:  # Empty prompt case
            length = 0
        hidden_states[batch_id][length:] = 0

    return hidden_states


class AnimaPipeline(BasePipeline):
    name = 'anima'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['Block']  # Default: don't train LLMAdapter

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        dtype = self.model_config['dtype']
        self.cache_text_embeddings = self.model_config.get('cache_text_embeddings', True)

        # Configure adapter target modules based on train_llm_adapter option
        train_llm_adapter = self.model_config.get('train_llm_adapter', False)
        if train_llm_adapter:
            self.adapter_target_modules = ['Block', 'LLMAdapterTransformerBlock']
            print("Note: train_llm_adapter=true - LLMAdapter will be trained with LoRA")
        else:
            self.adapter_target_modules = ['Block']

        # === Caption Processing Config ===
        # Build a config dict for caption processing
        self.caption_config = {
            # Tag processing
            'shuffle_tags': self.model_config.get('shuffle_tags', False),
            'tag_delimiter': self.model_config.get('tag_delimiter', ', '),
            'shuffle_keep_first_n': self.model_config.get('shuffle_keep_first_n', 0),
            'tag_dropout_percent': self.model_config.get('tag_dropout_percent', 0.0),

            # NL caption processing
            'nl_shuffle_sentences': self.model_config.get('nl_shuffle_sentences', False),
            'nl_keep_first_sentence': self.model_config.get('nl_keep_first_sentence', False),

            # Caption dropout (CFG training)
            'caption_dropout_percent': self.model_config.get('caption_dropout_percent', 0.0),

            # Mode selection
            'caption_mode': self.model_config.get('caption_mode', 'tags'),
            'mixed_weights': self.model_config.get('mixed_weights', DEFAULT_MIXED_WEIGHTS),

            # Debug
            'debug_caption_processing': self.model_config.get('debug_caption_processing', False),
            'debug_caption_interval': self.model_config.get('debug_caption_interval', 100),
        }

        # Load protected tags
        protected_tags_file = self.model_config.get('protected_tags_file', None)
        self.protected_tags = _load_protected_tags(protected_tags_file)
        if protected_tags_file and self.protected_tags:
            print(f"Loaded {len(self.protected_tags)} protected tags from {protected_tags_file}")

        # Caption processing state for stats tracking
        self.caption_debug_state = {}
        self.caption_sample_idx = 0

        # Validate config
        self._validate_caption_config()

        # Legacy compatibility
        self.shuffle_tags = self.caption_config['shuffle_tags']
        self.tag_delimiter = self.caption_config['tag_delimiter']
        self.keep_first_n_tags = self.caption_config['shuffle_keep_first_n']

        # Warn about caching incompatibility
        caption_mode = self.caption_config['caption_mode']
        if self.cache_text_embeddings and caption_mode != 'tags':
            print(f"WARNING: caption_mode='{caption_mode}' requires cache_text_embeddings=false. "
                  "Falling back to caption_mode='tags'.")
            self.caption_config['caption_mode'] = 'tags'
        if self.shuffle_tags and self.cache_text_embeddings:
            print("WARNING: shuffle_tags requires cache_text_embeddings=false to work at training time. "
                  "With cache_text_embeddings=true, use cache_shuffle_num in your dataset config instead.")

        # VAE - Qwen Image VAE (16 channel, same architecture/normalization as Wan VAE)
        self.vae = WanVAE(
            vae_pth=self.model_config['vae_path'],
            dtype=dtype,
        )
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        # T5 Tokenizer - for getting token IDs (used by LLMAdapter)
        self.t5_tokenizer = T5TokenizerFast(
            vocab_file='configs/t5_old/spiece.model',
            tokenizer_file='configs/t5_old/tokenizer.json',
        )

        # Qwen3 Tokenizer and Model - for getting embeddings
        qwen_path = self.model_config['qwen_path']
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        # Load Qwen3-0.6B model for text encoding
        qwen_config = AutoConfig.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)

        if self.model_config.get('qwen_nf4', False):
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quantization_config = None

        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            config=qwen_config,
            torch_dtype=dtype,
            local_files_only=True,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        if quantization_config is None and self.model_config.get('qwen_fp8', False):
            for name, p in self.qwen_model.named_parameters():
                if p.ndim == 2:
                    p.data = p.data.to(torch.float8_e4m3fn)

        self.qwen_model.requires_grad_(False)

    def _validate_caption_config(self):
        """Validate caption-related config options."""
        config = self.caption_config

        caption_mode = config.get('caption_mode', 'tags')
        valid_modes = ['tags', 'nl', 'mixed']
        if caption_mode not in valid_modes:
            raise ValueError(f"caption_mode must be one of {valid_modes}, got '{caption_mode}'")

        dropout = config.get('tag_dropout_percent', 0.0)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"tag_dropout_percent must be between 0.0 and 1.0, got {dropout}")

        caption_dropout = config.get('caption_dropout_percent', 0.0)
        if not 0.0 <= caption_dropout <= 1.0:
            raise ValueError(f"caption_dropout_percent must be between 0.0 and 1.0, got {caption_dropout}")

        if caption_mode in ['nl', 'mixed']:
            print(f"Note: caption_mode='{caption_mode}' expects {{name}}_nl.txt files. "
                  "Samples without NL captions will fall back to tags.")

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        state_dict = load_state_dict(self.model_config['transformer_path'])

        # Remove 'net.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                k = k[len('net.'):]
            # Handle ComfyUI format with 'diffusion_model.' prefix
            if k.startswith('diffusion_model.'):
                k = k[len('diffusion_model.'):]
            new_state_dict[k] = v
        state_dict = new_state_dict

        # Get config for base model (without llm_adapter weights)
        base_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('llm_adapter.')}
        dit_config = get_dit_config(base_state_dict)

        with init_empty_weights():
            transformer = Anima(**dit_config)

        for name, p in transformer.named_parameters():
            # Keep LLMAdapter and certain layers in higher precision
            dtype_to_use = dtype if (any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) or p.ndim == 1) else transformer_dtype
            if name in state_dict:
                set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
            else:
                # Initialize missing weights (shouldn't happen with proper checkpoint)
                print(f"Warning: Missing weight {name}, initializing randomly")
                set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=torch.randn_like(p))

        self.transformer = transformer
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae.model

    def get_text_encoders(self):
        if self.cache_text_embeddings:
            return [self.qwen_model]
        else:
            return []

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        state_dict = {'net.'+k: v for k, v in state_dict.items()}
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        """
        Returns a function that computes both:
        - Qwen3 embeddings (for LLMAdapter cross-attention context)
        - T5 token IDs (for LLMAdapter embedding input)
        """
        def fn(captions, is_video):
            # Get Qwen3 embeddings
            qwen_encoding = _tokenize_qwen(self.qwen_tokenizer, captions)
            qwen_embeds = _compute_qwen_embeddings(
                self.qwen_model,
                qwen_encoding.input_ids,
                qwen_encoding.attention_mask
            )

            # Get T5 token IDs
            t5_encoding = _tokenize_t5(self.t5_tokenizer, captions)

            return {
                'qwen_embeds': qwen_embeds,
                't5_input_ids': t5_encoding.input_ids,
                't5_attention_mask': t5_encoding.attention_mask,
            }
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            qwen_inputs = (inputs['qwen_embeds'],)
            t5_input_ids = inputs['t5_input_ids']
        else:
            # Compute on-the-fly with full caption processing
            captions = inputs['caption']
            image_specs = inputs.get('image_spec', None)

            # Process each caption through the full pipeline
            if isinstance(captions, list):
                processed_captions = []
                for i, caption in enumerate(captions):
                    # Get image_spec for this sample (for NL caption loading)
                    image_spec = image_specs[i] if image_specs else (None, None)

                    processed = _process_caption_full(
                        caption,
                        image_spec,
                        self.caption_config,
                        self.protected_tags,
                        self.caption_sample_idx,
                        self.caption_debug_state
                    )
                    processed_captions.append(processed)
                    self.caption_sample_idx += 1

                captions = processed_captions
            else:
                image_spec = image_specs[0] if image_specs else (None, None)
                captions = _process_caption_full(
                    captions,
                    image_spec,
                    self.caption_config,
                    self.protected_tags,
                    self.caption_sample_idx,
                    self.caption_debug_state
                )
                self.caption_sample_idx += 1

            # Log stats periodically
            _log_caption_stats(self.caption_debug_state, self.caption_sample_idx)

            qwen_encoding = _tokenize_qwen(self.qwen_tokenizer, captions)
            qwen_inputs = (qwen_encoding.input_ids, qwen_encoding.attention_mask)
            t5_encoding = _tokenize_t5(self.t5_tokenizer, captions)
            t5_input_ids = t5_encoding.input_ids

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded)*latents + t_expanded*noise
        target = noise - latents

        return (noisy_latents, t.view(-1, 1), *qwen_inputs, t5_input_ids), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        qwen_model = None if self.cache_text_embeddings else self.qwen_model
        layers = [InitialLayer(transformer, qwen_model, self.qwen_tokenizer, self.t5_tokenizer)]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
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
    def __init__(self, model, qwen_model, qwen_tokenizer, t5_tokenizer):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.pos_embedder = model.pos_embedder
        if model.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = model.extra_pos_embedder
        self.t_embedder = model.t_embedder
        self.t_embedding_norm = model.t_embedding_norm
        self.llm_adapter = model.llm_adapter
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_C_T_H_W, timesteps_B_T, *text_inputs = inputs
        batch_size = x_B_C_T_H_W.shape[0]
        target_device = x_B_C_T_H_W.device
        target_dtype = x_B_C_T_H_W.dtype

        # If qwen_model is not None, we need to compute embeddings on-the-fly.
        # In cached mode, qwen_embeds is already computed and passed through.
        if self.qwen_model is None:
            assert len(text_inputs) == 2, f"Expected cached inputs (qwen_embeds, t5_input_ids), got {len(text_inputs)} items."
            qwen_embeds, t5_input_ids = text_inputs

            if qwen_embeds.device != target_device:
                qwen_embeds = qwen_embeds.to(target_device)
            if t5_input_ids.device != target_device:
                t5_input_ids = t5_input_ids.to(target_device)
            if t5_input_ids.dtype != torch.long:
                t5_input_ids = t5_input_ids.long()

            # Process through LLM adapter
            crossattn_emb = self.llm_adapter(qwen_embeds, t5_input_ids)
        else:
            assert len(text_inputs) == 3, f"Expected non-cached inputs (qwen_input_ids, qwen_attention_mask, t5_input_ids), got {len(text_inputs)} items."
            qwen_input_ids, qwen_attention_mask, t5_input_ids = text_inputs

            # Always run through models - even for empty prompts
            # (The zeros optimization breaks pipeline parallelism gradient flow)
            with torch.no_grad():
                qwen_embeds = _compute_qwen_embeddings(
                    self.qwen_model,
                    qwen_input_ids,
                    qwen_attention_mask,
                )

            if qwen_embeds.device != target_device:
                qwen_embeds = qwen_embeds.to(target_device)
            if t5_input_ids.device != target_device:
                t5_input_ids = t5_input_ids.to(target_device)
            if t5_input_ids.dtype != torch.long:
                t5_input_ids = t5_input_ids.long()

            # Process through LLM adapter to get final cross-attention embeddings
            crossattn_emb = self.llm_adapter(qwen_embeds, t5_input_ids)

        # Pad to 512 tokens if needed
        if crossattn_emb.shape[1] < 512:
            crossattn_emb = F.pad(crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1]))

        padding_mask = torch.zeros(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4], dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.model[0].prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=None,
            padding_mask=padding_mask,
        )
        assert extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is None
        assert rope_emb_L_1_1_D is not None

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Note: timesteps_B_T is NOT included - it's only used here in InitialLayer
        # Including it breaks pipeline parallelism (no gradient flows through unused tensors)
        outputs = make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D)
        for item in outputs:
            item.requires_grad_(True)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D = inputs

        self.offloader.wait_for_block(self.block_idx)
        x_B_T_H_W_D = self.block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D = inputs
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        net_output_B_C_T_H_W = self.unpatchify(x_B_T_H_W_O)
        return net_output_B_C_T_H_W

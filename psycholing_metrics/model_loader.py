"""Initialization of HuggingFace language models and tokenizers."""

import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertTokenizerFast,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    LlamaForCausalLM,
    MambaForCausalLM,
)

logger = logging.getLogger(__name__)


def _has_meta_params(loaded) -> bool:
    """True if any parameter of a loaded *model* is still on the meta device.

    Some safetensors checkpoints store tied weights as a metadata alias only
    (e.g. redrussianarmy/gpt2-turkish-cased maps transformer.wte.weight to
    lm_head.weight); transformers' meta-init load path then leaves the
    aliased tensor on meta without raising, and the model silently has no
    input embeddings. Tokenizers have no ``parameters`` and return False.
    """
    params = getattr(loaded, "parameters", None)
    if params is None:
        return False
    return any(p.is_meta for p in loaded.parameters())


def _from_pretrained_local_first(cls, model_name: str, **kwargs):
    """Load from the local HF cache first, falling back to the network.

    Repos can become gated upstream *after* they were cached (e.g.
    AI-Sweden-Models/gpt-sw3-126m in mid-2026): the metadata refresh then
    fails with a gated-repo error even though every file is already on
    disk. ``local_files_only=True`` skips the network entirely when the
    cache is complete, and also removes hub rate-limit failures when many
    models are loaded in parallel.

    Additionally guards against broken tied-weight safetensors (see
    :func:`_has_meta_params`): if the loaded model still has meta-device
    parameters — or the load dies with the associated ``NotImplementedError:
    Cannot copy out of meta tensor`` — retry with ``use_safetensors=False``
    so the original .bin checkpoint (which stores the tied tensor
    explicitly) is used instead.
    """

    def _load(**extra):
        merged = {**kwargs, **extra}
        try:
            return cls.from_pretrained(model_name, local_files_only=True, **merged)
        except (OSError, ValueError) as exc:
            logger.info(
                "Local cache miss for %s (%s); fetching from the HuggingFace hub.",
                model_name,
                exc.__class__.__name__,
            )
            return cls.from_pretrained(model_name, **merged)

    try:
        loaded = _load()
    except NotImplementedError as exc:
        if "meta tensor" not in str(exc):
            raise
        logger.warning(
            "Meta-tensor error loading %s from safetensors; retrying with "
            "use_safetensors=False.",
            model_name,
        )
        return _load(use_safetensors=False)
    if _has_meta_params(loaded):
        logger.warning(
            "%s loaded with meta-device parameters (broken tied-weight "
            "safetensors); retrying with use_safetensors=False.",
            model_name,
        )
        return _load(use_safetensors=False)
    return loaded


# Supported model families and their HuggingFace identifiers:
#
# GPT-2:    gpt2, gpt2-medium, gpt2-large, gpt2-xl
# CzeGPT-2: MU-NLPC/CzeGPT-2 (Czech GPT-2)
# mGPT:     ai-forever/mGPT (multilingual GPT-2)
# Chinese GPT-2: uer/gpt2-xlarge-chinese-cluecorpussmall and siblings (uses BertTokenizerFast)
# GPT-Neo:  EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B,
#           EleutherAI/gpt-j-6B, EleutherAI/gpt-neox-20b
# OPT:      facebook/opt-125m, facebook/opt-350m, ..., facebook/opt-66b
# Pythia:   EleutherAI/pythia-70m, ..., EleutherAI/pythia-12b
# Llama:    meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf, meta-llama/Llama-2-70b-hf
# Gemma:    google/gemma-2b, google/gemma-7b, google/recurrentgemma-2b, google/recurrentgemma-9b
# Mamba:    state-spaces/mamba-130m-hf, ..., state-spaces/mamba-2.8b-hf
# Mistral:  mistralai/Mistral-7B-Instruct-v0.*


def load_tokenizer_and_model(
    model_name: str,
    device: str = "cpu",
    pythia_checkpoint: str | None = "step143000",
    hf_access_token: str | None = None,
):
    """Load a HuggingFace causal language model and its tokenizer.

    :param model_name: HuggingFace model identifier (e.g. 'gpt2', 'EleutherAI/pythia-70m').
    :param device: target device for the model. Defaults to 'cpu'.
    :param pythia_checkpoint: checkpoint revision for Pythia models. Defaults to 'step143000'.
    :param hf_access_token: HuggingFace access token, required for gated models (Llama, Gemma, Mistral).
    :raises ValueError: if the model family is not supported.
    :return: (tokenizer, model) tuple.
    """
    model_variant = model_name.split("/")[-1]

    if "chinese" in model_variant:
        tokenizer = _from_pretrained_local_first(BertTokenizerFast, model_name)

    elif any(
        variant in model_variant
        for variant in ["gpt-neo", "gpt", "CzeGPT", "mGPT", "opt", "mamba", "rwkv"]
    ):
        tokenizer = _from_pretrained_local_first(
            AutoTokenizer, model_name, use_fast=True
        )

    elif "gpt-neox" in model_variant:
        tokenizer = _from_pretrained_local_first(GPTNeoXTokenizerFast, model_name)

    elif "Eagle" in model_variant:  # RWKV V5
        tokenizer = _from_pretrained_local_first(
            AutoTokenizer, model_name, trust_remote_code=True
        )
    elif any(variant in model_variant for variant in ["Llama", "Mistral", "gemma"]):
        assert hf_access_token is not None, (
            f"Please provide the HuggingFace access token to load {model_name}"
        )
        tokenizer = _from_pretrained_local_first(
            AutoTokenizer, model_name, use_fast=True, token=hf_access_token
        )

    elif "pythia" in model_variant:
        tokenizer = _from_pretrained_local_first(
            AutoTokenizer, model_name, revision=pythia_checkpoint, use_fast=True
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if any(
        variant in model_variant
        for variant in ["gpt-neo", "gpt", "CzeGPT", "mGPT", "opt", "rwkv"]
    ):
        model = _from_pretrained_local_first(
            AutoModelForCausalLM, model_name, device_map="auto"
        )

    elif "Eagle" in model_variant:  # RWKV
        model = _from_pretrained_local_first(
            AutoModelForCausalLM, model_name, device_map="auto", trust_remote_code=True
        ).to(torch.float32)

    elif "pythia" in model_variant:
        model = _from_pretrained_local_first(
            GPTNeoXForCausalLM,
            model_name,
            revision=pythia_checkpoint,
            device_map="auto",
        )

    elif "mamba" in model_variant:
        model = _from_pretrained_local_first(
            MambaForCausalLM,
            model_name,
            device_map="auto",
        )

    elif "Llama" in model_variant:
        model = _from_pretrained_local_first(
            LlamaForCausalLM, model_name, token=hf_access_token, device_map="auto"
        )

    elif any(variant in model_variant for variant in ["gemma-2"]):
        model = _from_pretrained_local_first(
            AutoModelForCausalLM,
            model_name,
            token=hf_access_token,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    elif any(variant in model_variant for variant in ["Mistral", "gemma"]):
        model = _from_pretrained_local_first(
            AutoModelForCausalLM, model_name, token=hf_access_token, device_map="auto"
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return tokenizer, model

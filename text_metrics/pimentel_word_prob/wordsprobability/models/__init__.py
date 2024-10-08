from .bow_lm import (
    EnglishGpt2Small,
    EnglishGpt2Medium,
    EnglishGpt2Large,
    EnglishGpt2Xl,
    EnglishPythia70M,
    EnglishPythia160M,
    EnglishPythia410M,
    EnglishPythia14B,
    EnglishPythia28B,
    EnglishPythia69B,
    EnglishPythia120B,
)

MODELS = {
    "gpt2-small": EnglishGpt2Small,
    "gpt2-medium": EnglishGpt2Medium,
    "gpt2-large": EnglishGpt2Large,
    "gpt2-xl": EnglishGpt2Xl,
    "EleutherAI/pythia-70m": EnglishPythia70M,
    "EleutherAI/pythia-160m": EnglishPythia160M,
    "EleutherAI/pythia-410m": EnglishPythia410M,
    "EleutherAI/pythia-1.4b": EnglishPythia14B,
    "EleutherAI/pythia-2.8b": EnglishPythia28B,
    "EleutherAI/pythia-6.9b": EnglishPythia69B,
    "EleutherAI/pythia-12b": EnglishPythia120B,
}


def get_model(model_name, model, tokenizer):
    model_cls = MODELS[model_name]
    return model_cls(model=model, tokenizer=tokenizer)


def get_bow_symbol(model_name):
    return MODELS[model_name].bow_symbol

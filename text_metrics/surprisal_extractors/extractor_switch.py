from text_metrics.surprisal_extractors.soft_cat_extractors import (
    SoftCatWholeCtxSurpExtractor,
    SoftCatSentencesSurpExtractor,
)
from text_metrics.surprisal_extractors.text_cat_extractor import CatCtxLeftSurpExtractor
from enum import Enum


class SurpExtractorType(Enum):
    SOFT_CAT_WHOLE_CTX_LEFT = "SoftCatWholeCtxSurpExtractor"
    SOFT_CAT_SENTENCES = "SoftCatSentencesSurpExtractor"
    CAT_CTX_LEFT = "CatCtxLeftSurpExtractor"


def get_surp_extractor(
    extractor_type: SurpExtractorType,
    model_name: str,
    model_target_device: str = "cpu",
    pythia_checkpoint: str | None = "step143000",
    hf_access_token: str | None = None,
):
    if extractor_type.value == SurpExtractorType.SOFT_CAT_WHOLE_CTX_LEFT.value:
        return SoftCatWholeCtxSurpExtractor(
            model_name, model_target_device, pythia_checkpoint, hf_access_token
        )
    elif extractor_type.value == SurpExtractorType.SOFT_CAT_SENTENCES.value:
        return SoftCatSentencesSurpExtractor(
            model_name, model_target_device, pythia_checkpoint, hf_access_token
        )
    elif extractor_type.value == SurpExtractorType.CAT_CTX_LEFT.value:
        return CatCtxLeftSurpExtractor(
            model_name, model_target_device, pythia_checkpoint, hf_access_token
        )
    else:
        raise ValueError(f"Unrecognized extractor type: {extractor_type}")

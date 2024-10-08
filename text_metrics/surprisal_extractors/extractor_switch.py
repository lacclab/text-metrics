from text_metrics.surprisal_extractors.soft_cat_extractors import (
    SoftCatWholeCtxSurpExtractor,
    SoftCatSentencesSurpExtractor,
)
from text_metrics.surprisal_extractors.text_cat_extractor import CatCtxLeftSurpExtractor
from text_metrics.surprisal_extractors.pimentel_extractor import PimentelSurpExtractor
from enum import Enum


class SurpExtractorType(Enum):
    # left context: l, target context: t
    # '*' - concat

    """
    l_rep = averaged_representations(l)
    full_context = l_rep * t
    Dimensionsof the embedding level input: (1 + No. tokens in t, hidden_size)
    """

    SOFT_CAT_WHOLE_CTX_LEFT = "SoftCatWholeCtxSurpExtractor"

    """
    l_sentences = concat([averaged_representations(sentence) for sentence in l])
    full_context = l_sentences * t 
    Dimensionsof the embedding level input: (No. sentences in L + No. tokens in t, hidden_size)
    """

    SOFT_CAT_SENTENCES = "SoftCatSentencesSurpExtractor"

    """full_context = l * t"""

    CAT_CTX_LEFT = "CatCtxLeftSurpExtractor"

    PIMENTEL_CTX_LEFT = "PimentelSurpExtractor"


def get_surp_extractor(
    extractor_type: SurpExtractorType,
    model_name: str,
    model_target_device: str = "cpu",
    pythia_checkpoint: str | None = "step143000",
    hf_access_token: str | None = None,
):
    if extractor_type.value == SurpExtractorType.SOFT_CAT_WHOLE_CTX_LEFT.value:
        return SoftCatWholeCtxSurpExtractor(
            model_name,
            extractor_type.value,
            model_target_device,
            pythia_checkpoint,
            hf_access_token,
        )
    elif extractor_type.value == SurpExtractorType.SOFT_CAT_SENTENCES.value:
        return SoftCatSentencesSurpExtractor(
            model_name,
            extractor_type.value,
            model_target_device,
            pythia_checkpoint,
            hf_access_token,
        )
    elif extractor_type.value == SurpExtractorType.CAT_CTX_LEFT.value:
        return CatCtxLeftSurpExtractor(
            model_name,
            extractor_type.value,
            model_target_device,
            pythia_checkpoint,
            hf_access_token,
        )
    elif extractor_type.value == SurpExtractorType.PIMENTEL_CTX_LEFT.value:
        return PimentelSurpExtractor(
            model_name,
            extractor_type.value,
            model_target_device,
            pythia_checkpoint,
            hf_access_token,
        )
    else:
        raise ValueError(f"Unrecognized extractor type: {extractor_type}")

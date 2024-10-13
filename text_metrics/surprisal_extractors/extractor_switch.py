from text_metrics.surprisal_extractors.soft_cat_extractors import (
    SoftCatWholeCtxSurpExtractor,
    SoftCatSentencesSurpExtractor,
)
from text_metrics.surprisal_extractors.text_cat_extractor import CatCtxLeftSurpExtractor
from text_metrics.surprisal_extractors.pimentel_extractor import PimentelSurpExtractor
from text_metrics.surprisal_extractors.inv_effect_extractor import InvEffectExtractor
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType


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
    elif extractor_type.value == SurpExtractorType.INV_EFFECT_EXTRACTOR.value:
        return InvEffectExtractor(
            model_name=model_name,
            extractor_type_name=extractor_type.value,
            model_target_device=model_target_device,
            pythia_checkpoint=pythia_checkpoint,
            hf_access_token=hf_access_token,
            target_extractor_type=SurpExtractorType.CAT_CTX_LEFT,
        )
    else:
        raise ValueError(f"Unrecognized extractor type: {extractor_type}")

from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType
from text_metrics.surprisal_extractors.base_extractor import BaseSurprisalExtractor


class InvEffectExtractor(BaseSurprisalExtractor):
    def __init__(
        self,
        model_name: str,
        extractor_type_name: str,
        target_extractor_type: SurpExtractorType,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            extractor_type_name=extractor_type_name,
            model_target_device=model_target_device,
            pythia_checkpoint=pythia_checkpoint,
            hf_access_token=hf_access_token,
        )

        from text_metrics.surprisal_extractors.extractor_switch import (
            get_surp_extractor,
        )

        self.target_extractor = get_surp_extractor(
            model_name=model_name,
            extractor_type=target_extractor_type,
            model_target_device=model_target_device,
            pythia_checkpoint=pythia_checkpoint,
            hf_access_token=hf_access_token,
        )

    def surprise(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ):
        baseline_surp = self.target_extractor.surprise(
            target_text=target_text,
            left_context_text=None,
            overlap_size=overlap_size,
            allow_overlap=allow_overlap,
        )
        other_surp = self.target_extractor.surprise(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
            allow_overlap=allow_overlap,
        )

        surp_diff = other_surp[0] - baseline_surp[0]
        # make all positive entries zero (surp_diff is a numpy array)
        surp_diff[surp_diff > 0] = 0
        # make all negative entries positive
        surp_diff = -surp_diff
        baseline_surp = list(baseline_surp)
        baseline_surp[0] += surp_diff
        baseline_surp = tuple(baseline_surp)

        return baseline_surp

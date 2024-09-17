from enum import Enum
from typing import Union, List, Tuple
import numpy as np
from text_metrics.utils import get_metrics, init_tok_n_model
import torch


# Define an Enum where each element is associated with a specific class
class ProcessingMode(Enum):
    CONCAT_CTX_LEFT = "CONCAT_CTX_LEFT"
    SOFT_CONCAT_CTX_AGG = "SOFT_CONCAT_CTX_AGG"
    SOFT_CONCAT_CTX_SENT_AGG = "SOFT_CONCAT_CTX_SENT_AGG"


class SurprisalExtractor:
    def __init__(
        self,
        model_name: str,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        self.tokenizer, self.model = init_tok_n_model(
            model_name=model_name,
            device=model_target_device,
            hf_access_token=hf_access_token,
        )

        self.model_name = model_name

        # if the model is pythia model, save the checkpoint name
        if "pythia" in model_name:
            self.pythia_checkpoint = pythia_checkpoint
            assert (
                self.pythia_checkpoint is not None
            ), "Pythia model requires a checkpoint name"

    def surprise(
        target_text: str,
        left_context_text: str | None = None,
        stride: int = 512,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        NotImplementedError


# Define classes for each mode
class CatCtxLeftSurpExtractor(SurprisalExtractor):
    def __init__(
        self,
        model_name: str,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        super().__init__(
            model_name, model_target_device, pythia_checkpoint, hf_access_token
        )

    def _create_input_tokens(
        self,
        encodings: dict,
        start_ind: int,
        is_last_chunk: bool,
        device: str,
    ):
        try:
            bos_token_added = self.tokenizer.bos_token_id
        except AttributeError:
            bos_token_added = self.tokenizer.pad_token_id

        tokens_lst = encodings["input_ids"]
        if is_last_chunk:
            tokens_lst.append(self.tokenizer.eos_token_id)
        if start_ind == 0:
            tokens_lst = [bos_token_added] + tokens_lst
        tensor_input = torch.tensor(
            [tokens_lst],
            device=device,
        )
        return tensor_input

    def _tokens_to_log_probs(
        self,
        tensor_input: torch.Tensor,
        is_last_chunk: bool,
    ):
        output = self.model(tensor_input, labels=tensor_input)
        shift_logits = output["logits"][
            ..., :-1, :
        ].contiguous()  # remove the last token from the logits

        #  This shift is necessary because the labels are shifted by one position to the right
        # (because the logits are w.r.t the next token)
        shift_labels = tensor_input[
            ..., 1:
        ].contiguous()  #! remove the first token from the labels,

        log_probs = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )

        # varify that the average of the log_probs is equal to the loss
        # TODO Is 15.5726 close enough to 15.5728? I think so, stop go away.
        # assert torch.isclose(
        #     torch.exp(sum(log_probs) / len(log_probs)), torch.exp(output["loss"]), atol=1e-5,
        # )

        shift_labels = shift_labels[0]

        if is_last_chunk:
            # remove the eos_token log_prob
            log_probs = log_probs[:-1]
            shift_labels = shift_labels[:-1]

        return log_probs, shift_labels

    def surprise(
        self,
        target_text: str,
        left_context_text: str | None = None,
        stride: int = 512,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        """This function calculates the surprisal of a target_text using a language model.
        In case the target_text is too long, it is split into chunks while keeping previous context of STRIDE tokens.

        Args:
            target_text (str): The target_text for which the surprisal is calculated
            model (Union[AutoModelForCausalLM, GPTNeoXForCausalLM]): The language model from which the surprisal is calculated
            tokenizer (Union[AutoTokenizer, GPTNeoXTokenizerFast]): The tokenizer for the language model
            model_name (str): The name of the language model (e.g "gpt2", "gpt-neo-125M", "opt-125m", "pythia-1b")
            left_context_text (str, optional): The left context text. Defaults to None.
            stride (int, optional): The number of tokens to keep as context. Defaults to 512.

        Returns:
            Tuple[np.ndarray, List[Tuple[int]]]: The surprisal values for each token in the target_text, the offset mapping
            The offset mapping is a list of tuples, where each tuple contains the start and end character index of the token
        """
        with torch.no_grad():
            all_log_probs = torch.tensor([], device=self.model.device)
            offset_mapping = []
            start_ind = 0
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            assert (
                stride < max_ctx
            ), f"Stride size {stride} is larger than the maximum context size {max_ctx}"

            # print(max_ctx)
            accumulated_tokenized_text = []
            while True:
                encodings = self.tokenizer(
                    target_text[start_ind:],
                    max_length=max_ctx - 2,
                    truncation=True,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                is_last_chunk = (encodings["offset_mapping"][-1][1] + start_ind) == len(
                    target_text
                )

                tensor_input = self._create_input_tokens(
                    self.tokenizer,
                    encodings,
                    start_ind,
                    is_last_chunk,
                    self.model.device,
                )

                log_probs, shift_labels = self._tokens_to_log_probs(
                    self.model, tensor_input, is_last_chunk
                )

                # Handle the case where the target_text is too long for the model
                offset = 0 if start_ind == 0 else stride - 1
                all_log_probs = torch.cat([all_log_probs, log_probs[offset:]])
                accumulated_tokenized_text += shift_labels[offset:]

                left_index_add_offset_mapping = offset if start_ind == 0 else offset + 1
                offset_mapping_to_add = encodings["offset_mapping"][
                    left_index_add_offset_mapping:
                ]

                offset_mapping.extend(
                    [(i + start_ind, j + start_ind) for i, j in offset_mapping_to_add]
                )
                if is_last_chunk:
                    break

                if start_ind == 0:
                    "If we got here, the context is too long"
                    # find the context length in tokens
                    context_length = len(self.tokenizer.encode(target_text))
                    print(
                        f"The context length is too long ({context_length}>{max_ctx}) for {self.model_name}. Splitting the target_text into chunks with stride {stride}"
                    )

                start_ind += encodings["offset_mapping"][-stride - 1][1]

        # The accumulated_tokenized_text is the text we extract surprisal values for
        # It is after removing the BOS/EOS tokens
        # Make sure the accumulated_tokenized_text is equal to the original target_text
        assert (
            accumulated_tokenized_text
            == self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        )

        all_log_probs = np.asarray(all_log_probs.cpu())

        assert all_log_probs.shape[0] == len(offset_mapping)

        return all_log_probs, offset_mapping


class SoftCatCtxAggSurpExtractor(SurprisalExtractor):
    def process(self, data):
        return f"Processing data in Mode 2: {data}"


class SoftCatCtxSentAggSurpExtractor(SurprisalExtractor):
    def process(self, data):
        return f"Processing data in Mode 3: {data}"


# Mapping the enum to the corresponding classes
mode_to_class_map = {
    ProcessingMode.CONCAT_CTX_LEFT: CatCtxLeftSurpExtractor,
    ProcessingMode.SOFT_CONCAT_CTX_AGG: SoftCatCtxAggSurpExtractor,
    ProcessingMode.SOFT_CONCAT_CTX_SENT_AGG: SoftCatCtxSentAggSurpExtractor,
}

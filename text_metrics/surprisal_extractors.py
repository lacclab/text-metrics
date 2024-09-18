from enum import Enum
from typing import List, Tuple
import numpy as np
from utils import init_tok_n_model, remove_redundant_left_context
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
        overlap_size: int = 512,
        allow_overlap: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        NotImplementedError

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

    def _full_context_to_probs_and_offsets(
        self, full_context: str, overlap_size: int, allow_overlap: bool, max_ctx: int
    ) -> Tuple[torch.Tensor, List[Tuple[int]], List[torch.Tensor]]:
        """This function calculates the surprisal of a full_context using a language model.

        Args:
            full_context (str): The text for which to calculate surprisal.
            overlap_size (int): A number of tokens to overlap between chunks in case the full_context is too long.
            allow_overlap (bool): If True, the full_context will be split into chunks with overlap_size tokens.
            max_ctx (int): The maximum context size of the model.

        Raises:
            ValueError: In case the full_context is too long and allow_overlap is False.

        Returns:
            all_log_probs (torch.Tensor): The log probabilities of the full_context.
            offset_mapping (List[Tuple[int]]): The offset mapping of the full_context.
            accumulated_tokenized_text (List[torch.Tensor]): The tokenized text of the full_context.
        """
        start_ind = 0
        accumulated_tokenized_text = []
        all_log_probs = torch.tensor([], device=self.model.device)
        offset_mapping = []
        while True:
            encodings = self.tokenizer(
                full_context[start_ind:],
                max_length=max_ctx - 2,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            is_last_chunk = (encodings["offset_mapping"][-1][1] + start_ind) == len(
                full_context
            )

            tensor_input = self._create_input_tokens(
                encodings,
                start_ind,
                is_last_chunk,
                self.model.device,
            )

            log_probs, shift_labels = self._tokens_to_log_probs(
                tensor_input, is_last_chunk
            )

            # Handle the case where the target_text is too long for the model
            offset = 0 if start_ind == 0 else overlap_size - 1
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
                context_length = len(self.tokenizer.encode(full_context))
                if allow_overlap:
                    print(
                        f"The context length is too long ({context_length}>{max_ctx}) for {self.model_name}. Splitting the full text into chunks with overlap {overlap_size}"
                    )
                else:
                    raise ValueError(
                        f"The context length is too long ({context_length}>{max_ctx}) for {self.model_name}. Try enabling allow_overlap and specify overlap size"
                    )

            start_ind += encodings["offset_mapping"][-overlap_size - 1][1]

        return all_log_probs, offset_mapping, accumulated_tokenized_text

    def surprise_target_only(
        self, target_text: str, allow_overlap: bool = False, overlap_size: int = 512
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"
        full_context = target_text

        with torch.no_grad():
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            assert (
                overlap_size < max_ctx
            ), f"Stride size {overlap_size} is larger than the maximum context size {max_ctx}"

            all_log_probs, offset_mapping, accumulated_tokenized_text = (
                self._full_context_to_probs_and_offsets(
                    full_context, overlap_size, allow_overlap, max_ctx
                )
            )

        # The accumulated_tokenized_text is the text we extract surprisal values for
        # It is after removing the BOS/EOS tokens
        # Make sure the accumulated_tokenized_text is equal to the original target_text
        assert (
            accumulated_tokenized_text
            == self.tokenizer(full_context, add_special_tokens=False)["input_ids"]
        )

        all_log_probs = np.asarray(all_log_probs.cpu())

        assert all_log_probs.shape[0] == len(offset_mapping)

        return all_log_probs, offset_mapping


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

    def _ommit_left_ctx_from_surp_res(
        self,
        target_text: str,
        all_log_probs: torch.Tensor,
        offset_mapping: List[Tuple[int]],
    ):
        # cut all_log_probs and offset_mapping to the length of the target_text ONLY (without the left context)
        # notice that accumulated_tokenized_text contains both the left context and the target_text
        # so we need to remove the left context from it
        target_text_len = len(self.tokenizer(target_text)["input_ids"])
        target_text_log_probs = all_log_probs[
            -target_text_len + 1 :
        ]  # because we concatenate the left context to the target text

        target_text_offset_mapping = offset_mapping[-target_text_len + 1 :]
        offset_mapping_onset = target_text_offset_mapping[0][0]
        target_text_offset_mapping = [
            (i - offset_mapping_onset, j - offset_mapping_onset)
            for i, j in target_text_offset_mapping
        ]

        assert target_text_log_probs.shape[0] == len(target_text_offset_mapping)

        return target_text_log_probs, target_text_offset_mapping

    def surprise(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        """This function calculates the surprisal of a target_text using a language model.
         In case the target_text is too long, it is split into chunks while keeping previous context of STRIDE tokens.
         Args:
             target_text (str): The text to calculate surprisal for.
             left_context_text (str | None, optional): A left context to be concatenated with the target_text BEFORE PASSED THROUGH THE MODEL. Defaults to None.
             overlap_size (int | None, optional): In case the target_text + left_context is too long, it is split into chunks with overlap_size tokens. Defaults to None.
             allow_overlap (bool, optional): If True, the target_text will be split into chunks with overlap_size tokens. Defaults to False.

         Raises:
             ValueError: In case the target_text is too long and allow_overlap is False.

         Returns:
             Tuple[np.ndarray, List[Tuple[int]]]: The surprisal values of the target_text and the offset mapping of the target_text.

         Example:
         target_text = "Angela Erdmann never knew her grandfather. He died in 1946, six years before she was born. But, on Tuesday 8th April,
         2014, she described the extraordinary moment when she received a message in a bottle, 101 years after he had lobbed it into the Baltic Sea.
         Thought to be the world's oldest message in a bottle, it was presented to Erdmann by the museum that is now exhibiting it in Germany."

         left_context_text = 'How did Angela Erdmann find out about the bottle?'

         Returns:
         target_text_log_probs = array([4.97636890e+00, 5.20506799e-01, 6.58840360e-03, 5.63895130e+00,
            2.56178689e+00, 3.68980312e+00, 4.96654510e+00, 3.15608168e+00,
            3.41664624e+00, 3.26654100e+00, 9.50351954e-01, 5.09015131e+00,
            ...
            3.03512335e+00, 9.98807430e+00, 1.23654938e+00, 1.98460269e+00,
            4.46555328e+00, 3.09026152e-01], dtype=float32)

        target_text_offset_mapping = [(0, 7), (7, 11), (11, 15), (15, 21), (21, 26), (26, 30), (30, 42), (42, 43),
        (43, 46), (46, 51), (51, 54), (54, 59), (59, 60), (60, 64), (64, 70), (70, 77), (77, 81), (81, 85),
        ....]

        ! Notice that the offset mapping is non-inclusice on the right index (it says 0-7 but actially the first word is chars 0-6)
        """
        # assert that if allow_overlap is True, overlap_size is not None
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"

        if left_context_text in [None, ""]:
            return self.surprise_target_only(target_text, allow_overlap, overlap_size)

        with torch.no_grad():
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            # only surprisal for the target text will be extracted, so if the left_context is longer
            # than the max context, we remove the redundant left context tokens that cannot be used in-context with the target text
            left_context_text = remove_redundant_left_context(
                self.tokenizer,
                left_context_text=left_context_text,
                max_ctx_in_tokens=max_ctx,
            )

            full_context = left_context_text + " " + target_text

            assert (
                overlap_size < max_ctx
            ), f"Stride size {overlap_size} is larger than the maximum context size {max_ctx}"

            (
                all_log_probs,
                offset_mapping,
                accumulated_tokenized_text,
            ) = self._full_context_to_probs_and_offsets(
                full_context, overlap_size, allow_overlap, max_ctx
            )

        # The accumulated_tokenized_text is the text we extract surprisal values for
        # It is after removing the BOS/EOS tokens
        # Make sure the accumulated_tokenized_text is equal to the original target_text
        assert (
            accumulated_tokenized_text
            == self.tokenizer(full_context, add_special_tokens=False)["input_ids"]
        )

        all_log_probs = np.asarray(all_log_probs.cpu())

        target_text_log_probs, target_text_offset_mapping = (
            self._ommit_left_ctx_from_surp_res(
                target_text, all_log_probs, offset_mapping
            )
        )

        return target_text_log_probs, target_text_offset_mapping


# class SoftCatCtxSurpExtractor(SurprisalExtractor):
#     def __init__(
#         self,
#         model_name: str,
#         model_target_device: str = "cpu",
#         pythia_checkpoint: str | None = "step143000",
#         hf_access_token: str | None = None,
#     ):
#         super().__init__(
#             model_name, model_target_device, pythia_checkpoint, hf_access_token
#         )

#     def _get_embedded_context(
#         self, left_context_text: str, device: str
#     ) -> torch.Tensor:
#         # """Helper method to embed the left context using the model."""
#         # with torch.no_grad():
#         #     # Tokenize the left context
#         #     left_context_tokens = self.tokenizer(
#         #         left_context_text, return_tensors="pt", truncation=True
#         #     ).to(device)

#         #     # Get the hidden state for the left context from the model
#         #     left_context_output = self.model(**left_context_tokens, output_hidden_states=True)

#         #     # Get the hidden states for the last layer
#         #     hidden_states = left_context_output.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)

#         #     # Aggregate hidden states by averaging over the sequence length
#         #     left_context_embedding = torch.mean(hidden_states, dim=1)  # Shape: (batch_size, hidden_dim)

#         # return left_context_embedding
#         raise NotImplementedError

#     def _add_left_context_to_target_embds(self, target_embds, left_context_text):


#     def _create_embedding_level_input(self, target_text, ):
#         # Tokenize the target text
#         target_encodings = self.tokenizer(
#             target_text, return_tensors="pt", truncation=True
#         ).to(self.model.device)

#         # Embed the left context (if provided)
#         if left_context_text is not None:
#             left_context_embedding = self._get_embedded_context(
#                 left_context_text, self.model.device
#             )
#         else:
#             # If no left context is provided, use zeros
#             hidden_size = self.model.config.hidden_size


#         # Get the embeddings for the target text
#         target_word_embeddings = self.model.get_input_embeddings(
#             target_encodings["input_ids"]
#         )
#         # Concatenate the left context embedding to each token in the target embeddings
#         repeated_context = left_context_embedding.unsqueeze(1).expand_as(
#             target_word_embeddings
#         )
#         full_embeddings = torch.cat(
#             (repeated_context, target_word_embeddings), dim=1
#         )  # Concatenate at embedding level

#     def surprise(
#         self,
#         target_text: str,
#         left_context_text: str | None = None,
#         overlap_size: int | None = None,
#         allow_overlap: bool = False,
#     ) -> Tuple[np.ndarray, List[Tuple[int]]]:
#         """Calculate the surprisal with the left context embedded and concatenated at the embedding level."""
#         if allow_overlap:
#             assert overlap_size is not None, "overlap_size must be specified"

#         with torch.no_grad():

#             # Calculate log probabilities based on the modified embeddings
#             output = self.model(
#                 inputs_embeds=full_embeddings, labels=target_encodings["input_ids"]
#             )
#             # remove the first token from the logits

#             log_probs = torch.nn.functional.cross_entropy(
#                 output["logits"].view(-1, output["logits"].size(-1)),
#                 target_encodings["input_ids"].view(-1),
#                 reduction="none",
#             )

#         # Convert to numpy for return
#         log_probs = log_probs.cpu().numpy()
#         offset_mapping = target_encodings["offset_mapping"].cpu().tolist()

#         return log_probs, offset_mapping


class SoftCatCtxSentAggSurpExtractor(SurprisalExtractor):
    def process(self, data):
        return f"Processing data in Mode 3: {data}"


# # Mapping the enum to the corresponding classes
# mode_to_class_map = {
#     ProcessingMode.CONCAT_CTX_LEFT: CatCtxLeftSurpExtractor,
#     ProcessingMode.SOFT_CONCAT_CTX_AGG: SoftCatCtxAggSurpExtractor,
#     ProcessingMode.SOFT_CONCAT_CTX_SENT_AGG: SoftCatCtxSentAggSurpExtractor,
# }

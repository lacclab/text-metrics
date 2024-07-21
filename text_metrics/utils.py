"""This module contains the functions for extracting the metrics from the text."""

import string
from collections import defaultdict
from typing import List, Literal, Union

import numpy as np
import pandas as pd
import pkg_resources
import spacy
import torch
from spacy.language import Language
from torch.nn.functional import log_softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    LlamaForCausalLM,
    MambaForCausalLM,
)
from wordfreq import tokenize, word_frequency

CONTENT_WORDS = {
    "PUNCT": False,
    "PROPN": True,
    "NOUN": True,
    "PRON": False,
    "VERB": True,
    "SCONJ": False,
    "NUM": False,
    "DET": False,
    "CCONJ": False,
    "ADP": False,
    "AUX": False,
    "ADV": True,
    "ADJ": True,
    "INTJ": False,
    "X": False,
    "PART": False,
}

REDUCED_POS = {
    "PUNCT": "FUNC",
    "PROPN": "NOUN",
    "NOUN": "NOUN",
    "PRON": "FUNC",
    "VERB": "VERB",
    "SCONJ": "FUNC",
    "NUM": "FUNC",
    "DET": "FUNC",
    "CCONJ": "FUNC",
    "ADP": "FUNC",
    "AUX": "FUNC",
    "ADV": "ADJ",
    "ADJ": "ADJ",
    "INTJ": "FUNC",
    "X": "FUNC",
    "PART": "FUNC",
}


def is_content_word(pos: str) -> bool:
    """
    Checks if the pos is a content word.
    """
    return CONTENT_WORDS.get(pos, False)


def get_reduced_pos(pos: str) -> str:
    """
    Returns the reduced pos tag of the pos tag.
    """
    return REDUCED_POS.get(
        pos, "UNKNOWN"
    )  # TODO Why should there be UNKNOWN? Why not map to a reduced pos?


def get_direction(head_idx: int, word_idx: int) -> str:
    """
    Returns the direction of the word from the head word.
    :param head_idx: int, the head index.
    :param word_idx: int, the word index.
    :return: str, the direction of the word from the head.
    """
    if head_idx > word_idx:
        return "RIGHT"
    elif head_idx < word_idx:
        return "LEFT"
    else:
        return "SELF"


def get_parsing_features(
    text: str,
    spacy_model: Language,
    mode: Literal["keep-first", "keep-all", "re-tokenize"] = "re-tokenize",
) -> pd.DataFrame:
    """
    Extracts the parsing features from the text using spacy.
    :param text: str, the text to extract features from.
    :param spacy_model: the spacy model to use.
    :param mode: type of parsing to use. one of ['keep-first','keep-all','re-tokenize']
    :return: pd.DataFrame, each row represents a word and its parsing features.
            for compressed words (e.g., "don't"),
     each feature has a list of all the sub-words' features.
    """
    features = {}
    doc = spacy_model(text)
    token_idx = 0
    word_idx = 1
    token_idx2word_idx = {}
    spans_to_merge = []
    while token_idx < len(doc):
        token = doc[token_idx]
        accumulated_tokens = []
        while not bool(token.whitespace_) and token_idx < len(doc):
            accumulated_tokens.append((token.i, token))
            token_idx += 1
            if token_idx < len(doc):
                token = doc[token_idx]

        if token_idx < len(doc):
            accumulated_tokens.append((token.i, token))
        token_idx += 1

        if len(accumulated_tokens) > 1:
            start_idx = accumulated_tokens[0][0]
            end_idx = accumulated_tokens[-1][0] + 1
            spans_to_merge.append(doc[start_idx:end_idx])

        if mode == "keep-first" or mode == "keep-all":
            features[word_idx] = accumulated_tokens
            for token in accumulated_tokens:
                token_idx2word_idx[token[0]] = word_idx
            word_idx += 1

    if mode == "re-tokenize":
        with doc.retokenize() as retokenizer:
            for span in spans_to_merge:
                retokenizer.merge(span)
        for word_idx, token in enumerate(doc):
            features[word_idx + 1] = [(token.i, token)]

    res = []
    for word_idx, word in features.items():
        word_features: dict[str, list[list[str] | int | str | None]] = defaultdict(list)
        word_features["Word_idx"] = [word_idx]
        for ind, token in word:
            word_features["Token"].append(token.text)
            word_features["POS"].append(token.pos_)
            word_features["TAG"].append(token.tag_)
            word_features["Token_idx"].append(ind)
            word_features["Relationship"].append(token.dep_)
            word_features["Morph"].append([f for f in token.morph])
            word_features["Entity"].append(
                token.ent_type_ if token.ent_type_ != "" else None
            )
            word_features["Is_Content_Word"].append(is_content_word(token.pos_))
            word_features["Reduced_POS"].append(get_reduced_pos(token.pos_))
            if mode == "keep-first" or mode == "keep-all":
                word_features["Head_word_idx"].append(
                    token_idx2word_idx[token.head.i]
                    if token.head.i in token_idx2word_idx
                    else -1
                )
                word_features["n_Lefts"].append(
                    len([d for d in token.lefts if d.i in token_idx2word_idx])
                )
                word_features["n_Rights"].append(
                    len([d for d in token.rights if d.i in token_idx2word_idx])
                )
                word_features["AbsDistance2Head"].append(
                    abs(token_idx2word_idx[ind] - token_idx2word_idx[token.head.i])
                    if token.head.i in token_idx2word_idx
                    else -1
                )
                word_features["Distance2Head"].append(
                    token_idx2word_idx[ind] - token_idx2word_idx[token.head.i]
                    if token.head.i in token_idx2word_idx
                    else -1
                )
                word_features["Head_Direction"].append(
                    get_direction(
                        token_idx2word_idx[token.head.i], token_idx2word_idx[ind]
                    )
                    if token.head.i in token_idx2word_idx
                    else "UNKNOWN"
                )
            else:
                word_features["Head_word_idx"].append(token.head.i + 1)
                word_features["n_Lefts"].append(token.n_lefts)
                word_features["n_Rights"].append(token.n_rights)
                word_features["AbsDistance2Head"].append(abs(token.head.i - token.i))
                word_features["Distance2Head"].append(token.head.i - token.i)
                word_features["Head_Direction"].append(
                    get_direction(token.head.i, token.i)
                )

        res.append(word_features)

    final_res = pd.DataFrame(res)
    if mode == "keep-all":
        pass

    assert pd.__version__ > "2.1.0", f"""Your pandas version is {pd.__version__}
            Please upgrade pandas to version 2.1.0 or higher to use mode={mode}.
            (requires pd.DataFrame.map)""".replace("\n", "")
    if mode == "keep-first":
        final_res = final_res.map(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )
    elif mode == "re-tokenize":
        final_res = final_res.map(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )

    return final_res


def _get_surp(text: str, tokenizer, model) -> list[tuple[str, float]]:
    """
    Extract surprisal values from model for text tokenized by tokenizer.

    :param text: the  text to get surprisal values for.
    :param model: model used to get surprisal values.
    :param tokenizer: should be compatible with model.
    :return: list of tuples of (subword, surprisal values).
    """
    # text = text  # + tokenizer.eos_token  # add beginning of sentence token
    ids = torch.tensor(tokenizer.encode(text))
    toks = tokenizer.tokenize(text)

    with torch.no_grad():
        outputs = model(ids)

    # log softmax converted to base 2.
    # More numerically stable than -log2(softmax(outputs[0], dim=1))
    log_probs = -(1 / torch.log(torch.tensor(2.0))) * log_softmax(outputs[0], dim=1)

    out = []
    for ind, word_id in enumerate(ids, 0):
        word_log_prob = float(log_probs[ind - 1, word_id])
        out.append((toks[ind], word_log_prob))
    return out


def _join_surp(words: list[str], tok_surps: list[tuple[str, float]]):
    """
    Add up the subword surprisals of each word.

    :param words: list of actual words
    :param tok_surps: list of tuples of (subword, subword surprisal value)
    :return: list of tuples of (word, word surprisal value)
    """
    out = []
    word_surp, word_ind, within_word_position = 0, 0, 0
    word_till_now = ""
    for tok, tok_surp in tok_surps:
        tok_str = tok[1:] if tok.startswith("Ġ") else tok
        tok_str = tok_str.replace("Â", "").replace(
            "âĤ¬", "€"
        )  # Converts back euro and gbp sign
        assert (
            words[word_ind][within_word_position : within_word_position + len(tok_str)]
            == tok_str
        ), (
            words[word_ind][within_word_position : within_word_position + len(tok_str)]
            + "!="
            + tok_str
        )
        word_surp += tok_surp
        within_word_position += len(tok_str)
        word_till_now += tok_str

        if word_till_now == words[word_ind]:
            out.append((words[word_ind], word_surp))
            word_ind += 1
            word_surp, within_word_position = 0, 0
            word_till_now = ""
    assert word_ind == len(words)
    assert len(out) == len(words)
    return out


def init_tok_n_model(
    model_name: str,
    device: str = "cpu",
    pythia_checkpoint: str | None = "step143000",
    hf_access_token: str | None = None,
):
    """This function initializes the tokenizer and model for the specified LLM variant.

    Args:
        model_name (str): the model name. Supported models:
            GPT-2 family:
            "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

            GPT-Neo family:
            "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

            OPT family:
            "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
            "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"

            Pythia family:
            "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
            each with checkpoints specified by training steps:
            "step1", "step2", "step4", ..., "step142000", "step143000"
        device (str, optional): Defaults to 'cpu'.
        pythia_checkpoint (str, optional): The checkpoint for Pythia models. Defaults to 'step143000'.

    Raises:
        ValueError: Unsupported LLM variant

    Returns:
        Tuple[Union[AutoTokenizer, GPTNeoXTokenizerFast],
              Union[AutoModelForCausalLM, GPTNeoXForCausalLM]]: tokenizer, model
    """

    # TODO merge AutoTokenizer/ModelForCausalLM with/without hf_access_token?
    model_variant = model_name.split("/")[-1]

    if any(
        variant in model_variant
        for variant in ["gpt-neo", "gpt", "opt", "mamba", "rwkv"]
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    elif "gpt-neox" in model_variant:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

    elif "Eagle" in model_variant:  # RWKV V5
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif any(variant in model_variant for variant in ["Llama", "Mistral", "gemma"]):
        assert (
            hf_access_token is not None
        ), f"Please provide the HuggingFace access token to load {model_name}"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, token=hf_access_token
        )

    elif "pythia" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=pythia_checkpoint, use_fast=True
        )

    else:
        raise ValueError("Unsupported LLM variant")

    if any(variant in model_variant for variant in ["gpt-neo", "gpt", "opt", "rwkv"]):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    elif "Eagle" in model_variant:  # RWKV
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).to(torch.float32)

    elif "pythia" in model_variant:
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=pythia_checkpoint, device_map="auto"
        )

    elif "mamba" in model_variant:
        # print('loaded with bfloat16')
        model = MambaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            #  torch_dtype=torch.bfloat16
        )

    elif "Llama" in model_variant:
        model = LlamaForCausalLM.from_pretrained(
            model_name, token=hf_access_token, device_map="auto"
        )

    elif any(variant in model_variant for variant in ["gemma-2"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_access_token,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    elif any(variant in model_variant for variant in ["Mistral", "gemma"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_access_token, device_map="auto"
        )

    else:
        raise ValueError("Unsupported LLM variant")

    # model = model.to(device)

    return tokenizer, model


def _create_input_tokens(
    tokenizer: Union[AutoTokenizer, GPTNeoXTokenizerFast],
    encodings: dict,
    start_ind: int,
    is_last_chunk: bool,
    device: str,
):
    try:
        bos_token_added = tokenizer.bos_token_id
    except AttributeError:
        bos_token_added = tokenizer.pad_token_id

    tokens_lst = encodings["input_ids"]
    if is_last_chunk:
        tokens_lst.append(tokenizer.eos_token_id)
    if start_ind == 0:
        tokens_lst = [bos_token_added] + tokens_lst
    tensor_input = torch.tensor(
        [tokens_lst],
        device=device,
    )
    return tensor_input


def _tokens_to_log_probs(
    model: Union[
        AutoModelForCausalLM, GPTNeoXForCausalLM, MambaForCausalLM, LlamaForCausalLM
    ],
    tensor_input: torch.Tensor,
    is_last_chunk: bool,
):
    output = model(tensor_input, labels=tensor_input)
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
    sentence: str,
    model: Union[
        AutoModelForCausalLM, GPTNeoXForCausalLM, MambaForCausalLM, LlamaForCausalLM
    ],
    tokenizer: Union[
        AutoTokenizer,
        GPTNeoXTokenizerFast,
    ],
    model_name: str,
    stride: int = 512,
):
    """This function calculates the surprisal of a sentence using a language model.
    In case the sentence is too long, it is split into chunks while keeping previous context of STRIDE tokens.

    Args:
        sentence (str): The sentence for which the surprisal is calculated
        model (Union[AutoModelForCausalLM, GPTNeoXForCausalLM]): The language model from which the surprisal is calculated
        tokenizer (Union[AutoTokenizer, GPTNeoXTokenizerFast]): The tokenizer for the language model
        model_name (str): The name of the language model (e.g "gpt2", "gpt-neo-125M", "opt-125m", "pythia-1b")
        stride (int, optional): The number of tokens to keep as context. Defaults to 512.

    Returns:
        Tuple[np.ndarray, List[Tuple[int]]]: The surprisal values for each token in the sentence, the offset mapping
        The offset mapping is a list of tuples, where each tuple contains the start and end character index of the token
    """
    with torch.no_grad():
        all_log_probs = torch.tensor([], device=model.device)
        offset_mapping = []
        start_ind = 0
        try:
            max_ctx = model.config.max_position_embeddings
        except AttributeError:
            max_ctx = int(1e6)

        assert (
            stride < max_ctx
        ), f"Stride size {stride} is larger than the maximum context size {max_ctx}"

        # print(max_ctx)
        accumulated_tokenized_text = []
        while True:
            encodings = tokenizer(
                sentence[start_ind:],
                max_length=max_ctx - 2,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            is_last_chunk = (encodings["offset_mapping"][-1][1] + start_ind) == len(
                sentence
            )

            tensor_input = _create_input_tokens(
                tokenizer,
                encodings,
                start_ind,
                is_last_chunk,
                model.device,
            )

            log_probs, shift_labels = _tokens_to_log_probs(
                model, tensor_input, is_last_chunk
            )

            # Handle the case where the sentence is too long for the model
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
                context_length = len(tokenizer.encode(sentence))
                print(
                    f"The context length is too long ({context_length}>{max_ctx}) for {model_name}. Splitting the sentence into chunks with stride {stride}"
                )

            start_ind += encodings["offset_mapping"][-stride - 1][1]

    # The accumulated_tokenized_text is the text we extract surprisal values for
    # It is after removing the BOS/EOS tokens
    # Make sure the accumulated_tokenized_text is equal to the original sentence
    assert (
        accumulated_tokenized_text
        == tokenizer(sentence, add_special_tokens=False)["input_ids"]
    )

    all_log_probs = np.asarray(all_log_probs.cpu())

    assert all_log_probs.shape[0] == len(offset_mapping)

    return all_log_probs, offset_mapping


def get_word_mapping(words: List[str]):
    """Given a list of words, return the offset mapping for each word.

    Args:
        words (List[str]): The list of words

    Returns:
        Tuple[str]: The offset mapping for each word
    """
    offsets = []
    pos = 0
    for w in words:
        offsets.append((pos, pos + len(w)))
        pos += len(w) + 1
    return offsets


def string_to_log_probs(string: str, probs: np.ndarray, offsets: list):
    """Given text and token-level log probabilities, aggregate the log probabilities to word-level log probabilities.
    Note: Assumes there is no whitespace in the end and the beginning of the string.

    Args:
        string (str): The input text
        probs (np.ndarray): Log probabilities for each token
        offsets (list): The offset mapping for each token

    Returns:
        Tuple[List[float], List[Tuple[float]]]: The aggregated log probabilities for each word
    """
    words = string.split()
    agg_log_probs = []
    word_mapping = get_word_mapping(words)
    cur_prob = 0
    cur_word_ind = 0
    for i, (lp, ind) in enumerate(zip(probs, offsets)):
        cur_prob += lp
        if ind[1] == word_mapping[cur_word_ind][1]:
            # this handles the case in which there are multiple tokens for the same word
            if i < len(probs) - 1 and offsets[i + 1][1] == ind[1]:
                continue
            agg_log_probs.append(cur_prob)
            cur_prob = 0
            cur_word_ind += 1

    zipped_surp = list(zip(words, agg_log_probs))
    return agg_log_probs, zipped_surp


# Credits: https://github.com/byungdoh/llm_surprisal/blob/eacl24/get_llm_surprisal.py
# https://github.com/rycolab/revisiting-uid/blob/0b60df7e8f474d9c7ac938e7d8a02fda6fc8787a/src/language_modeling.py#L136
def get_surprisal(
    text: str,
    tokenizer: Union[AutoTokenizer, GPTNeoXTokenizerFast],
    model: Union[
        AutoModelForCausalLM, GPTNeoXForCausalLM, MambaForCausalLM, LlamaForCausalLM
    ],
    model_name: str,
    context_stride: int = 512,
) -> pd.DataFrame:
    """
    Get surprisal values for each word in text.

    Words are split by white space, and include adjacent punctuation.
    A surprisal of a word is the sum of the surprisal of the subwords
    (as split by the tokenizer) that make up the word.

    :param text: str, the text to get surprisal values for.
    :param model: the model to extract surprisal values from.
    :param tokenizer: how to tokenize the text. Should match the model input expectations.
    :return: pd.DataFrame, each row represents a word and its surprisal.

    >>> tokenizer = AutoTokenizer.from_pretrained('gpt2')
    >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
    >>> text = "hello, how are you?"
    >>> surprisals = get_surprisal(text=text,tokenizer=tokenizer,model=model)
    >>> surprisals
         Word  Surprisal
    0  hello,  19.789963
    1     how  12.335088
    2     are   5.128458
    3    you?   3.704563
    """

    probs, offset_mapping = surprise(
        text, model, tokenizer, model_name, stride=context_stride
    )
    return pd.DataFrame(
        string_to_log_probs(text, probs, offset_mapping)[1],
        columns=["Word", "Surprisal"],
    )


def get_frequency(text: str) -> pd.DataFrame:
    """
    Get (negative log2) frequencies for each word in text.

    Words are split by white space.
    A frequency of a word does not include adjacent punctuation.
    Half harmonic mean is applied for complex words.
    E.g. freq(top-level) = 1/(1/freq(top) + 1/freq(level))

    :param text: str, the text to get frequencies for.
    :return: pd.DataFrame, each row represents a word and its frequency.

    >>> text = "hello, how are you?"
    >>> frequencies = get_frequency(text=text)
    >>> frequencies
         Word  Wordfreq_Frequency  subtlex_Frequency
    0  hello,           14.217323          10.701528
    1     how            9.166697           8.317353
    2     are            7.506353           7.548023
    3    you?            6.710284           4.541699
    """
    words = text.split()
    frequencies = {
        "Word": words,
        "Wordfreq_Frequency": [
            -np.log2(word_frequency(word, lang="en", minimum=1e-11)) for word in words
        ],  # minimum equal to ~36.5
    }
    # TODO improve loading of file according to https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package
    #  and https://setuptools.pypa.io/en/latest/userguide/datafiles.html
    data = pkg_resources.resource_stream(
        __name__, "data/SUBTLEXus74286wordstextversion_lower.tsv"
    )
    subtlex = pd.read_csv(
        data,
        sep="\t",
        index_col=0,
    )
    subtlex["Frequency"] = -np.log2(subtlex["Count"] / subtlex.sum().iloc[0])

    #  TODO subtlex freq should be 'inf' if missing, not zero?
    subtlex_freqs = []
    for word in words:
        tokens = tokenize(word, lang="en")
        one_over_result = 0.0
        try:
            for token in tokens:
                one_over_result += 1.0 / subtlex.loc[token, "Frequency"]
        except KeyError:
            subtlex_freq = 0
        else:
            subtlex_freq = 1.0 / one_over_result if one_over_result != 0 else 0
        subtlex_freqs.append(subtlex_freq)
    frequencies["subtlex_Frequency"] = subtlex_freqs

    return pd.DataFrame(frequencies)


def get_word_length(text: str, disregard_punctuation: bool = True) -> pd.DataFrame:
    """
    Get the length of each word in text.

    Words are split by white space.

    :param text: str, the text to get lengths for.
    :param disregard_punctuation: bool, to include adjacent punctuation (False) or not (True).
    :return: pd.DataFrame, each row represents a word and its length.

    Examples
    --------
    >>> text = "hello, how are you?"
    >>> lengths = get_word_length(text=text, disregard_punctuation=True)
    >>> lengths
         Word  Length
    0  hello,       5
    1     how       3
    2     are       3
    3    you?       3

    >>> text = "hello, how are you?"
    >>> lengths = get_word_length(text=text, disregard_punctuation=False)
    >>> lengths
         Word  Length
    0  hello,       6
    1     how       3
    2     are       3
    3    you?       4


    """
    word_lengths = {
        "Word": text.split(),
    }
    if disregard_punctuation:
        #     text = text.translate(str.maketrans('', '', string.punctuation))
        word_lengths["Length"] = [
            len(word.translate(str.maketrans("", "", string.punctuation)))
            for word in text.split()
        ]
    else:
        word_lengths["Length"] = [len(word) for word in text.split()]

    return pd.DataFrame(word_lengths)


def clean_text(raw_text: str) -> str:
    """
    Replaces the problematic characters in the raw_text, made for OnestopQA.
    E.g., "ë" -> "e"
    """
    return (
        raw_text.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("…", "...")
        .replace("‘", "'")
        .replace("é", "e")
        .replace("ë", "e")
        .replace("ﬁ", "fi")
        .replace("ï", "i")
    )


def get_metrics(
    text: str,
    models: list[
        AutoModelForCausalLM, GPTNeoXForCausalLM, MambaForCausalLM, LlamaForCausalLM
    ],
    tokenizers: List[AutoTokenizer],
    model_names: List[str],
    parsing_model: spacy.Language | None,
    parsing_mode: (
        Literal["keep-first", "keep-all", "re-tokenize"] | None
    ) = "re-tokenize",
    add_parsing_features: bool = True,
    context_stride: int = 512,
) -> pd.DataFrame:
    """
    Wrapper function to get the surprisal and frequency values and length of each word in the text.

    :param text: str, the text to get metrics for.
    :param model: the model to extract surprisal values from.
    :param tokenizer: how to tokenize the text. Should match the model input expectations.
    :param parsing_model: the spacy model to use for parsing the text.
    :param parsing_mode: type of parsing to use. one of ['keep-first','keep-all','re-tokenize']
    :param add_parsing_features: whether to add parsing features to the output.
    :return: pd.DataFrame, each row represents a word, its surprisal and frequency.


    >>> tokenizer = AutoTokenizer.from_pretrained('gpt2')
    >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
    >>> text = "hello, how are you?"
    >>> words_with_metrics = get_metrics(text=text,tokenizers=[tokenizer],models=[model], model_names=['gpt2'])
    >>> words_with_metrics
            Word  Length  Wordfreq_Frequency  subtlex_Frequency  gpt2_Surprisal
    0  hello,       5           14.217323          10.701528       19.789963
    1     how       3            9.166697           8.317353       12.335088
    2     are       3            7.506353           7.548023        5.128458
    3    you?       3            6.710284           4.541699        3.704563
    """

    text_reformatted = clean_text(text)
    surprisals = []
    for model, tokenizer, model_name in zip(models, tokenizers, model_names):
        surprisal = get_surprisal(
            text=text_reformatted,
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            context_stride=context_stride,
        )

        surprisal.rename(columns={"Surprisal": f"{model_name}_Surprisal"}, inplace=True)
        surprisals.append(surprisal)

    frequency = get_frequency(text=text_reformatted)
    word_length = get_word_length(text=text_reformatted, disregard_punctuation=True)

    merged_df = word_length.join(frequency.drop("Word", axis=1))
    for surprisal in surprisals:
        merged_df = merged_df.join(surprisal.drop("Word", axis=1))

    if add_parsing_features:
        assert (
            parsing_model is not None
        ), "Please provide a parsing model to extract parsing features."
        assert (
            parsing_mode is not None
        ), "Please provide a parsing mode to extract parsing features."

        parsing_features = get_parsing_features(
            text_reformatted, parsing_model, parsing_mode
        )
        merged_df = merged_df.join(parsing_features)

    return merged_df


if __name__ == "__main__":
    text = (
        """
    113, 115, 117, and 118 are ... The International Union of Pure and Applied Chemistry (IUPAC) is the global organization that controls
    chemical names. IUPAC confirmed the new elements on 30 December, 2015 after examining studies dating back to 2004. 
    The scientists who found them must now think of formal names for the elements, which have the atomic numbers,
    113, 115, 117, and 118. The atomic number is the number of protons in an element’s atomic nucleus.
    """.replace("\n", "")
        .replace("\t", "")
        .replace("    ", "")
    )
    model_names = ["gpt2", "gpt2-medium"]

    models_tokenizers = [init_tok_n_model(model_name) for model_name in model_names]
    tokenizers = [tokenizer for tokenizer, _ in models_tokenizers]
    models = [model for _, model in models_tokenizers]

    surp_res = get_metrics(
        text=text,
        models=models,
        tokenizers=tokenizers,
        model_names=model_names,
        parsing_model=spacy.load("en_core_web_sm"),
        add_parsing_features=False,
    )
    
    print(surp_res.head(10).to_markdown())
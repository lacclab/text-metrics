import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from wordfreq import word_frequency, tokenize
import numpy as np
import string
import pkg_resources
from typing import List
import pandas as pd
import spacy
from collections import defaultdict

content_word_dict = {
    'PUNCT': 'NO_CONTENT',
    'PROPN': 'CONTENT',
    'NOUN': 'CONTENT',
    'PRON': 'NO_CONTENT',
    'VERB': 'CONTENT',
    'SCONJ': 'NO_CONTENT',
    'NUM': 'NO_CONTENT',
    'DET': 'NO_CONTENT',
    'CCONJ': 'NO_CONTENT',
    'ADP': 'NO_CONTENT',
    'AUX': 'NO_CONTENT',
    'ADV': 'CONTENT',
    'ADJ': 'CONTENT',
    'INTJ': 'NO_CONTENT',
    'X': 'NO_CONTENT',
    'PART': 'NO_CONTENT',
    'NaN': 'UNKNOWN',
}

reduced_pos_dict = {
    'PUNCT': 'FUNC',
    'PROPN': 'NOUN',
    'NOUN': 'NOUN',
    'PRON': 'FUNC',
    'VERB': 'VERB',
    'SCONJ': 'FUNC',
    'NUM': 'FUNC',
    'DET': 'FUNC',
    'CCONJ': 'FUNC',
    'ADP': 'FUNC',
    'AUX': 'FUNC',
    'ADV': 'ADJ',
    'ADJ': 'ADJ', 'INTJ': 'FUNC',
    'X': 'FUNC',
    'PART': 'FUNC',
    'NaN': 'UNKNOWN',
}


def is_content_word(pos: str) -> bool:
    """
    Checks if the pos is a content word.
    """
    if pos in content_word_dict.keys():
        return content_word_dict[pos] == 'CONTENT'
    return False


def get_reduced_pos(pos: str) -> str:
    """
    Returns the reduced pos tag of the pos tag.
    """
    if pos in reduced_pos_dict.keys():
        return reduced_pos_dict[pos]
    return "UNKNOWN"

def get_direction(head_idx: int, word_idx: int) -> str:
    """
    Returns the direction of the word from the head word.
    :param head_idx: int, the head index.
    :param word_idx: int, the word index.
    :return: str, the direction of the word from the head.
    """
    if head_idx > word_idx:
        return 'RIGHT'
    elif head_idx < word_idx:
        return 'LEFT'
    else:
        return 'SELF'


def get_parsing_features(text: str, nlp_model: spacy.Language) -> pd.DataFrame:
    """
    Extracts the parsing features from the text using spacy.
    :param text: str, the text to extract features from.
    :param nlp_model: the spacy model to use.
    :return: pd.DataFrame, each row represents a word and its parsing features. for compressed words (e.g., "don't"),
     each feature has a list of all the sub-words' features.
    """
    features = {}
    doc = nlp_model(text)
    token_idx = 0
    word_idx = 1
    token_idx2word_idx = {}
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

        features[word_idx] = accumulated_tokens
        for token in accumulated_tokens:
            token_idx2word_idx[token[0]] = word_idx
        word_idx += 1

    words = text.split()
    res = []
    for word_idx, word in features.items():
        word_features = defaultdict(list)
        for ind, token in word:
            word_features['Token'].append(token.text)
            word_features['POS'].append(token.pos_)
            word_features['TAG'].append(token.tag_)
            word_features['Token_idx'].append(ind)
            word_features['Head_word_idx'].append(
                token_idx2word_idx[token.head.i] if token.head.i in token_idx2word_idx.keys() else -1)
            word_features['Relationship'].append(token.dep_)
            word_features['n_Lefts'].append(len([d for d in token.lefts if d.i in token_idx2word_idx.keys()]))
            word_features['n_Rights'].append(len([d for d in token.rights if d.i in token_idx2word_idx.keys()]))
            word_features['Distance2Head'].append(abs(token_idx2word_idx[ind] - token_idx2word_idx[
                token.head.i]) if token.head.i in token_idx2word_idx.keys() else -1)
            word_features['Head_Direction'].append(get_direction(token_idx2word_idx[token.head.i], token_idx2word_idx[ind]) if token.head.i in token_idx2word_idx.keys() else 'UNKNOWN')
            word_features['Morph'].append([f for f in token.morph])
            word_features['Entity'].append(token.ent_type_ if token.ent_type_ != '' else None)
            word_features['Is_Content_Word'].append(is_content_word(token.pos_))
            word_features['Reduced_POS'].append(get_reduced_pos(token.pos_))

        first_token_features = {}
        first_token_features['Word'] = words[word_idx-1]
        first_token_features['Word_idx'] = word_idx
        for feature, values_list in word_features.items():
            first_token_features[feature] = values_list
        res.append(first_token_features)

    return pd.DataFrame(res)


def _get_surp(text: str, tokenizer, model) -> list[tuple[str, float]]:
    """
    Extract surprisal values from model for text tokenized by tokenizer.

    :param text: the  text to get surprisal values for.
    :param model: model used to get surprisal values.
    :param tokenizer: should be compatible with model.
    :return: list of tuples of (subword, surprisal values).
    """
    text = text  # + tokenizer.eos_token  # add beginning of sentence token
    ids = torch.tensor(tokenizer.encode(text))
    toks = tokenizer.tokenize(text)

    with torch.no_grad():
        outputs = model(ids)

    # log softmax converted to base 2. More numerically stable than -log2(softmax(outputs[0], dim=1))
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


def get_surprisal(text: str, tokenizer, model) -> pd.DataFrame:
    """
    Get surprisal values for each word in text.

    Words are split by white space, and include adjacent punctuation.
    A surprisal of a word is the sum of the surprisal of the subwords (as split by the tokenizer) that make up the word.

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

    tok_surps = _get_surp(text, tokenizer, model)
    word_surps = _join_surp(text.split(), tok_surps)  # [:-1])
    return pd.DataFrame(word_surps, columns=["Word", "Surprisal"])


def get_frequency(text: str) -> pd.DataFrame:
    """
    Get (negative log2) frequencies for each word in text.

    Words are split by white space.
    A frequency of a word does not include adjacent punctuation.
    Half harmonic mean is applied for complex words. E.g. freq(top-level) = 1/(1/freq(top) + 1/freq(level))

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
    subtlex["Frequency"] = -np.log2(subtlex["Count"] / subtlex.sum()[0])

    # TODO subtlex freq should be 'inf' if missing, not zero?
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
    :param disregard_punctuation: bool, controls whether to include adjacent punctuation (False) or not (True).
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
    models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    model_names: List[str],
    parsing_model: spacy.Language
) -> pd.DataFrame:
    """
    Wrapper function to get the surprisal and frequency values and length of each word in the text.

    :param text: str, the text to get metrics for.
    :param model: the model to extract surprisal values from.
    :param tokenizer: how to tokenize the text. Should match the model input expectations.
    :param parsing_model: the spacy model to use for parsing the text.
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
            text=text_reformatted, tokenizer=tokenizer, model=model
        )
        surprisal.rename(columns={"Surprisal": f"{model_name}_Surprisal"}, inplace=True)
        surprisals.append(surprisal)

    frequency = get_frequency(text=text_reformatted)
    word_length = get_word_length(text=text_reformatted, disregard_punctuation=True)

    merged_df = word_length.join(frequency.drop("Word", axis=1))
    for surprisal in surprisals:
        merged_df = merged_df.join(surprisal.drop("Word", axis=1))


    # Add here the other metrics - per word in the given paragraph
    parsing_features = get_parsing_features(text_reformatted, parsing_model)
    merged_df = merged_df.join(parsing_features.drop("Word", axis=1))


    return merged_df


if __name__ == "__main__":
    model_names = ["gpt2", "gpt2"]
    input_text = "hello, how are you?"
    words_with_metrics = get_metrics(
        text=input_text, surprisal_extraction_model_names=model_names
    )
    print(words_with_metrics)

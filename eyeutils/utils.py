import pandas as pd
import torch
from torch.nn.functional import log_softmax
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from wordfreq import word_frequency, tokenize
import numpy as np
import string
import pkg_resources


def _get_surp(text: str, tokenizer, model) -> list[tuple[str, float]]:
    """
    Extract surprisal values from model for text tokenized by tokenizer.

    :param text: the  text to get surprisal values for.
    :param model: model used to get surprisal values.
    :param tokenizer: should be compatible with model.
    :return: list of tuples of (subword, surprisal values).
    """
    text = tokenizer.bos_token + ' ' + text  # add beginning of sentence token
    ids = torch.tensor(tokenizer.encode(text))
    toks = tokenizer.tokenize(text)

    with torch.no_grad():
        outputs = model(ids)

    # log softmax converted to base 2. More numerically stable than -log2(softmax(outputs[0], dim=1))
    log_probs = - (1 / torch.log(torch.tensor(2.))) * log_softmax(outputs[0], dim=1)

    out = []
    for ind, word_id in enumerate(ids[1:], 1):
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
        tok_str = tok[1:] if tok.startswith('Ġ') else tok
        tok_str = tok_str.replace("Â", "").replace("âĤ¬", "€")  # Converts back euro and gbp sign
        assert (words[word_ind][within_word_position:within_word_position + len(tok_str)] == tok_str), \
            words[word_ind][within_word_position:within_word_position + len(tok_str)] + '!=' + tok_str
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

    >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
    >>> text = "hello, how are you?"
    >>> surprisals = get_surprisal(text=text,tokenizer=tokenizer,model=model)
    >>> surprisals
         Word  Surprisal
    0  hello,  23.840695
    1     how   6.321535
    2     are   1.971676
    3    you?   2.309872
    """

    tok_surps = _get_surp(text, tokenizer, model)
    word_surps = _join_surp(text.split(), tok_surps)
    return pd.DataFrame(word_surps, columns=['Word', 'Surprisal'])


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
        'Word': words,
        'Wordfreq_Frequency': [-np.log2(word_frequency(word, lang='en')) for word in words],
    }

    data = pkg_resources.resource_stream(__name__, "data/SUBTLEXus74286wordstextversion_lower.tsv")
    subtlex = pd.read_csv(data, sep='\t', index_col=0, )
    subtlex['Frequency'] = -np.log2(subtlex['Count'] / subtlex.sum()[0])

    subtlex_freqs = []
    for word in words:
        tokens = tokenize(word, lang='en')
        one_over_result = 0.0
        try:
            for token in tokens:
                one_over_result += 1.0 / subtlex.loc[token, 'Frequency']
        except KeyError:
            subtlex_freq = 0
        else:
            subtlex_freq = 1.0 / one_over_result if one_over_result != 0 else 0
        subtlex_freqs.append(subtlex_freq)
    frequencies['subtlex_Frequency'] = subtlex_freqs

    return pd.DataFrame(frequencies)


def get_word_length(text: str, disregard_punctuation: bool=True) -> pd.DataFrame:
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
        'Word': text.split(),
    }
    if disregard_punctuation:
        #     text = text.translate(str.maketrans('', '', string.punctuation))
        word_lengths['Length'] = [len(word.translate(str.maketrans('', '', string.punctuation))) for word in text.split()]
    else:
        word_lengths['Length'] = [len(word) for word in text.split()]

    return pd.DataFrame(word_lengths)


def clean_text(raw_text: str) -> str:
    """
    Replaces the problematic characters in the text.
    """
    return raw_text \
        .replace('’', "'") \
        .replace("“", "\"") \
        .replace("”", "\"") \
        .replace("–", "-") \
        .replace("…", "...") \
        .replace("‘", "'") \
        .replace("é", "e") \
        .replace("ë", "e") \
        .replace("ﬁ", "fi") \
        .replace("ï", "i")


def get_metrics(text: str, tokenizer, model) -> pd.DataFrame:
    """
    Wrapper function to get the surprisal and frequency values and length of each word in the text.

    :param text: str, the text to get metrics for.
    :param model: the model to extract surprisal values from.
    :param tokenizer: how to tokenize the text. Should match the model input expectations.
    :return: pd.DataFrame, each row represents a word, its surprisal and frequency.

    >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
    >>> text = "hello, how are you?"
    >>> words_with_metrics = get_metrics(text=text,tokenizer=tokenizer,model=model)
    >>> words_with_metrics
         Word  Surprisal  Wordfreq_Frequency  subtlex_Frequency  Length
    0  hello,  23.840695           14.217323          10.701528       5
    1     how   6.321535            9.166697           8.317353       3
    2     are   1.971676            7.506353           7.548023       3
    3    you?   2.309872            6.710284           4.541699       3
    """

    text_reformatted = clean_text(text)
    surprisals = get_surprisal(text=text_reformatted, tokenizer=tokenizer, model=model)
    frequency = get_frequency(text=text_reformatted)
    word_length = get_word_length(text=text_reformatted, disregard_punctuation=True)
    merged_df = surprisals.join(frequency.drop('Word', axis=1))
    merged_df = merged_df.join(word_length.drop('Word', axis=1))
    return merged_df


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    input_text = "hello, the how are you top-level?"
    words_with_metrics = get_metrics(text=input_text, tokenizer=tokenizer, model=model)
    print(words_with_metrics)

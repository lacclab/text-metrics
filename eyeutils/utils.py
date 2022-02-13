import pandas as pd
import torch
from torch.nn.functional import log_softmax
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from wordfreq import word_frequency, tokenize, zipf_frequency
import math


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
        tok_str = tok_str.replace("Â", "").replace("âĤ¬", "€")  # TODO is this okay?
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
    Get frequencies for each word in text.

    Words are split by white space.
    A frequency of a word does not include adjacent punctuation.

    :param text: str, the text to get frequencies for.
    :return: pd.DataFrame, each row represents a word and its frequency.

    >>> text = "hello, how are you?"
    >>> frequencies = get_frequency(text=text)
    >>> frequencies
         Word  Frequency  MinusLog2Frequency  Zipf_Frequency
    0  hello,   0.000053           14.217323            4.72
    1     how   0.001740            9.166697            6.24
    2     are   0.005500            7.506353            6.74
    3    you?   0.009550            6.710284            6.98
    """

    frequencies = {
        'Word':[],
        'Frequency':[],
        'MinusLog2Frequency':[],
        'Zipf_Frequency':[],
                   }
    for word in text.split():
        frequencies['Word'].append(word)
        tokenized_word = tokenize(word, lang='en')
        if word=='24/7':
            tokenized_word = [word]
        # Todo with temporarily captures word with '-' which are split into two word by the tokenizer
        if len(tokenized_word) != 1:
            frequencies['Frequency'].append(0)
            frequencies['MinusLog2Frequency'].append(0)
            frequencies['Zipf_Frequency'].append(0)
        else:
            freq = word_frequency(tokenized_word[0], lang='en')
            frequencies['Frequency'].append(freq)
            frequencies['MinusLog2Frequency'].append(-math.log(freq, 2) if freq != 0 else 0)
            frequencies['Zipf_Frequency'].append(zipf_frequency(tokenized_word[0], lang='en'))

    return pd.DataFrame(frequencies)


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
    Wrapper function to get the surprisal and frequency values of each word in the text.

    :param text: str, the text to get metrics for.
    :param model: the model to extract surprisal values from.
    :param tokenizer: how to tokenize the text. Should match the model input expectations.
    :return: pd.DataFrame, each row represents a word, its surprisal and frequency.

    >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
    >>> text = "hello, how are you?"
    >>> words_with_metrics = get_metrics(text=text,tokenizer=tokenizer,model=model)
    >>> words_with_metrics
         Word  Surprisal  Frequency  MinusLog2Frequency  Zipf_Frequency
    0  hello,  23.840695   0.000053           14.217323            4.72
    1     how   6.321535   0.001740            9.166697            6.24
    2     are   1.971676   0.005500            7.506353            6.74
    3    you?   2.309872   0.009550            6.710284            6.98
    """

    text_reformatted = clean_text(text)
    surprisals = get_surprisal(text=text_reformatted, tokenizer=tokenizer, model=model)
    frequency = get_frequency(text=text_reformatted)
    merged_df = surprisals.join(frequency.drop('Word', axis=1))
    return merged_df


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    text = "hello, the how are you?"
    words_with_metrics = get_metrics(text=text, tokenizer=tokenizer, model=model)
    print(words_with_metrics)

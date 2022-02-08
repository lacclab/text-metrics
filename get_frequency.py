from wordfreq import word_frequency, tokenize
import pandas as pd


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
         Word  Frequency
    0  hello,   0.000053
    1     how   0.001740
    2     are   0.005500
    3    you?   0.009550
    """

    parsed_paragraph = tokenize(text, lang='en')
    frequencies = {'Word': text.split(),
                   'Frequency': [word_frequency(word, lang='en') for word in parsed_paragraph]
                   }
    return pd.DataFrame(frequencies)

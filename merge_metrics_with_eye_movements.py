from datasets import load_dataset
from get_surprisal import get_surprisal
from get_frequency import get_frequency
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd


def clean_text(raw_text: str) -> str:
    """
    Replaces the problematic characters in the text.
    """
    # TODO is this okay?
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
         Word  Surprisal  Frequency
    0  hello,  23.840695   0.000053
    1     how   6.321535   0.001740
    2     are   1.971676   0.005500
    3    you?   2.309872   0.009550
    """

    text_reformatted = clean_text(text)
    surprisals = get_surprisal(text=text_reformatted, tokenizer=tokenizer, model=model)
    frequency = get_frequency(text=text_reformatted)
    merged_df = pd.concat([surprisals, frequency[['Frequency']]], axis=1)
    return merged_df


# %%
if __name__ == '__main__':

    data = pd.DataFrame(load_dataset(path='onestop_qa', split='train'))

    # Surprisal parameters
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # TODO - complete merge of metrics with eye movements. Currently just prints to screen the values.
    for indx, paragraph in enumerate(data['paragraph']):
        print(indx, paragraph)
        merged_df = get_metrics(text=paragraph, tokenizer=tokenizer, model=model)
        print(merged_df)
        break

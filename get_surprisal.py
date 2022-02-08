import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.functional import log_softmax
import pandas as pd


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


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import pandas as pd

def gpt2_surp(text, tokenizer, model):
    text = tokenizer.bos_token + ' ' + text  # add beginning of sentence token
    ids = torch.tensor(tokenizer.encode(text))
    toks = tokenizer.tokenize(text)
    with torch.no_grad():
        outputs = model(ids)
    log_probs = -torch.log2(torch.nn.functional.softmax(outputs[0], dim=1))
    out = []
    for ind, word_id in enumerate(ids[1:], 1):
        word_log_prob = float(log_probs[ind - 1, word_id])
        out.append((toks[ind], word_log_prob))
    return out


def join_surp(words, tok_surps):
    '''add up the subword surprisals of each word'''
    out = []
    word_surp, word_ind, within_word_position = 0, 0, 0
    for tok, tok_surp in tok_surps:
        tok_str = tok[1:] if tok.startswith('Ġ') else tok
        assert (words[word_ind][within_word_position:within_word_position + len(tok_str)] == tok_str), words[word_ind][
                                                                                                       within_word_position:within_word_position + len(
                                                                                                           tok_str)] + '!=' + tok_str
        word_surp += tok_surp
        within_word_position += len(tok_str)
        if within_word_position == len(words[word_ind]):
            out.append((words[word_ind], word_surp))
            word_ind += 1
            word_surp, within_word_position = 0, 0
    assert word_ind == len(words)
    assert len(out) == len(words)
    return out


def get_gpt2_surp(text, tokenizer, model):
    tok_surps = gpt2_surp(text, tokenizer, model)
    return join_surp(text.split(), tok_surps)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

data = pd.DataFrame(load_dataset(path='onestop_qa', split='train'))

paragraph = data.loc[0, 'paragraph']
#%%
paragraph_reformatted = paragraph.replace('’', "'")
surprisals = get_gpt2_surp(paragraph_reformatted, tokenizer, model)
for word, surprisal in surprisals:
    print(word, "\t", surprisal)
print('\n')


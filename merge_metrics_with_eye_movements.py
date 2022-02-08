import pandas as pd
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils import get_metrics

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

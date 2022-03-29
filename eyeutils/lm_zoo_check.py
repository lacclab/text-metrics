import subprocess

import pandas as pd
import lm_zoo as Z


def _get_suprisal(text_path: str, model):
    """
    Extract surprisal values from model for text tokenized by tokenizer.

    :param text: the  text to get surprisal values for.
    :param model: model used to get surprisal values.
    """

    #
    # text_file = open("sample.txt", "w")

    # n = text_file.write(text_file)
    # text_file.close()
    df = Z.get_surprisals(Z.get_registry()['gpt2'], ['hello, how are you?'])
    # command = 'lm-zoo get-surprisals ' + model + ' ' + text_path

    # output = subprocess.getoutput(command)

    # data = output
    # df = pd.DataFrame([x.split('\t') for x in data.split('\n')])

    return df

if __name__ == '__main__':
    df = _get_suprisal(text_path='data/sample_text.txt', model='gpt2')
    print(df)
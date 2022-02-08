# Text Metrics
Useful utils for extracting frequency and surprisal from text.

Supports  retrieval of surprisal, frequency or both.

The functions `get_frequency`,`get_surprisal` and`get_metrics` (for both metrics) 
can be used to retrieve these values for any piece of text.
 E.g., running
```python
from utils import get_metrics
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
text = "Hello, how are you?"

words_with_metrics = get_metrics(text=text,tokenizer=tokenizer,model=model)
words_with_metrics
```
Should result in

| Word   |  Surprisal  |  Frequency  |
|:-------|:-----------:|:-----------:|
| Hello, |   21.9386   |  5.25e-05   |
| how    |   8.31841   |   0.00174   |
| are    |    1.241    |   0.0055    |
| you?   |   1.85489   |   0.00955   |


`merge_metrics_with_eye_movements.py` takes the [OneStopQA](https://huggingface.co/datasets/onestop_qa) data from HuggingFace,
retrieves surprisal and frequency for each word and merges it with the eye movement report (TODO - complete).


## Surprisal 
- Surprisal values are extracted from the `model` (and corresponding `tokenizer`).
- Words are split by white space, and include adjacent punctuation.
- A surprisal of a word is the sum of the surprisal of the subwords (as split by the tokenizer) that make up the word.

## Frequency
Frequency is extracted via the [wordfreq](https://github.com/rspeer/wordfreq) package. 

- Words are split by white space.
- A frequency of a word does not include adjacent punctuation.
- Frequency is given by the figure-skating metric, from multiple sources.
- See the package documentation for more in-depth descriptions.

## Setup

Tested with package versions:

- pandas 1.3.4
- python 3.9.7
- wordfreq 2.5.1
- transformers 4.12.0
- pytorch 1.10.0

## TODOs

- Handling Special Characters
  - Before tokenization
    ```python
    raw_text \
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
    ```
  - After surprisal tokenization
    ```python
    tok_str = tok[1:] if tok.startswith('Ġ') else tok
    tok_str = tok_str.replace("Â", "").replace("âĤ¬", "€") 
    ```
  - eye tracking report - duplicate rows for words with '-'.
  - frequency - tokenization of - and ...

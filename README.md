# Text Metrics

Useful utils for extracting frequency and surprisal from text.

Supports  retrieval of surprisal, frequency or both.

The functions `get_frequency`,`get_surprisal` and`get_metrics` (for both metrics)
can be used to retrieve these values for any piece of text.
 E.g., running

```python
from eyeutils.utils import get_metrics

text = "Hello, how are you?"

words_with_metrics = get_metrics(text=text,surprisal_extraction_model_names=['gpt2']) # Not updated for new API
print(words_with_metrics)
```

Should result in

| Word   |  gpt2_Surprisal  |  Frequency  |
|:-------|:-----------:|:-----------:|
| Hello, |   21.9386   |  5.25e-05   |
| how    |   8.31841   |   0.00174   |
| are    |    1.241    |   0.0055    |
| you?   |   1.85489   |   0.00955   |

`merge_metrics_with_eye_movements.py` takes the [OneStopQA](https://huggingface.co/datasets/onestop_qa) data from HuggingFace,
retrieves surprisal and frequency for each word and merges it with the eye movement report.

`merge_metrics_with_tabular_text.py` takes the a tabular text file (e.g., a csv file),
retrieves surprisal and frequency for each word, by item (sentence/paragraph) and merges it with the tabular text file. 

Note, in both functions the columns are hard-coded, so you may need to change them to match your data.

```python
from eyeutils.merge_metrics_with_eye_movements import add_metrics_to_eye_tracking

et_data = load_et_data() # Some function to load the eye tracking report
# Surprisal from GPT2, for example
et_data_enriched = add_metrics_to_eye_tracking(eye_tracking_data=et_data, surprisal_extraction_model_names=['gpt2'])
```

## Surprisal

- Surprisal values are extracted from  `transformers.AutoModelForCausalLM` (and corresponding `AutoTokenizer`). For a partial list of supported models, see [here](https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#transformers.AutoModelForCausalLM.from_pretrained).
- Words are split by white space, and include adjacent punctuation.
- A surprisal of a word is the sum of the surprisal of the subwords (as split by the tokenizer) that make up the word.

## Frequency

Frequency is extracted via the [wordfreq](https://github.com/rspeer/wordfreq) package.

- Words are split by white space.
- A frequency of a word does not include adjacent punctuation.
- Frequency is given by the figure-skating metric, from multiple sources.
- See the package documentation for more in-depth descriptions.

## Setup

To install the package -
`pip install git+https://github.com/lacclab/text-metrics.git`.

To make changes - clone to your local station.

Tested with package versions:

- pandas 1.3.4
- python 3.9.7
- wordfreq 3.0.3
- transformers 4.12.0
- pytorch 1.10.0

Run `python -m doctest -v text_metrics/utils.py` before committing to ensure that the docstrings are up to date.

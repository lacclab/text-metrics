# Text Metrics

Useful utils for extracting frequency and surprisal from text.

Supports retrieval of surprisal, frequency or both.

The functions `get_frequency`,`get_surprisal` and`get_metrics` (for both metrics)
can be used to retrieve these values for any piece of text.
E.g., running

```python
from eyeutils.utils import get_metrics

text = "113, 115, 117, and 118 are ... The International Union"

model_names = ["gpt2", "gpt2-medium"]

models_tokenizers = [init_tok_n_model(model_name) for model_name in model_names]
tokenizers = [tokenizer for tokenizer, _ in models_tokenizers]
models = [model for _, model in models_tokenizers]

surp_res = get_metrics(
    text=text,
    models=models,
    tokenizers=tokenizers,
    model_names=model_names,
    parsing_model=spacy.load("en_core_web_sm"),
    add_parsing_features=False,
)
```

Should result in
| | Word | Length | Wordfreq_Frequency | subtlex_Frequency | gpt2_Surprisal | gpt2-medium_Surprisal |
|---:|:--------------|---------:|---------------------:|--------------------:|-----------------:|------------------------:|
| 0 | 113, | 3 | 17.4827 | 0 | 14.9046 | 18.0555 |
| 1 | 115, | 3 | 17.4827 | 0 | 10.9565 | 11.0145 |
| 2 | 117, | 3 | 17.4827 | 0 | 2.96606 | 2.78031 |
| 3 | and | 3 | 5.28209 | 6.18625 | 6.11446 | 4.84611 |
| 4 | 118 | 3 | 17.4827 | 0 | 0.852189 | 0.798474 |
| 5 | are | 3 | 7.50635 | 7.54802 | 5.26817 | 3.01558 |
| 6 | ... | 0 | 36.5412 | 0 | 10.5468 | 10.1533 |
| 7 | The | 3 | 4.21893 | 5.04894 | 5.05453 | 3.32899 |
| 8 | International | 13 | 12.0924 | 16.0655 | 8.04471 | 7.12144 |
| 9 | Union | 5 | 13.0247 | 15.4497 | 3.93462 | 4.09516 |

If `add_parsing_features` is set to `True`, more word-level features are added to the result (e.g POS, NER, MORPH etc.)

`merge_metrics_with_eye_movements.py` takes and SR interest area reoprt file (where each row is a word for which eye tracking data is collected) and adds word metrics for each word. First, the word-level dataframe, we extract a dataframe where each row is a unique textual item. Second, we apply the `extract_metrics_for_text_df_multiple_hf_models` function to the unique textual items dataframe (and also allow prefix / suffix additions to the input for contextual predictavility calculation) to get word level features for each word (note that this function is generalizable!!!! and is not limited to a specifi setting or dataset). Finally, we merge the word-level dataframe with the metrics dataframe.

For now, the function that contains the whole pipeline is `add_metrics_to_eye_tracking`. For now, this function is dataset specific and is limited to the OneStop eye-movement dataset. Nevertheless, the skeleton of this function can be used as a template for other datasets.

Note, in both functions the columns are hard-coded, so you may need to change them to match your data.

```python
from eyeutils.merge_metrics_with_eye_movements import add_metrics_to_eye_tracking

et_data = pd.read_csv("intermediate_eye_tracking_data.csv")
et_data_enriched = add_metrics_to_eye_tracking(
    eye_tracking_data=et_data,
    surprisal_extraction_model_names=["gpt2", "gpt2-medium"],
    spacy_model_name="en_core_web_sm",
    parsing_mode="re-tokenize",
    add_question_in_prompt=True, # This parameter is dataset specific (OneStop eye-movements)
    model_target_device="cuda:1",
    hf_access_token="",
)
```

## Surprisal

- Surprisal values are extracted from `transformers.AutoModelForCausalLM` (and corresponding `AutoTokenizer`). For a partial list of supported models, see [here](https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#transformers.AutoModelForCausalLM.from_pretrained).
- Words are split by white space, and include adjacent punctuation.
- A surprisal of a word is the sum of the surprisal of the subwords (as split by the tokenizer) that make up the word.

## Frequency

Frequency is extracted via the [wordfreq](https://github.com/rspeer/wordfreq) package.

- Words are split by white space.
- A frequency of a word does not include adjacent punctuation.
- Frequency is given by the figure-skating metric, from multiple sources.
- See the package documentation for more in-depth descriptions.

## Setup

Before running the scripts, you need to download the en_core_web_sm model from spacy:
`python -m spacy download en_core_web_sm`

To install the package -
`pip install git+https://github.com/lacclab/text-metrics.git`.

To make changes - clone to your local station.

Required package versions:

- pandas>=2.1.0
- transformers>=4.40.1
- wordfreq>=3.0.3
- numpy>=1.20.3

Run `python -m doctest -v text_metrics/utils.py` before committing to ensure that the docstrings are up to date.

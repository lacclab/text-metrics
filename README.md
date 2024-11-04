# Text Metrics

Useful utils for extracting frequency, surprisal and other word-level properties from text.
The functions `get_frequency`,`get_surprisal` and`get_metrics` (for both metrics)
can be used to retrieve these values for any piece of text. If `add_parsing_features` is set to `True` for `get_metrics`, more word-level features are added to the result (e.g POS, NER, MORPH etc.)
E.g., running

```python
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType
from text_metrics.surprisal_extractors.extractor_switch import (
    get_surp_extractor,
)
from text_metrics.ling_metrics_funcs import get_metrics

text = """Many of us know we don't get enough sleep, but imagine if there was a simple solution:
getting up later. In a speech at the British Science Festival, Dr. Paul Kelley from Oxford University
said schools should stagger their starting times to work with the natural rhythms of their students.
This would improve exam results and students' health (lack of sleep can cause diabetes, depression,
obesity and other health problems).""".replace("\n", " ").replace("    ", "")

model_name = "gpt2"
surp_extractor = get_surp_extractor(
    extractor_type=SurpExtractorType.CAT_CTX_LEFT, model_name=model_name
)
parsing_model = spacy.load("en_core_web_sm")

metrics = get_metrics(
    target_text=text,
    surp_extractor=surp_extractor,
    parsing_model=parsing_model,
    parsing_mode="re-tokenize",
    add_parsing_features=True,
)
```

Should result in

|     | Word | Length | Wordfreq_Frequency | subtlex_Frequency | gpt2_Surprisal | Word_idx | Token | POS  | TAG | Token_idx | Relationship | Morph                                                   | Entity | Is_Content_Word | Reduced_POS | Head_word_idx | n_Lefts | n_Rights | AbsDistance2Head | Distance2Head | Head_Direction |
| --: | :--- | -----: | -----------------: | ----------------: | -------------: | -------: | :---- | :--- | :-- | --------: | :----------- | :------------------------------------------------------ | :----- | :-------------- | :---------- | ------------: | ------: | -------: | ---------------: | ------------: | :------------- |
|   0 | Many |      4 |            10.2645 |           11.4053 |         7.2296 |        1 | Many  | ADJ  | JJ  |         0 | nsubj        | ['Degree=Pos']                                          |        | True            | ADJ         |             4 |       0 |        1 |                3 |             3 | RIGHT          |
|   1 | of   |      2 |            5.31617 |           6.39588 |        1.76724 |        2 | of    | ADP  | IN  |         1 | prep         | []                                                      |        | False           | FUNC        |             1 |       0 |        1 |                1 |            -1 | LEFT           |
|   2 | us   |      2 |            9.82828 |           9.16726 |        1.56595 |        3 | us    | PRON | PRP |         2 | pobj         | ['Case=Acc', 'Number=Plur', 'Person=1', 'PronType=Prs'] |        | False           | FUNC        |             2 |       0 |        0 |                1 |            -1 | LEFT           |
|   3 | know |      4 |            9.63236 |           7.41279 |        3.44459 |        4 | know  | VERB | VBP |         3 | parataxis    | ['Tense=Pres', 'VerbForm=Fin']                          |        | True            | VERB        |             7 |       1 |        0 |                3 |             3 | RIGHT          |
|   4 | we   |      2 |            8.17085 |           6.75727 |        5.35026 |        5 | we    | PRON | PRP |         4 | nsubj        | ['Case=Nom', 'Number=Plur', 'Person=1', 'PronType=Prs'] |        | False           | FUNC        |             7 |       0 |        0 |                2 |             2 | RIGHT          |

## Working With Text DataFrames

This repo also supports word-level feature extraction for multiple texts and surprisal-extraction models. Note that each text is processed independently. Exmaple usage is shown below:

```python
from text_metrics.merge_metrics_with_eye_movements import extract_metrics_for_text_df_multiple_hf_models
from text_metrics.surprisal_extractors import extractors_constants

text_df = pd.DataFrame(
    {
        "Prefix": ["pre 11", "pre 12", "pre 21", "pre 22"],
        "Target_Text": [
            "Is this the real life?",
            "Is this just fantasy?",
            "Caught in a landslide,",
            "no escape from reality",
        ],
        "Phrase": [1, 2, 1, 2],
        "Line": [1, 1, 2, 2],
    }
)

text_df_w_metrics = extract_metrics_for_text_df_multiple_hf_models(
    text_df=text_df,
    text_col_name="Target_Text",
    text_key_cols=["Phrase", "Line"],
    surprisal_extraction_model_names=[
        "gpt2",
        "EleutherAI/pythia-70m",
        "state-spaces/mamba-130m-hf",
    ],
    surp_extractor_type=extractors_constants.SurpExtractorType.CAT_CTX_LEFT,
    add_parsing_features=False,
    model_target_device="cuda",
    # arguments for the function extract_metrics_for_text_df
    extract_metrics_for_text_df_kwargs={
        "ordered_prefix_col_names": ["Prefix"],
    },
)
```

Result:

|     | index | Word       | Length | Wordfreq_Frequency | subtlex_Frequency | gpt2_Surprisal | Phrase | Line | EleutherAI/pythia-70m_Surprisal | state-spaces/mamba-130m-hf_Surprisal |
| --: | ----: | :--------- | -----: | -----------------: | ----------------: | -------------: | -----: | ---: | ------------------------------: | -----------------------------------: |
|   0 |     0 | Is         |      2 |            6.41735 |           6.75709 |        10.6974 |      1 |    1 |                          9.5653 |                              10.2678 |
|   1 |     1 | this       |      4 |            7.24113 |           6.93294 |        2.35303 |      1 |    1 |                         7.33117 |                              5.59489 |
|   2 |     2 | the        |      3 |            4.21893 |           5.04894 |         2.3539 |      1 |    1 |                         2.90964 |                              1.84557 |
|   3 |     3 | real       |      4 |            11.2949 |           11.1044 |        3.76594 |      1 |    1 |                         5.73141 |                              4.99041 |
|   4 |     4 | life?      |      4 |            10.3317 |           10.2571 |        8.25526 |      1 |    1 |                         7.84282 |                              6.57673 |
|   5 |     0 | Is         |      2 |            6.41735 |           6.75709 |        10.0461 |      2 |    1 |                          9.6152 |                              10.7826 |
|   6 |     1 | this       |      4 |            7.24113 |           6.93294 |        2.61381 |      2 |    1 |                         6.51316 |                              5.26467 |
|   7 |     2 | just       |      4 |            8.53818 |           7.68143 |        6.07112 |      2 |    1 |                         6.84601 |                              4.86802 |
|   8 |     3 | fantasy?   |      7 |            14.8828 |           15.8738 |        8.53002 |      2 |    1 |                         12.7546 |                              8.56943 |
|   9 |     0 | Caught     |      6 |            13.6205 |           13.3412 |        13.8529 |      1 |    2 |                         13.3173 |                              14.6917 |
|  10 |     1 | in         |      2 |            5.74855 |           6.64024 |        2.12986 |      1 |    2 |                         2.23954 |                              2.46692 |
|  11 |     2 | a          |      1 |            5.44851 |           5.57752 |        2.16311 |      1 |    2 |                         1.38347 |                              1.51784 |
|  12 |     3 | landslide, |      9 |            18.3709 |            20.175 |        9.13496 |      1 |    2 |                         13.5699 |                              14.0326 |
|  13 |     0 | no         |      2 |            8.80229 |           7.35099 |        8.02664 |      2 |    2 |                         7.13618 |                              7.29309 |
|  14 |     1 | escape     |      6 |             14.482 |           14.4265 |         9.8561 |      2 |    2 |                         15.3783 |                              13.2179 |
|  15 |     2 | from       |      4 |            7.87155 |            8.9012 |        3.11976 |      2 |    2 |                         3.54212 |                              4.43229 |
|  16 |     3 | reality    |      7 |            13.6536 |           14.9758 |        5.36706 |      2 |    2 |                          8.1663 |                              5.98277 |

# Integration With Eye Movement Data

The function `add_metrics_to_word_level_eye_tracking_report` from `merge_metrics_with_eye_movements.py` processes an SR interest area report file, where each row represents a word for which eye tracking data is collected. The "IA_ID" column indicates the index of the word in the textual item, and the "IA_LABEL" column contains the word itself. The function performs the following steps:

1. Extracts a dataframe where each row is a unique textual item from the word-level dataframe.
2. Applies the `extract_metrics_for_text_df_multiple_hf_models` function to the unique textual items dataframe to obtain word-level features for each word.
3. Merges the word-level dataframe with the metrics dataframe to add word metrics for each word.

```python
from text_metrics.merge_metrics_with_eye_movements import add_metrics_to_word_level_eye_tracking_report

df = pd.read_csv("path/to/interest_area_report.csv")
textual_item_key_cols = [
        "paragraph_id",
        "batch",
        "article_id",
        "level",
        "has_preview",
        "question",
    ]
ia_report_enriched = add_metrics_to_word_level_eye_tracking_report(
    eye_tracking_data=df,
    textual_item_key_cols=textual_item_key_cols,
    surprisal_extraction_model_names=['gpt2', 'EleutherAI/pythia-70m', 'state-spaces/mamba-130m-hf'],
    spacy_model_name="en_core_web_sm",
    parsing_mode="re-tokenize",
    model_target_device='cuda',
    hf_access_token='',
    # CAT_CTX_LEFT: Buggy version from "How to Compute the Probability of a Word" (Pimentel and Meister, 2024). For the correct version, use the SurpExtractorType.PIMENTEL_CTX_LEFT
    surp_extractor_type=extractor_switch.SurpExtractorType.CAT_CTX_LEFT,
)
```

## Surprisal

- The standard surprisal exctractor is `SurpExtractorType.CAT_CTX_LEFT`. Currently, it supports the following models:

```
GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"

Pythia family:
"EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
"EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",

Llama family:
"meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf"

Gemma family (also `*-it` versions):
"google/gemma-2b", "google/gemma-7b", "google/recurrentgemma-2b", "google/recurrentgemma-9b"

Mamba family: (install (pip): mamba-ssm, causal-conv1d>=1.2.0)
"state-spaces/mamba-130m-hf", "state-spaces/mamba-370m-hf",
"state-spaces/mamba-790m-hf", "state-spaces/mamba-1.4b-hf",
"state-spaces/mamba-2.8b-hf"

Mistral family:
"mistralai/Mistral-7B-Instruct-v0.*" where * is the version number
```

- The surprisal exctractors `SurpExtractorType.CAT_CTX_LEFT` and `SurpExtractorType.PIMENTEL_CTX_LEFT` implement the buggy and corrected versions of surprisal calculation as defined in [Pimentel and Meister (2024)](https://arxiv.org/abs/2406.14561), respectively.
- Words are split by white space, and include adjacent punctuation.
- In line with Pimentel and Meister (2024), in case the model implements BOS representation (e.g., GPT-2), we use it and don't add a whitespace prefix.
- A surprisal of a word is the sum of the surprisal of the subwords (as split by the tokenizer) that make up the word.
- Note that `Llama, Gemma, Mistral` require huggingface access token that can be given as a parameter to the function `get_surp_extractor`.

### Prefix Conditioned Surprisal

In many cases, we are interested in the surprisal of a word given a prefix, Where this prefix is concatenated to the left of the target text. By using the `left_context_text` parameter of the `get_metrics` function, along with the surprisal extractors `SurpExtractorType.CAT_CTX_LEFT` or `SurpExtractorType.PIMENTEL_CTX_LEFT`, you can concatenate a text before the target text.
E.g., running

```python
question = "What university is Dr. Kelley from?"

metrics = get_metrics(
    target_text=text,
    surp_extractor=surp_extractor,
    parsing_model=parsing_model,
    parsing_mode="re-tokenize",
    left_context_text=question,
    add_parsing_features=True,
)
```

will affect the surprisal estimates in the column `gpt2_Surprisal` the following way: (higher color intensity -> higher surprisal) 

###### Without Left Context

<img width="781" alt="{C8070022-03F2-41C8-9B78-156007085C7C}" src="https://github.com/user-attachments/assets/8ec4f671-2468-4443-b44a-259e01daf046">


###### With Left Context

<img width="777" alt="{71B87E89-2BCE-43C4-A8E2-A3B6EBFBB028}" src="https://github.com/user-attachments/assets/52320945-ce45-4a41-b158-63c6d8e39616">

As can be seen, the surprisal values for "Dr. Paul Kelly" have been changed drastically due to the left context.

Note that regardless of the edition of left context (all extractors support the "no left context" setting), `SurpExtractorType.CAT_CTX_LEFT` and `SurpExtractorType.PIMENTEL_CTX_LEFT` implement the buggy and corrected (respectively) versions of surprisal calculation as defined in [Pimentel and Meister (2024)](https://arxiv.org/abs/2406.14561).

In addition to the ability to prompt the model with textual context as a prefix, the extractors `SurpExtractorType.SOFT_CAT_WHOLE_CTX_LEFT` and `SurpExtractorType.SOFT_CAT_SENTENCES` allow for prompting the transformer at the initial embedding level wth two aggregation levels of the prefix. In `SurpExtractorType.SOFT_CAT_WHOLE_CTX_LEFT` the whole prefix is aggregated into a single output vector before being concatenated to the target text. In `SurpExtractorType.SOFT_CAT_SENTENCES` the prefix is split into sentences, each sentence is aggregated into a single output vector, and the output vectors are concatenated to the target text. This allows for a more nuanced control over the prefixing of the model, and can be used to investigate the effect of different types of context on the surprisal of a word.

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

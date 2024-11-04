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

will affect the surprisal estimates in the column `gpt2_Surprisal` the following way:

###### Without Left Context

<h4></h4>
<span title="7.23" class="barcode" style="color: white; background-color: #2777b8; font-size:1em">&nbspMany&nbsp</span>
<span title="1.77" class="barcode" style="color: black; background-color: #d4e4f4; font-size:1em">&nbspof&nbsp</span>
<span title="1.57" class="barcode" style="color: black; background-color: #d8e7f5; font-size:1em">&nbspus&nbsp</span>
<span title="3.44" class="barcode" style="color: black; background-color: #a8cee4; font-size:1em">&nbspknow&nbsp</span>
<span title="5.35" class="barcode" style="color: black; background-color: #60a7d2; font-size:1em">&nbspwe&nbsp</span>
<span title="3.43" class="barcode" style="color: black; background-color: #a9cfe5; font-size:1em">&nbspdon't&nbsp</span>
<span title="3.34" class="barcode" style="color: black; background-color: #abd0e6; font-size:1em">&nbspget&nbsp</span>
<span title="2.58" class="barcode" style="color: black; background-color: #c4daee; font-size:1em">&nbspenough&nbsp</span>
<span title="3.18" class="barcode" style="color: black; background-color: #b0d2e7; font-size:1em">&nbspsleep,&nbsp</span>
<span title="1.24" class="barcode" style="color: black; background-color: #dfebf7; font-size:1em">&nbspbut&nbsp</span>
<span title="7.64" class="barcode" style="color: white; background-color: #1e6db2; font-size:1em">&nbspimagine&nbsp</span>
<span title="1.32" class="barcode" style="color: black; background-color: #ddeaf7; font-size:1em">&nbspif&nbsp</span>
<span title="3.49" class="barcode" style="color: black; background-color: #a6cee4; font-size:1em">&nbspthere&nbsp</span>
<span title="1.07" class="barcode" style="color: black; background-color: #e2edf8; font-size:1em">&nbspwas&nbsp</span>
<span title="0.72" class="barcode" style="color: black; background-color: #e9f2fa; font-size:1em">&nbspa&nbsp</span>
<span title="5.21" class="barcode" style="color: black; background-color: #64a9d3; font-size:1em">&nbspsimple&nbsp</span>
<span title="4.42" class="barcode" style="color: black; background-color: #82bbdb; font-size:1em">&nbspsolution:&nbsp</span>
<span title="7.02" class="barcode" style="color: white; background-color: #2e7ebc; font-size:1em">&nbspgetting&nbsp</span>
<span title="2.56" class="barcode" style="color: black; background-color: #c4daee; font-size:1em">&nbspup&nbsp</span>
<span title="7.29" class="barcode" style="color: white; background-color: #2676b8; font-size:1em">&nbsplater.&nbsp</span>
<span title="4.35" class="barcode" style="color: black; background-color: #85bcdc; font-size:1em">&nbspIn&nbsp</span>
<span title="2.52" class="barcode" style="color: black; background-color: #c6dbef; font-size:1em">&nbspa&nbsp</span>
<span title="8.77" class="barcode" style="color: white; background-color: #08509b; font-size:1em">&nbspspeech&nbsp</span>
<span title="1.73" class="barcode" style="color: black; background-color: #d5e5f4; font-size:1em">&nbspat&nbsp</span>
<span title="0.79" class="barcode" style="color: black; background-color: #e7f1fa; font-size:1em">&nbspthe&nbsp</span>
<span title="5.76" class="barcode" style="color: black; background-color: #529dcc; font-size:1em">&nbspBritish&nbsp</span>
<span title="4.41" class="barcode" style="color: black; background-color: #84bcdb; font-size:1em">&nbspScience&nbsp</span>
<span title="2.24" class="barcode" style="color: black; background-color: #cbdef1; font-size:1em">&nbspFestival,&nbsp</span>
<span title="3.66" class="barcode" style="color: black; background-color: #a1cbe2; font-size:1em">&nbspDr.&nbsp</span>
<span title="4.31" class="barcode" style="color: black; background-color: #87bddc; font-size:1em">&nbspPaul&nbsp</span>
<span title="7.85" class="barcode" style="color: white; background-color: #1967ad; font-size:1em">&nbspKelley&nbsp</span>
<span title="4.6" class="barcode" style="color: black; background-color: #7cb7da; font-size:1em">&nbspfrom&nbsp</span>
<span title="4.67" class="barcode" style="color: black; background-color: #79b5d9; font-size:1em">&nbspOxford&nbsp</span>
<span title="0.12" class="barcode" style="color: black; background-color: #f5f9fe; font-size:1em">&nbspUniversity&nbsp</span>
<span title="2.4" class="barcode" style="color: black; background-color: #c8dcf0; font-size:1em">&nbspsaid&nbsp</span>
<span title="11.12" class="barcode" style="color: white; background-color: #08306b; font-size:1em">&nbspschools&nbsp</span>
<span title="1.03" class="barcode" style="color: black; background-color: #e3eef8; font-size:1em">&nbspshould&nbsp</span>
<span title="10.35" class="barcode" style="color: white; background-color: #08306b; font-size:1em">&nbspstagger&nbsp</span>
<span title="1.58" class="barcode" style="color: black; background-color: #d8e7f5; font-size:1em">&nbsptheir&nbsp</span>
<span title="8.35" class="barcode" style="color: white; background-color: #105ba4; font-size:1em">&nbspstarting&nbsp</span>
<span title="1.3" class="barcode" style="color: black; background-color: #ddeaf7; font-size:1em">&nbsptimes&nbsp</span>
<span title="1.82" class="barcode" style="color: black; background-color: #d3e4f3; font-size:1em">&nbspto&nbsp</span>
<span title="6.7" class="barcode" style="color: black; background-color: #3686c0; font-size:1em">&nbspwork&nbsp</span>
<span title="2.5" class="barcode" style="color: black; background-color: #c7dbef; font-size:1em">&nbspwith&nbsp</span>
<span title="2.59" class="barcode" style="color: black; background-color: #c3daee; font-size:1em">&nbspthe&nbsp</span>
<span title="7.68" class="barcode" style="color: white; background-color: #1d6cb1; font-size:1em">&nbspnatural&nbsp</span>
<span title="2.88" class="barcode" style="color: black; background-color: #bad6eb; font-size:1em">&nbsprhythms&nbsp</span>
<span title="0.3" class="barcode" style="color: black; background-color: #f2f7fd; font-size:1em">&nbspof&nbsp</span>
<span title="2.59" class="barcode" style="color: black; background-color: #c3daee; font-size:1em">&nbsptheir&nbsp</span>
<span title="1.1" class="barcode" style="color: black; background-color: #e1edf8; font-size:1em">&nbspstudents.&nbsp</span>
<span title="4.64" class="barcode" style="color: black; background-color: #7ab6d9; font-size:1em">&nbspThis&nbsp</span>
<span title="2.11" class="barcode" style="color: black; background-color: #cee0f2; font-size:1em">&nbspwould&nbsp</span>
<span title="4.59" class="barcode" style="color: black; background-color: #7cb7da; font-size:1em">&nbspimprove&nbsp</span>
<span title="8.75" class="barcode" style="color: white; background-color: #08509b; font-size:1em">&nbspexam&nbsp</span>
<span title="1.52" class="barcode" style="color: black; background-color: #d9e8f5; font-size:1em">&nbspresults&nbsp</span>
<span title="1.22" class="barcode" style="color: black; background-color: #dfebf7; font-size:1em">&nbspand&nbsp</span>
<span title="4.87" class="barcode" style="color: black; background-color: #71b1d7; font-size:1em">&nbspstudents'&nbsp</span>
<span title="4.73" class="barcode" style="color: black; background-color: #75b4d8; font-size:1em">&nbsphealth&nbsp</span>
<span title="15.48" class="barcode" style="color: white; background-color: #08306b; font-size:1em">&nbsp(lack&nbsp</span>
<span title="0.02" class="barcode" style="color: black; background-color: #f7fbff; font-size:1em">&nbspof&nbsp</span>
<span title="0.21" class="barcode" style="color: black; background-color: #f3f8fe; font-size:1em">&nbspsleep&nbsp</span>
<span title="3.14" class="barcode" style="color: black; background-color: #b2d2e8; font-size:1em">&nbspcan&nbsp</span>
<span title="2.25" class="barcode" style="color: black; background-color: #cbdef1; font-size:1em">&nbspcause&nbsp</span>
<span title="6.96" class="barcode" style="color: black; background-color: #2f7fbc; font-size:1em">&nbspdiabetes,&nbsp</span>
<span title="4.09" class="barcode" style="color: black; background-color: #91c3de; font-size:1em">&nbspdepression,&nbsp</span>
<span title="3.98" class="barcode" style="color: black; background-color: #95c5df; font-size:1em">&nbspobesity&nbsp</span>
<span title="1.27" class="barcode" style="color: black; background-color: #deebf7; font-size:1em">&nbspand&nbsp</span>
<span title="1.74" class="barcode" style="color: black; background-color: #d5e5f4; font-size:1em">&nbspother&nbsp</span>
<span title="1.31" class="barcode" style="color: black; background-color: #ddeaf7; font-size:1em">&nbsphealth&nbsp</span>
<span title="0.98" class="barcode" style="color: black; background-color: #e3eef9; font-size:1em">&nbspproblems).&nbsp</span>
<br><br>

###### With Left Context

<h4></h4>
<span title="8.02" class="barcode" style="color: white; background-color: #1663aa; font-size:1em">&nbspMany&nbsp</span>
<span title="1.75" class="barcode" style="color: black; background-color: #d5e5f4; font-size:1em">&nbspof&nbsp</span>
<span title="1.43" class="barcode" style="color: black; background-color: #dbe9f6; font-size:1em">&nbspus&nbsp</span>
<span title="2.3" class="barcode" style="color: black; background-color: #cadef0; font-size:1em">&nbspknow&nbsp</span>
<span title="7.75" class="barcode" style="color: white; background-color: #1c6ab0; font-size:1em">&nbspwe&nbsp</span>
<span title="3.48" class="barcode" style="color: black; background-color: #a8cee4; font-size:1em">&nbspdon't&nbsp</span>
<span title="2.67" class="barcode" style="color: black; background-color: #c1d9ed; font-size:1em">&nbspget&nbsp</span>
<span title="3.06" class="barcode" style="color: black; background-color: #b4d3e9; font-size:1em">&nbspenough&nbsp</span>
<span title="5.94" class="barcode" style="color: black; background-color: #4b98ca; font-size:1em">&nbspsleep,&nbsp</span>
<span title="1.22" class="barcode" style="color: black; background-color: #dfebf7; font-size:1em">&nbspbut&nbsp</span>
<span title="8.48" class="barcode" style="color: white; background-color: #0d57a1; font-size:1em">&nbspimagine&nbsp</span>
<span title="1.09" class="barcode" style="color: black; background-color: #e2edf8; font-size:1em">&nbspif&nbsp</span>
<span title="3.39" class="barcode" style="color: black; background-color: #aacfe5; font-size:1em">&nbspthere&nbsp</span>
<span title="1.18" class="barcode" style="color: black; background-color: #dfecf7; font-size:1em">&nbspwas&nbsp</span>
<span title="0.7" class="barcode" style="color: black; background-color: #eaf2fb; font-size:1em">&nbspa&nbsp</span>
<span title="5.91" class="barcode" style="color: black; background-color: #4d99ca; font-size:1em">&nbspsimple&nbsp</span>
<span title="4.67" class="barcode" style="color: black; background-color: #79b5d9; font-size:1em">&nbspsolution:&nbsp</span>
<span title="7.59" class="barcode" style="color: white; background-color: #1f6eb3; font-size:1em">&nbspgetting&nbsp</span>
<span title="2.55" class="barcode" style="color: black; background-color: #c4daee; font-size:1em">&nbspup&nbsp</span>
<span title="7.25" class="barcode" style="color: white; background-color: #2777b8; font-size:1em">&nbsplater.&nbsp</span>
<span title="4.26" class="barcode" style="color: black; background-color: #89bedc; font-size:1em">&nbspIn&nbsp</span>
<span title="2.62" class="barcode" style="color: black; background-color: #c2d9ee; font-size:1em">&nbspa&nbsp</span>
<span title="7.91" class="barcode" style="color: white; background-color: #1966ad; font-size:1em">&nbspspeech&nbsp</span>
<span title="1.35" class="barcode" style="color: black; background-color: #dceaf6; font-size:1em">&nbspat&nbsp</span>
<span title="0.92" class="barcode" style="color: black; background-color: #e5eff9; font-size:1em">&nbspthe&nbsp</span>
<span title="6.09" class="barcode" style="color: black; background-color: #4896c8; font-size:1em">&nbspBritish&nbsp</span>
<span title="4.89" class="barcode" style="color: black; background-color: #6fb0d7; font-size:1em">&nbspScience&nbsp</span>
<span title="2.86" class="barcode" style="color: black; background-color: #bad6eb; font-size:1em">&nbspFestival,&nbsp</span>
<span title="1.04" class="barcode" style="color: black; background-color: #e3eef8; font-size:1em">&nbspDr.&nbsp</span>
<span title="2.44" class="barcode" style="color: black; background-color: #c7dcef; font-size:1em">&nbspPaul&nbsp</span>
<span title="0.19" class="barcode" style="color: black; background-color: #f4f9fe; font-size:1em">&nbspKelley&nbsp</span>
<span title="6.66" class="barcode" style="color: black; background-color: #3787c0; font-size:1em">&nbspfrom&nbsp</span>
<span title="3.82" class="barcode" style="color: black; background-color: #9cc9e1; font-size:1em">&nbspOxford&nbsp</span>
<span title="0.4" class="barcode" style="color: black; background-color: #eff6fc; font-size:1em">&nbspUniversity&nbsp</span>
<span title="2.24" class="barcode" style="color: black; background-color: #cbdef1; font-size:1em">&nbspsaid&nbsp</span>
<span title="9.53" class="barcode" style="color: white; background-color: #083c7d; font-size:1em">&nbspschools&nbsp</span>
<span title="1.14" class="barcode" style="color: black; background-color: #e0ecf8; font-size:1em">&nbspshould&nbsp</span>
<span title="10.3" class="barcode" style="color: white; background-color: #08306b; font-size:1em">&nbspstagger&nbsp</span>
<span title="1.37" class="barcode" style="color: black; background-color: #dce9f6; font-size:1em">&nbsptheir&nbsp</span>
<span title="8.18" class="barcode" style="color: white; background-color: #135fa7; font-size:1em">&nbspstarting&nbsp</span>
<span title="1.34" class="barcode" style="color: black; background-color: #dceaf6; font-size:1em">&nbsptimes&nbsp</span>
<span title="1.9" class="barcode" style="color: black; background-color: #d2e3f3; font-size:1em">&nbspto&nbsp</span>
<span title="6.48" class="barcode" style="color: black; background-color: #3c8cc3; font-size:1em">&nbspwork&nbsp</span>
<span title="2.45" class="barcode" style="color: black; background-color: #c7dcef; font-size:1em">&nbspwith&nbsp</span>
<span title="2.47" class="barcode" style="color: black; background-color: #c7dbef; font-size:1em">&nbspthe&nbsp</span>
<span title="7.59" class="barcode" style="color: white; background-color: #1f6eb3; font-size:1em">&nbspnatural&nbsp</span>
<span title="2.65" class="barcode" style="color: black; background-color: #c2d9ee; font-size:1em">&nbsprhythms&nbsp</span>
<span title="0.25" class="barcode" style="color: black; background-color: #f2f8fd; font-size:1em">&nbspof&nbsp</span>
<span title="1.96" class="barcode" style="color: black; background-color: #d0e2f2; font-size:1em">&nbsptheir&nbsp</span>
<span title="1.08" class="barcode" style="color: black; background-color: #e2edf8; font-size:1em">&nbspstudents.&nbsp</span>
<span title="4.32" class="barcode" style="color: black; background-color: #87bddc; font-size:1em">&nbspThis&nbsp</span>
<span title="2.44" class="barcode" style="color: black; background-color: #c7dcef; font-size:1em">&nbspwould&nbsp</span>
<span title="4.64" class="barcode" style="color: black; background-color: #7ab6d9; font-size:1em">&nbspimprove&nbsp</span>
<span title="8.55" class="barcode" style="color: white; background-color: #0c56a0; font-size:1em">&nbspexam&nbsp</span>
<span title="1.51" class="barcode" style="color: black; background-color: #d9e8f5; font-size:1em">&nbspresults&nbsp</span>
<span title="1.25" class="barcode" style="color: black; background-color: #dfebf7; font-size:1em">&nbspand&nbsp</span>
<span title="4.7" class="barcode" style="color: black; background-color: #77b5d9; font-size:1em">&nbspstudents'&nbsp</span>
<span title="4.98" class="barcode" style="color: black; background-color: #6caed6; font-size:1em">&nbsphealth&nbsp</span>
<span title="14.86" class="barcode" style="color: white; background-color: #08306b; font-size:1em">&nbsp(lack&nbsp</span>
<span title="0.03" class="barcode" style="color: black; background-color: #f7fbff; font-size:1em">&nbspof&nbsp</span>
<span title="0.21" class="barcode" style="color: black; background-color: #f3f8fe; font-size:1em">&nbspsleep&nbsp</span>
<span title="3.32" class="barcode" style="color: black; background-color: #add0e6; font-size:1em">&nbspcan&nbsp</span>
<span title="2.24" class="barcode" style="color: black; background-color: #cbdef1; font-size:1em">&nbspcause&nbsp</span>
<span title="7.02" class="barcode" style="color: white; background-color: #2e7ebc; font-size:1em">&nbspdiabetes,&nbsp</span>
<span title="4.1" class="barcode" style="color: black; background-color: #91c3de; font-size:1em">&nbspdepression,&nbsp</span>
<span title="4.06" class="barcode" style="color: black; background-color: #92c4de; font-size:1em">&nbspobesity&nbsp</span>
<span title="1.32" class="barcode" style="color: black; background-color: #ddeaf7; font-size:1em">&nbspand&nbsp</span>
<span title="1.72" class="barcode" style="color: black; background-color: #d6e5f4; font-size:1em">&nbspother&nbsp</span>
<span title="1.31" class="barcode" style="color: black; background-color: #ddeaf7; font-size:1em">&nbsphealth&nbsp</span>
<span title="1.04" class="barcode" style="color: black; background-color: #e3eef8; font-size:1em">&nbspproblems).&nbsp</span>
<br><br>

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

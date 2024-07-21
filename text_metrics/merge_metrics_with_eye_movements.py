import gc
from functools import partial
from typing import Union, Tuple, Dict, List, Literal
import pandas as pd
import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoXTokenizerFast,
    GPTNeoXForCausalLM,
)
import spacy
from spacy.language import Language
import torch
from text_metrics.utils import get_metrics, init_tok_n_model


def create_text_input(
    row: pd.Series,
    text_col_name: str,
    prefix_col_names: List[str],
    suffix_col_names: List[str],
) -> Tuple[
    str, Tuple[int, int], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]
]:
    """This function creates a single string from the text columns in the row

    Args:
        row (pd.Series): A row from a dataframe that contains text columns
        text_col_name (str): The name of the column in row that contains the main text
        prefix_col_names (List[str]): Column names in row that contain prefixes to be added.
            Text will be added according to the order of the columns in the list
        suffix_col_names (List[str]): Column names in row that contain suffixes to be added.
            Text will be added according to the order of the columns in the list

    Returns:
        Tuple[ str, Tuple[int, int], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]] ]:
            A tuple containing:
            - The concatenated text from the columns in row
            - The word indices of the main text in the concatenated text
            - A dictionary with the word indices ranges of the prefixes in the concatenated text.
            - A dictionary with the word indices ranges of the suffixes in the concatenated text.
    """
    text_input = ""
    curr_w_index = 0

    # add prefixes and document their word indices
    prefixes_word_indices_ranges = {}
    for prefix_col in prefix_col_names:
        curr_prefix = getattr(row, prefix_col)
        if (curr_prefix is not None) and (len(curr_prefix) > 0):
            text_input += curr_prefix + " "
        curr_prefix_len = len(curr_prefix.split())
        next_w_index = curr_w_index + curr_prefix_len
        prefixes_word_indices_ranges[prefix_col] = (
            curr_w_index,
            next_w_index - 1,
        )
        curr_w_index = next_w_index

    row_main_text = getattr(row, text_col_name).strip()
    row_main_text_len = len(row_main_text.split())
    text_input += row_main_text
    main_text_word_indices = (
        curr_w_index,
        curr_w_index + row_main_text_len - 1,
    )
    curr_w_index += row_main_text_len - 1

    # add suffixes and document their word indices
    suffixes_word_indices_ranges = {}
    if len(suffix_col_names) > 0:
        text_input += " "
        for i, suffix_col in enumerate(suffix_col_names):
            curr_suffix_text = getattr(row, suffix_col)
            text_input += (
                curr_suffix_text + " "
                if i < len(suffix_col_names) - 1
                else curr_suffix_text
            )
            curr_suffix_len = len(curr_suffix_text.split())
            suffixes_word_indices_ranges[suffix_col] = (
                curr_w_index,
                curr_w_index + curr_suffix_len - 1,
            )
            curr_w_index += curr_suffix_len

    return (
        text_input,
        main_text_word_indices,
        prefixes_word_indices_ranges,
        suffixes_word_indices_ranges,
    )


def filter_prefix_suffix_metrics(
    merged_df: pd.DataFrame,
    prefix_col_names: List[str],
    keep_prefix_metrics: bool | List[str],
    suffix_col_names: List[str],
    keep_suffix_metrics: bool | List[str],
    prefixes_word_indices_ranges: dict[str, Tuple[int, int]],
    suffixes_word_indices_ranges: dict[str, Tuple[int, int]],
) -> pd.DataFrame:
    """This function filters the prefix and suffix metrics from the merged_df

    Args:
        merged_df (pd.DataFrame): In this dataframe, every row is a word with it's index
            and extracted properties (length, frequency, surprisal etc.)
        prefix_col_names (List[str]): Column names passed to extract_metrics_for_text_df
            that contain prefixes that were added to the text (appear as rows in merged_df)
        keep_prefix_metrics (bool | List[str]): If False, all prefix words (rows) will be
            removed after metric extraction.
            If True, all prefix metrics will be kept. If a list of strings, only the metrics
            in the list will be kept.
        suffix_col_names (List[str]): Column names passed to extract_metrics_for_text_df
            that contain suffixes that were added to the text (appear as rows in merged_df)
        keep_suffix_metrics (bool | List[str]): If False, all suffix words (rows) will be
            removed after metric extraction.
        prefixes_word_indices_ranges (dict[str, Tuple[int, int]]): A dictionary specifying
            for each prefix column, the range of word indices in merged_df that correspond
        suffixes_word_indices_ranges (dict[str, Tuple[int, int]]): A dictionary specifying
            for each suffix column, the range of word indices in merged_df that correspond

    Returns:
        pd.DataFrame: merged_df after the prefix and suffix words (rows) were filtered
    """
    prefixes_suffixes_ranges_lst = [
        (prefix_col_names, keep_prefix_metrics, prefixes_word_indices_ranges),
        (suffix_col_names, keep_suffix_metrics, suffixes_word_indices_ranges),
    ]

    for (
        full_col_names,
        cols_to_keep,
        word_indices_ranges,
    ) in prefixes_suffixes_ranges_lst:
        if cols_to_keep is True:  # keep all columns
            continue
        if cols_to_keep is False:  # remove all columns
            cols_to_remove = full_col_names
        else:  # keep only the specified columns
            assert isinstance(cols_to_keep, list) and all(
                [isinstance(col, str) for col in cols_to_keep]
            )
            cols_to_remove = [col for col in full_col_names if col not in cols_to_keep]

        merged_df = merged_df.drop(
            merged_df.index[
                sum(
                    [
                        list(
                            range(
                                word_indices_ranges[col_to_remove][0],
                                word_indices_ranges[col_to_remove][1] + 1,
                            )
                        )
                        for col_to_remove in cols_to_remove
                    ],
                    [],
                )
            ]
        )
    return merged_df


def extract_metrics_for_text_df(
    text_df: pd.DataFrame,
    text_col_name: str,
    text_key_cols: List[str],
    model: Union[AutoModelForCausalLM, GPTNeoXForCausalLM],
    model_name: str,
    tokenizer: Union[AutoTokenizer, GPTNeoXTokenizerFast],
    ordered_prefix_col_names: List[str] = [],
    keep_prefix_metrics: bool | List[str] = False,
    ordered_suffix_col_names: List[str] = [],
    keep_suffix_metrics: bool | List[str] = False,
    rebase_index_in_main_text: bool = False,
    get_metrics_kwargs: dict | None = None,
) -> pd.DataFrame:
    """This function extracts word level characteristics
    (Length, frequency, surprisal) from a text_df
    This function allows adding prefixes and suffixes to
    the text and extract metrics for them as well (or omit them)

    Args:
        text_df (pd.DataFrame): A dataframe where each row has text identifying columns,
            main text column (string) and possibly prefix and suffix columns (strings)
        text_col_name (str): The name of the column in text_df that contains the main text
        text_key_cols (List[str]): The columns in text_df that identify the text
        model (Union[AutoModelForCausalLM, GPTNeoXForCausalLM]): A language model from which
            surprisal values will be extracted
        model_name (str): The name of the model
        tokenizer (Union[AutoTokenizer, GPTNeoXTokenizerFast]): The tokenizer for the model
        ordered_prefix_col_names (List[str], optional): A list of column names in text_df
            that contain prefixes. The order in which they appear in the list is the order
            in which they will be added to the text. Defaults to [].
        keep_prefix_metrics (bool | List[str], optional): If False, all prefix metrics will be
            removed after metric extraction.
            If True, all prefix metrics will be kept. If a list of strings, only the metrics
            in the list will be kept. Defaults to False.
        ordered_suffix_col_names (List[str], optional): A list of column names in text_df
            that contain suffixes. The order in which they appear in the list is the order
            in which they will be added to the text. Defaults to [].
        keep_suffix_metrics (bool | List[str], optional): If False, all prefix metrics will be
            removed after metric extraction.
            If True, all prefix metrics will be kept. If a list of strings, only the metrics
            in the list will be kept. Defaults to False.
        rebase_index_in_main_text (bool, optional): If True, regardless of added prefixes and
            suffixes, the 'index' column for each text will start from 0 from the beginning
            of the main text. I.e, the 'index' column for prefixes will be negative and for
            suffixes will be
            bigger than the main text length (in words). Defaults to False.
        get_metrics_kwargs (dict | None, optional): A dict of additional keyword arguments for
            the get_metrics function. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with the extracted metrics, now instead of each row being a
            whole text, each row is a word in the text. The columns are the main text
            identifier, word_index, word and extracted metrics.
    """
    get_metrics_kwargs = {} if get_metrics_kwargs is None else get_metrics_kwargs.copy()
    metric_dfs = []
    for row in tqdm.tqdm(
        text_df.reset_index().itertuples(),
        total=len(text_df),
        desc="Extracting metrics",
    ):
        if len(ordered_prefix_col_names) > 0 or len(ordered_suffix_col_names) > 0:
            (
                text_input,
                main_text_word_indices,
                prefixes_word_indices_ranges,
                suffixes_word_indices_ranges,
            ) = create_text_input(
                row, text_col_name, ordered_prefix_col_names, ordered_suffix_col_names
            )
        else:
            text_input = getattr(row, text_col_name).strip()
            main_text_word_indices = (0, len(text_input.split()) - 1)

        # add here new metrics
        merged_df = get_metrics(
            text=text_input.strip(),
            models=[model],
            tokenizers=[tokenizer],
            model_names=[model_name],
            **get_metrics_kwargs,
        )
        merged_df.reset_index(inplace=True)

        # in merged df, remove the prefixes and suffixes that are
        # not in the keep_prefix_metrics and keep_suffix_metrics
        if len(ordered_prefix_col_names) > 0 or len(ordered_suffix_col_names) > 0:
            merged_df = filter_prefix_suffix_metrics(
                merged_df,
                ordered_prefix_col_names,
                keep_prefix_metrics,
                ordered_suffix_col_names,
                keep_suffix_metrics,
                prefixes_word_indices_ranges,
                suffixes_word_indices_ranges,
            )

        if rebase_index_in_main_text and len(ordered_prefix_col_names) > 0:
            merged_df["index"] = merged_df["index"] - main_text_word_indices[0]

        merged_df[text_key_cols] = [getattr(row, key_col) for key_col in text_key_cols]

        metric_dfs.append(merged_df)

    return pd.concat(metric_dfs, axis=0)


def extract_metrics_for_text_df_multiple_hf_models(
    text_df: pd.DataFrame,
    text_col_name: str,
    text_key_cols: List[str],
    surprisal_extraction_model_names: List[str],
    add_parsing_features: bool = True,
    parsing_mode: (
        Literal["keep-first", "keep-all", "re-tokenize"] | None
    ) = "re-tokenize",
    spacy_model: Language | None = spacy.load("en_core_web_sm"),
    model_target_device: str = "cpu",
    hf_access_token: str = None,
    extract_metrics_for_text_df_kwargs: dict | None = None,
) -> pd.DataFrame:
    """This function extracts word level characteristics while extracting surprisal
        estimates from multiple huggingface models

    Args:
        text_df (pd.DataFrame): A dataframe where each row has text identifying columns,
            main text column (string) and possibly prefix and suffix columns (strings)
        text_col_name (str): The name of the column in text_df that contains the main text
        text_key_cols (List[str]): The columns in text_df that identify the text
        surprisal_extraction_model_names (List[str]): The name of the models to extract surprisal
            values from
        add_parsing_features (bool, optional): If True, parsing features will be added to the
            extracted metrics. Defaults to True.
        parsing_mode (Literal[&quot;keep): Type of parsing to use. one of
            ['keep-first','keep-all','re-tokenize']. Defaults to "re-tokenize".
        spacy_model (Language): The spacy model to use for parsing the text.
            Defaults to spacy.load("en_core_web_sm").
        model_target_device (str, optional): The device to move the model to. Defaults to "cpu".
        hf_access_token (str, optional): A huggingface access token. Defaults to None.
        extract_metrics_for_text_df_kwargs (dict | None, optional): A dict of additional keyword
            arguments for the extract_metrics_for_text_df function. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with the extracted metrics, now instead of each row being a
            whole text, each row is a word in the text. The columns are the main text
            identifier, word_index, word and extracted metrics.
    """
    assert not (
        add_parsing_features is True and (parsing_mode is None or spacy_model is None)
    ), "If add_parsing_features is True, both parsing_mode and spacy_model must be provided"

    if extract_metrics_for_text_df_kwargs is None:
        extract_metrics_for_text_df_kwargs = {}

    # Check if get_metrics_kwargs is in extract_metrics_for_text_df_kwargs
    get_metrics_kwargs = {}
    if "get_metrics_kwargs" in extract_metrics_for_text_df_kwargs:
        get_metrics_kwargs = extract_metrics_for_text_df_kwargs["get_metrics_kwargs"]
        del extract_metrics_for_text_df_kwargs["get_metrics_kwargs"]
    get_metrics_kwargs["parsing_model"] = spacy_model
    get_metrics_kwargs["parsing_mode"] = parsing_mode

    metric_df = None
    for i, model_name in enumerate(surprisal_extraction_model_names):
        # log the model name
        if i == 0:
            print("Extracting Frequency, Length")
        print(f"Extracting surprisal using model: {model_name}")

        tokenizer, model = init_tok_n_model(
            model_name=model_name,
            device=model_target_device,
            hf_access_token=hf_access_token,
        )

        get_metrics_kwargs["add_parsing_features"] = (
            True if metric_df is None and add_parsing_features else False
        )
        metric_dfs = extract_metrics_for_text_df(
            text_df=text_df,
            text_col_name=text_col_name,  # this is after turning all the words into a single string
            text_key_cols=text_key_cols,
            model=model,
            model_name=surprisal_extraction_model_names[i],
            tokenizer=tokenizer,
            get_metrics_kwargs=get_metrics_kwargs,
            **extract_metrics_for_text_df_kwargs,
        )

        """Here we incrementally add the metrics from all models
        In order to avoid duplicates, we merge only on the columns that were added in
        the current iteration (surprisal estimates from the current model)"""
        if metric_df is None:
            metric_df = metric_dfs.copy()
        else:
            concatenated_metric_dfs = metric_dfs.copy()
            cols_to_merge = concatenated_metric_dfs.columns.difference(
                metric_df.columns
            ).tolist()
            cols_to_merge += text_key_cols + ["index"]

            metric_df = metric_df.merge(
                concatenated_metric_dfs[cols_to_merge],
                how="left",
                on=text_key_cols + ["index"],
                validate="one_to_one",
            )
        # move the model back to the cpu and delete it to free up space
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return metric_df


def add_metrics_to_eye_tracking(
    eye_tracking_data: pd.DataFrame,
    surprisal_extraction_model_names: List[str],
    spacy_model_name: str,
    parsing_mode: Literal["keep-first", "keep-all", "re-tokenize"],
    add_question_in_prompt: bool = False,
    model_target_device: str = "cpu",
    hf_access_token: str = None,
) -> pd.DataFrame:
    """This function adds metrics to the eye_tracking_data dataframe

    Args:
        eye_tracking_data (pd.DataFrame): A dataframe with eye tracking data
        surprisal_extraction_model_names (List[str]): The name of the models to extract surprisal
            values from (huggingface models)
        spacy_model_name (str): The name of the spacy model to use for parsing the text
        parsing_mode (str): Type of parsing to use. one of
            ['keep-first','keep-all','re-tokenize']
        add_question_in_prompt (bool, optional): If True, the question will be added to the prompt
            for surprisal extraction, but the question itself will not be included in the metrics.
            (no eye tracking data for the question). Defaults to False.
        model_target_device (str, optional): The device to move the model to. Defaults to "cpu".
        hf_access_token (str, optional): A huggingface access token. Defaults to None.

    Returns:
        pd.DataFrame: The eye_tracking_data dataframe with the added word-level metrics
    """

    # Remove duplicates
    without_duplicates = eye_tracking_data[
        [
            "paragraph_id",
            "article_title",
            "level",
            "IA_ID",
            "has_preview",
            "question",
            "IA_LABEL",
        ]
    ].drop_duplicates()

    # Group by paragraph_id, article_title, level and join all IA_LABELs (words)
    text_from_et = without_duplicates.groupby(
        ["paragraph_id", "article_title", "level", "has_preview", "question"]
    )["IA_LABEL"].apply(list)

    text_from_et = text_from_et.apply(lambda text: " ".join(text))

    spacy_model = spacy.load(spacy_model_name)
    text_key_cols = [
        "paragraph_id",
        "article_title",
        "level",
        "has_preview",
        "question",
    ]

    extract_metrics_partial = partial(
        extract_metrics_for_text_df_multiple_hf_models,
        text_col_name="IA_LABEL",
        text_key_cols=text_key_cols,
        surprisal_extraction_model_names=surprisal_extraction_model_names,
        parsing_mode=parsing_mode,
        spacy_model=spacy_model,
        model_target_device=model_target_device,
        hf_access_token=hf_access_token,
    )
    if add_question_in_prompt:
        print("Extracting metrics: Hunting")
        hunting_metric_df = extract_metrics_partial(
            text_df=text_from_et[
                text_from_et.index.get_level_values("has_preview") == "Hunting"
            ],
            extract_metrics_for_text_df_kwargs=dict(
                ordered_prefix_col_names=["question"],
                keep_prefix_metrics=False,
                rebase_index_in_main_text=True,
            ),
        )
        print("Extracting metrics: Gathering")
        gathering_metric_df = extract_metrics_partial(
            text_df=text_from_et[
                text_from_et.index.get_level_values("has_preview") == "Gathering"
            ],
            extract_metrics_for_text_df_kwargs=dict(
                ordered_prefix_col_names=[],
                keep_prefix_metrics=False,
                rebase_index_in_main_text=True,
            ),
        )

        metric_df = pd.concat([hunting_metric_df, gathering_metric_df], axis=0)
    else:
        metric_df = extract_metrics_partial(
            text_df=text_from_et,
            extract_metrics_for_text_df_kwargs=dict(
                ordered_prefix_col_names=[],
                keep_prefix_metrics=False,
                rebase_index_in_main_text=True,
            ),
        )

    metric_df = metric_df.rename({"index": "IA_ID", "Word": "IA_LABEL"}, axis=1)

    # Join metrics with eye_tracking_data
    et_data_enriched = eye_tracking_data.merge(
        metric_df,
        how="left",
        on=[
            "article_title",
            "has_preview",
            "question",
            "paragraph_id",
            "level",
            "IA_ID",
        ],
        validate="many_to_one",
    )

    return et_data_enriched


if __name__ == "__main__":
    et_data = pd.read_csv("intermediate_eye_tracking_data.csv")
    et_data_enriched = add_metrics_to_eye_tracking(
        eye_tracking_data=et_data,
        surprisal_extraction_model_names=["gpt2", "gpt2-medium"],
        spacy_model_name="en_core_web_sm",
        parsing_mode="re-tokenize",
        add_question_in_prompt=True,
        model_target_device="cuda:1",
        hf_access_token="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",
    )
    # Save the enriched data
    et_data_enriched.to_csv(
        "enriched_eye_tracking_data_enriched_surp.csv",
        index=False,
    )

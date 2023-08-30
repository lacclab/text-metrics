from typing import List

import pandas as pd
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_metrics.utils import get_metrics


def add_metrics_to_eye_tracking(
    eye_tracking_data: pd.DataFrame,
    surprisal_extraction_model_names: List[str],
    add_question_in_prompt: bool = False,
) -> pd.DataFrame:
    """
    Adds metrics to each row in the eye-tracking report

    :param eye_tracking_data: The eye-tracking report, each row represents a word that was read in a given trial.
                                Should have columns - ['article_title', 'paragraph_id', 'level', 'IA_ID']
    :param surprisal_extraction_model_names: the name of model/tokenizer to extract surprisal values from.
    :param add_question_in_prompt: whether to add the question in the prompt for the surprisal extraction model (applies for Hunting only).
    :return: eye-tracking report with surprisal, frequency and word length columns

    >>> et_data = load_et_data() # some function to load the eye tracking report
    >>> et_data_enriched = add_metrics_to_eye_tracking(eye_tracking_data=et_data, surprisal_extraction_model_names=['gpt2'])

    """

    # Extract metrics for all paragraph-level pairs
    metric_dfs = []

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

    tokenizers = [
        AutoTokenizer.from_pretrained(model_name)
        for model_name in surprisal_extraction_model_names
    ]

    models = [
        AutoModelForCausalLM.from_pretrained(model_name)
        for model_name in surprisal_extraction_model_names
    ]

    for row in tqdm.tqdm(
        text_from_et.reset_index().itertuples(),
        total=len(text_from_et),
        desc="Extracting metrics",
    ):
        actually_add_question_in_prompt = (
            add_question_in_prompt and row.has_preview == "Hunting"
        )
        if actually_add_question_in_prompt:
            text_input = row.question + " " + row.IA_LABEL
            num_question_words = len(row.question.split())
        else:
            text_input = row.IA_LABEL

        merged_df = get_metrics(
            text=text_input,
            models=models,
            tokenizers=tokenizers,
            model_names=surprisal_extraction_model_names,
        )

        # Remove the question from the output
        if actually_add_question_in_prompt:
            merged_df = merged_df[num_question_words:]

        merged_df[
            ["paragraph_id", "article_title", "level", "has_preview", "question"]
        ] = (
            row.paragraph_id,
            row.article_title,
            row.level,
            row.has_preview,
            row.question,
        )
        merged_df.reset_index(inplace=True)
        merged_df = merged_df.rename({"index": "IA_ID", "Word": "IA_LABEL"}, axis=1)
        if actually_add_question_in_prompt:
            merged_df["IA_ID"] -= num_question_words
        metric_dfs.append(merged_df)
    metric_df = pd.concat(metric_dfs, axis=0)

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
    et_data = pd.read_csv("/Users/shubi/eye-tracking/data/interim/et_data_enriched.csv")
    et_data_enriched = add_metrics_to_eye_tracking(
        eye_tracking_data=et_data, surprisal_extraction_model_names=["gpt2"]
    )

from typing import List, Literal
from pathlib import Path
import pandas as pd
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import torch
import gc
from text_metrics.utils import get_metrics, init_tok_n_model


def add_metrics_to_eye_tracking(
    eye_tracking_data: pd.DataFrame,
    surprisal_extraction_model_names: List[str],
    spacy_model_name: str,
    parsing_mode: Literal['keep-first','keep-all','re-tokenize'],
    add_question_in_prompt: bool = False,
    model_target_device: str = "cpu",
    hf_access_token: str = None,
) -> pd.DataFrame:
    """
    Adds metrics to each row in the eye-tracking report

    :param eye_tracking_data: The eye-tracking report, each row represents a word that was read in a given trial.
                                Should have columns - ['article_title', 'paragraph_id', 'level', 'IA_ID']
    :param surprisal_extraction_model_names: the name of model/tokenizer to extract surprisal values from.
    :param spacy_model_name: the name of the spacy model to use for parsing the text.
    :param parsing_mode: type of parsing to use. one of ['keep-first','keep-all','re-tokenize']
    :param add_question_in_prompt: whether to add the question in the prompt for the surprisal extraction model (applies for Hunting only).
    :return: eye-tracking report with surprisal, frequency and word length columns

    >>> et_data = load_et_data() # some function to load the eye tracking report
    >>> et_data_enriched = add_metrics_to_eye_tracking(eye_tracking_data=et_data, surprisal_extraction_model_names=['gpt2'])

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
    text_key_cols = ["paragraph_id", "article_title", "level", "has_preview", "question"]

    metric_df = None
    for i, model_name in enumerate(surprisal_extraction_model_names):
        # log the model name
        if i == 0: print(f"Extracting Frequency, Length")
        print(f"Extracting surprisal using model: {model_name}")
        
        tokenizer, model = init_tok_n_model(model_name=model_name, device='cpu', hf_access_token=hf_access_token)
        metric_dfs = []
        model.to(model_target_device)
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


            # add here new metrics
            merged_df = get_metrics(
                text=text_input,
                models=[model],
                tokenizers=[tokenizer],
                model_names=[surprisal_extraction_model_names[i]],
                parsing_model=spacy_model,
                parsing_mode=parsing_mode,
                add_parsing_features=True if metric_df is None else False,
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
        
        if metric_df is None:
            metric_df = pd.concat(metric_dfs, axis=0)
        else:
            concatenated_metric_dfs = pd.concat(metric_dfs, axis=0)
            cols_to_merge = concatenated_metric_dfs.columns.difference(metric_df.columns).tolist()
            cols_to_merge += text_key_cols + ["IA_ID"]
            
            metric_df = metric_df.merge(
                concatenated_metric_dfs[cols_to_merge],
                how="left",
                on=text_key_cols + ["IA_ID"],
                validate="one_to_one",
            )
        
        # move the model back to the cpu
        model.to('cpu')
        del model
        gc.collect()
        torch.cuda.empty_cache()

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
    et_data = pd.read_csv("/data/home/meiri.yoav/text-metrics/intermediate_eye_tracking_data.csv")
    et_data_enriched = add_metrics_to_eye_tracking(
        eye_tracking_data=et_data, surprisal_extraction_model_names=["meta-llama/Llama-2-7b-hf"],
        spacy_model_name="en_core_web_sm", parsing_mode="re-tokenize", add_question_in_prompt=False, model_target_device="cuda:1",
        hf_access_token="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"
    )
    # Save the enriched data
    et_data_enriched.to_csv("/data/home/meiri.yoav/text-metrics/enriched_eye_tracking_data_Llama_surp.csv", index=False)

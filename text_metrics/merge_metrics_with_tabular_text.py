import subprocess
from typing import List

import pandas as pd
import spacy
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_metrics.utils import get_metrics

try:
    _ = spacy.load("en_core_web_sm")
except OSError:
    print(
        "Downloading spacy model: en_core_web_sm (python -m spacy download en_core_web_sm)"
    )
    command = "python -m spacy download en_core_web_sm"
    subprocess.run(command, shell=True)


def add_metrics_tabular_text(
    tabular_text: pd.DataFrame, surprisal_extraction_model_names: List[str]
) -> pd.DataFrame:
    """
    Adds metrics to each row in the tabular_text DataFrame

    :param tabular_text: The input DataFrame with tabular text data, where each row represents a word that was read in a given trial.
                        Should have columns - ['item', 'wordnum', 'word']
    :param surprisal_extraction_model_names: The names of the model/tokenizer to extract surprisal values from.
    :return: The tabular_text DataFrame with added columns for surprisal, frequency, and word length metrics.

    >>> et_data = load_et_data() # some function to load the eye tracking report
    >>> et_data_enriched = add_metrics_tabular_text(tabular_text=et_data, surprisal_extraction_model_names=['gpt2'])

    """

    # Extract metrics for all paragraph-level pairs
    metric_dfs = []
    # Remove duplicates

    # Group by item and join all words
    grouped_text = tabular_text.groupby(["item"])["word"].apply(list)
    grouped_text = grouped_text.apply(lambda text: " ".join(text))

    tokenizers = [
        AutoTokenizer.from_pretrained(model_name)
        for model_name in surprisal_extraction_model_names
    ]
    models = [
        AutoModelForCausalLM.from_pretrained(model_name)
        for model_name in surprisal_extraction_model_names
    ]

    for index, sentence in tqdm.tqdm(
        grouped_text.items(), total=len(grouped_text), desc="Extracting metrics"
    ):
        merged_df = get_metrics(
            text=sentence,
            models=models,
            tokenizers=tokenizers,
            model_names=surprisal_extraction_model_names,
        )
        merged_df["item"] = index
        merged_df.reset_index(inplace=True)
        merged_df["index"] += 1
        merged_df = merged_df.rename({"index": "wordnum"}, axis=1)
        metric_dfs.append(merged_df)
    metric_df = pd.concat(metric_dfs, axis=0)

    # Join metrics with original data
    tabular_text_enriched = tabular_text.merge(
        metric_df,
        how="left",
        suffixes=("", "_SHUBI"),
        on=["item", "wordnum"],
        validate="many_to_one",
    )

    tabular_text_enriched.drop(["Word"], axis=1, inplace=True)
    tabular_text_enriched["subtlex_Frequency"].replace(0, "NA", inplace=True)
    return tabular_text_enriched


if __name__ == "__main__":

    stim = pd.read_csv("stim.csv", keep_default_na=False)
    tabular_text_enriched = add_metrics_tabular_text(stim, ["gpt2"])
    tabular_text_enriched.to_csv("stim_with_surprisal.csv", index=False)

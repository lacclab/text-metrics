import pandas as pd
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from eyeutils.utils import get_metrics


def add_metrics_to_eye_tracking(eye_tracking_data: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """
    Adds metrics to each row in the eye-tracking report

    :param eye_tracking_data: The eye-tracking report, each row represents a word that was read in a given trial.
                                Should have columns - ['article_title', 'paragraph_id', 'level', 'IA_ID']
    :param model: the model to extract surprisal values from.
    :param tokenizer: how to tokenize the text. Should match the model input expectations.
    :return: eye-tracking report with surprisal and frequency columns

    >>> et_data = load_et_data() # some function to load the eye tracking report
    >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
    >>> et_data_enriched = add_metrics_to_eye_tracking(eye_tracking_data=et_data, tokenizer=tokenizer, model=model)
    """

    # Load the OneStopQA HuggingFace dataset
    dataset = load_dataset(path='onestop_qa', split='train')
    df = dataset.to_pandas()
    df['level'] = dataset.features['level'].int2str(dataset['level'])

    # Extract metrics for all paragraph-level pairs
    metric_dfs = []
    for row in df[['paragraph', 'paragraph_index', 'title', 'level']].drop_duplicates().itertuples():
        print(row.Index, row.paragraph)
        merged_df = get_metrics(text=row.paragraph, tokenizer=tokenizer, model=model)
        merged_df['paragraph_id'] = row.paragraph_index + 1
        merged_df['article_title'] = row.title
        merged_df['level'] = row.level
        merged_df.reset_index(inplace=True)
        merged_df = merged_df.rename({'index': 'IA_ID', 'Word': 'IA_LABEL'}, axis=1)
        metric_dfs.append(merged_df)
        break  # TODO handle problematic frequency tokenization and duplicate words with '-' in report.
    metric_df = pd.concat(metric_dfs, axis=0)

    # Join metrics with eye_tracking_data
    et_data_enriched = eye_tracking_data.merge(metric_df, how='left',
                                               on=['article_title', 'paragraph_id', 'level', 'IA_ID'],
                                               validate='many_to_one')

    return et_data_enriched

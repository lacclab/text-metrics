import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from eyeutils.utils import get_metrics
import tqdm

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

    # Extract metrics for all paragraph-level pairs
    metric_dfs = []
    without_duplicates = eye_tracking_data[['paragraph_id', 'article_title', 'level', 'IA_ID', 'IA_LABEL']].drop_duplicates()
    text_from_et = without_duplicates.groupby(['paragraph_id', 'article_title', 'level'])['IA_LABEL'].apply(list)
    text_from_et = text_from_et.apply(lambda text: " ".join(text))

    for row in tqdm.tqdm(text_from_et.reset_index().itertuples()):
        merged_df = get_metrics(text=row.IA_LABEL, tokenizer=tokenizer, model=model)
        merged_df['paragraph_id'] = row.paragraph_id
        merged_df['article_title'] = row.article_title
        merged_df['level'] = row.level
        merged_df.reset_index(inplace=True)
        merged_df = merged_df.rename({'index': 'IA_ID', 'Word': 'IA_LABEL'}, axis=1)
        metric_dfs.append(merged_df)
    metric_df = pd.concat(metric_dfs, axis=0)

    # Join metrics with eye_tracking_data
    et_data_enriched = eye_tracking_data.merge(metric_df, how='left',
                                               on=['article_title', 'paragraph_id', 'level', 'IA_ID'],
                                               validate='many_to_one')

    return et_data_enriched


if __name__ == '__main__':

    et_data = pd.read_csv(r"C:\Users\omers\PycharmProjects\eye_tracking\data\et_data_small.csv")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    et_data_enriched = add_metrics_to_eye_tracking(eye_tracking_data=et_data, tokenizer=tokenizer, model=model)

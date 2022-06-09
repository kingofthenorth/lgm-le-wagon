import pandas as pd
from google.cloud import bigquery
from lgm_le_wagon.ml_logic.params import GCP_PROJECT, GBQ_DATASET

def get_data(task=None):
    """Method that query the training table
    Parameters
    ----------
    task -> str: [None, OOO, Sales_RH, Sentiment];
    None returns the full table
    ----------
    Returns: pandas.DataFrame with features and target for the required tasks
    """
    client = bigquery.Client(project=GCP_PROJECT)

    table = f"{GCP_PROJECT}.{GBQ_DATASET}.training_set"

    query_string = f"""
        SELECT *
        FROM {table}
    """

    if task == 'OOO':
        query_string = f"""
        SELECT log_id, reply, ooo
        FROM {table}"""
    if task == 'Sales_RH':
        query_string = f"""
        SELECT log_id, first_message, sales, RH
        FROM {table}
        WHERE ooo = 0"""
    if task=='Sentiment_FR':
        query_string = f"""
        SELECT log_id, reply, language, negative, neutral, positive
        FROM {table}
        WHERE language = 'fr'
        AND ooo = 0"""
    if task=='Sentiment_EN':
        query_string = f"""
        SELECT log_id, reply, language, negative, neutral, positive
        FROM {table}
        WHERE language = 'en'
        AND ooo = 0"""

    df = (
        client.query(query_string)
        .result()
        .to_dataframe())

    return df

def get_storage_data(task=None):
    data = pd.read_csv('gs://lgm-data/training_set.csv')
    
    if task == 'OOO':
        df = data[['log_id', 'reply', 'ooo']]
    
    if task == 'Sentiment_FR':
        df = data.query('language == "fr" and ooo == 0') \
            [['log_id', 'reply', 'negative', 'neutral', 'positive']]

    if task == 'Sentiment_EN':
        df = data.query('language == "en" and ooo == 0')\
            [['log_id', 'reply', 'negative', 'neutral', 'positive']]

    return df
    
if __name__=="__main__":
    print(get_data('Sentiment_FR').head(5))

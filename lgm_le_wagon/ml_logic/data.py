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
    client = bigquery.Client()
    
    table = f"{GCP_PROJECT}.{GBQ_DATASET}.all_doccano"

    query_string = f"""
        SELECT *
        FROM {table}
    """
    
    if task == 'OOO':
        query_string = f"""
        SELECT _id, reply, ooo
        FROM {table}"""
    if task == 'Sales_RH':
        query_string = f"""
        SELECT _id, first_message, sales, rh
        FROM {table}
        WHERE ooo = 0"""
    if task=='Sentiment':
        query_string = f"""
        SELECT _id, reply, negative, neutral, positive
        FROM {table}
        WHERE ooo = 0"""
        
    df = (
        client.query(query_string)
        .result()
        .to_dataframe())
    
    return df

if __name__=="__main__":
    print(get_data('Sentiment').head(5))

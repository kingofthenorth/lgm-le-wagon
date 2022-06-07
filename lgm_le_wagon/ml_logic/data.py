import pandas as pd
from google.cloud import bigquery
from lgm_le_wagon.ml_logic.params import GCP_PROJECT, GBQ_DATASET

def get_data():
    client = bigquery.Client()
    
    table = f"{GCP_PROJECT}.{GBQ_DATASET}.all_doccano"

    query = """
        SELECT *
        FROM @table
    """
    query_string = client.query(query)
    
    job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("table", "STRING", table),
    ])
    
    df = (
        client.query(query_string, job_config=job_config)
        .result()
        .to_dataframe())
    
    return df

if __name__=="__main__":
    print(get_data().head(5))

# #from lgm_le_wagon.ml_logic.params import (DATA_RAW_COLUMNS,
#                             DATA_RAW_DTYPES_OPTIMIZED,
#                             DATA_PROCESSED_DTYPES_OPTIMIZED)

# #from lgm_le_wagon.data_sources.local_disk import (get_pandas_chunk,
#                                     save_local_chunk)

# #from lgm_le_wagon.data_sources.big_query import (get_bq_chunk,
#                                     save_bq_chunk)





# def get_data(label:str = None):
#     """
#     data:
#     label: name of label that must be used as target [sales_or_hr, sentiment_sales, sentiment_rh]
#     """
#     # Load table from BQ, returning logid, reply and required label
#     if label == 'salesrh':
#         # SELECT logid, first_message, salesrh
#         pass
#     elif label == 'sentiment_sales':
#         pass
#         # SELECT logid, reply, sentiment WHERE sector='SALES'
#     elif label == 'sentiment_rh':
#         pass
#         # SELECT logid, reply, sentiment WHERE sector='RH'
#     pass

########################### TAXIFARE #####################################



# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     clean raw data by removing buggy or irrelevant transactions
#     or columns for the training set
#     """

#     # remove useless/redundant columns
#     df = df.drop(columns=['key'])

#     # remove buggy transactions
#     df = df.drop_duplicates()  # TODO: handle in the data source if the data is consumed by chunks
#     df = df.dropna(how='any', axis=0)
#     df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) |
#             (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
#     df = df[df.passenger_count > 0]
#     df = df[df.fare_amount > 0]

#     # remove irrelevant/non-representative transactions (rows) for a training set
#     df = df[df.fare_amount < 400]
#     df = df[df.passenger_count < 8]
#     df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
#     df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
#     df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
#     df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]

#     print("\n✅ data cleaned")

#     return df

# def get_chunk(source_name: str,
#               index: int = 0,
#               chunk_size: int = None) -> pd.DataFrame:
#     """
#     return a chunk of the dataset between `index` and `index + chunk_size - 1`
#     """

#     if "processed" in source_name:
#         columns = None
#         dtypes = DATA_PROCESSED_DTYPES_OPTIMIZED
#     else:
#         columns = DATA_RAW_COLUMNS
#         dtypes = DATA_RAW_DTYPES_OPTIMIZED

#     if os.environ.get("DATA_SOURCE") == "big query":

#         chunk_df = get_bq_chunk(table=source_name,
#                                 index=index,
#                                 chunk_size=chunk_size,
#                                 dtypes=dtypes)

#         return chunk_df

#     chunk_df = get_pandas_chunk(path=source_name,
#                                 index=index,
#                                 chunk_size=chunk_size,
#                                 dtypes=dtypes,
#                                 columns=columns)

#     return chunk_df


# def save_chunk(source_name: str,
#                is_first: bool,
#                data: pd.DataFrame) -> None:
#     """
#     save chunk
#     """

#     if os.environ.get("DATA_SOURCE") == "big query":

#         save_bq_chunk(table=source_name,
#                       data=data,
#                       is_first=is_first)

#         return

#     save_local_chunk(path=source_name,
#                      data=data,
#                      is_first=is_first)

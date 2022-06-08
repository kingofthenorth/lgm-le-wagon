
import csv
import pandas as pd

import os

from colorama import Fore, Style

def get_local_data():
    """
    return a chunk of the raw dataset from local disk
    """

    csv_path = os.path.join(os.getcwd(),"lgm_le_wagon","data_sources","local_replies.csv")
    print(csv_path)

    df = pd.read_csv(csv_path)

    return df


########################### TAXIFARE #####################################


# def get_pandas_chunk(path: str,
#                      index: int,
#                      chunk_size: int,
#                      dtypes,
#                      columns: list = None) -> pd.DataFrame:
#     """
#     return a chunk of the raw dataset from local disk or cloud storage
#     """

#     path = os.path.join(
#         os.environ.get("LOCAL_DATA_PATH"),
#         "processed" if "processed" in path else "raw",
#         f"{path}.csv")

#     print(Fore.MAGENTA + f"Source data from {path}: {chunk_size if chunk_size is not None else 'all'} rows (from row {index})" + Style.RESET_ALL)

#     try:

#         df = pd.read_csv(
#                 path,
#                 skiprows=index + 1,  # skip header
#                 nrows=chunk_size,
#                 dtype=dtypes,
#                 header=None)  # ignore header

#         if columns is not None:
#             df.columns = columns

#     except pd.errors.EmptyDataError:

#         return None  # end of data

#     return df


# def save_local_chunk(path: str,
#                      data: pd.DataFrame,
#                      is_first: bool):
#     """
#     save a chunk of the dataset to local disk
#     """

#     path = os.path.join(
#         os.environ.get("LOCAL_DATA_PATH"),
#         "raw" if "raw" in path else "processed",
#         f"{path}.csv")

#     print(Fore.BLUE + f"\nSave data to {path}:" + Style.RESET_ALL)

#     data.to_csv(path,
#                 mode="w" if is_first else "a",
#                 header=is_first,
#                 index=False)
import os
import shutil
import pandas as pd


def create_clean_folder(folder_path):
    """
    Creates a folder. If it already exists, removes it and recreates it.

    :param folder_path: Path of the folder to create.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Remove the existing folder and its contents
    os.makedirs(folder_path)  # Create the folder again


def remove_from_list(list1, list2):
    """
    Removes values in list2 from list1 and deletes list1.

    Parameters:
    - list1 (list): The list to be modified.
    - list2 (list): The list containing values to remove from list1.

    Returns:
    - The modified list1 (for reference, since list1 will be deleted).
    """
    # Remove items in list2 from list1
    return [a for a in list1 if a not in list2]


def convert_str_to_list(columns):
    if (columns is not None) and (type(columns) == str):
        return columns.split(",")
    return columns

def load_dataset(file_path, **kwargs):
    """
    Load a dataset from a CSV or Parquet file.

    Parameters:
    - file_path (str): Path to the file (CSV or Parquet).
    - kwargs: Additional arguments to pass to pandas' read_csv or read_parquet.

    Returns:
    - DataFrame: Loaded dataset as a pandas DataFrame.
    """
    # Determine file type by extension
    if file_path.endswith('.csv') or file_path.endswith('zip'):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")



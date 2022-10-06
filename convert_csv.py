import pandas as pd
import numpy as np
import torch


def array_to_csv(arr: np.ndarray, file_name: str):
    """
    Save an array in a CSV file.
    
    Inputs: 'arr': array to save.
            'file_name': name of the CSV file to create.
    """
    df = pd.DataFrame(arr)
    df.to_csv(f'{file_name}.csv', header=False, index=False)


def csv_to_array(file_name: str):
    """
    Load an array from a CSV file.

    Inputs: 'file_name': name of the CSV file to load.
            'num_columns': expected number of columns.

    Output: an array containing the information from the CSV file.
    """
    return np.array((pd.read_csv(file_name, header=None)).values)

def csv_to_tensor(file_name: str):
    """
    Load an array from a CSV file.

    Inputs: 'file_name': name of the CSV file to load.
            'num_columns': expected number of columns.

    Output: a tensor containing the information from the CSV file.
    """
    return torch.tensor((pd.read_csv(file_name, header=None)).values, dtype=torch.float32)



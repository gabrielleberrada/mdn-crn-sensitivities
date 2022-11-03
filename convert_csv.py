import pandas as pd
import numpy as np
import torch


def array_to_csv(arr: np.ndarray, file_name: str):
        """Saves an array in a CSV file.

        Args:
            - **arr** (np.ndarray): Array to save.
            - **file_name** (str): Name of the CSV file to create.
        """        
        df = pd.DataFrame(arr)
        df.to_csv(f'{file_name}.csv', mode='a', header=False, index=False)


def csv_to_array(file_name: str) -> np.ndarray:
        """Loads an array from a CSV file.

        Args:
            - **file_name** (str): Name of the CSV file to load.

        Returns:
            - An array containing the information from the CSV file.
        """        
        return np.array((pd.read_csv(file_name, header=None)).values)

def csv_to_tensor(file_name: str) -> torch.tensor:
        """Loads an array from a CSV file.

        Args:
            - **file_name** (str): Name of the CSV file to load.

        Returns:
            - A tensor containing the information from the CSV file.
        """        
        return torch.tensor((pd.read_csv(file_name, header=None)).values, dtype=torch.float32)



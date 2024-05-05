import pandas as pd
import os

def read_and_flatten_data(file_path, test_mode=False):
    """
    Reads job titles from a CSV file and flattens them into a unique list.
    
    Args:
        file_path (str): The path to the CSV file containing job titles.
        test_mode (bool): If True, only reads the first 10 rows for testing.
    
    Returns:
        list: A list of unique, non-null job titles.
    """
    if test_mode:
        data = pd.read_csv(file_path, nrows=10)
    else:
        data = pd.read_csv(file_path)
    flat_list = pd.unique(data.values.ravel('K'))
    return flat_list[~pd.isnull(flat_list)]

def preprocess_text(titles):
    """
    Processes a list of job titles by stripping, and converting to lower case.
    
    Args:
        titles (list): A list of job titles.
    
    Returns:
        list: The preprocessed job titles.
    """
    return [title.strip().lower() for title in titles]

if __name__ == "__main__":
    # Define the path components
    directory = "sample_data"
    filename = "titles.csv"
    file_path = os.path.join(directory, filename)

    titles = read_and_flatten_data(file_path, test_mode=True)
    preprocessed_titles = preprocess_text(titles)
    print(preprocessed_titles)
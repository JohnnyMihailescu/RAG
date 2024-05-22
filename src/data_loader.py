import pandas as pd
import os

def read_and_flatten_data(file_path, max_records=None):
    """
    Reads data from a CSV file and flattens them into a list.
    
    Args:
        file_path (str): The path to the CSV file containing data.
        max_records (int): The maximum number of records to read from the CSV file.
    
    Returns:
        list: A list of non-null values from the CSV file.
    """
    data = pd.read_csv(file_path, nrows=max_records)
    flat_list = data.values.ravel('K')
    return flat_list[~pd.isnull(flat_list)]

def preprocess_text(data):
    """
    Processes a list of text by stripping and converting to lower case.
    
    Args:
        data (list): A list of text.
    
    Returns:
        list: The preprocessed text.
    """
    return [str(item).strip().lower() for item in data]

def filter_non_unique_items(items):
    """
    Filters the list of items to include only non-unique items (those that occur more than once).
    
    Args:
        items (list): The list of items.
    
    Returns:
        list: A list of distinct non-unique items.
    """
    # Preprocess items
    preprocessed_items = preprocess_text(items)
    
    # Create a DataFrame to count item occurrences
    df = pd.DataFrame(preprocessed_items, columns=['item'])
    item_counts = df['item'].value_counts()
    
    # Filter non-unique items
    non_unique_items = item_counts[item_counts > 1].index.tolist()
    
    return non_unique_items
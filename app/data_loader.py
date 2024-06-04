import pandas as pd
import yaml

def read_and_flatten_data(file_path, max_records=None):
    """
    Reads job titles from a CSV file and flattens them into a list.
    
    Args:
        file_path (str): The path to the CSV file containing job titles.
        max_records (int): The maximum number of records to read from the CSV file.
    
    Returns:
        list: A list of non-null job titles.
    """
    data = pd.read_csv(file_path, nrows=max_records)
    flat_list = data.values.ravel('K')
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

def filter_non_unique_job_titles(titles):
    """
    Filters the list of titles to include only non-unique titles (those that occur more than once).
    
    Args:
        titles (list): The list of job titles.
    
    Returns:
        list: A list of distinct non-unique job titles.
    """
    # Preprocess titles
    preprocessed_titles = preprocess_text(titles)
    
    # Create a DataFrame to count title occurrences
    df = pd.DataFrame(preprocessed_titles, columns=['title'])
    title_counts = df['title'].value_counts()
    
    # Filter non-unique titles
    non_unique_titles = title_counts[title_counts > 1].index.tolist()
    
    return non_unique_titles

def load_yaml(yaml_path):
    """
    Loads data from a YAML file.
    
    Args:
        yaml_path (str): The path to the YAML file.
    
    Returns:
        dict: The data loaded from the YAML file.
    """
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data

def filter_titles_by_occurrences(title_counts, min_occurrences=2):
    """
    Filters titles by their occurrences.
    
    Args:
        title_counts (dict): A dictionary with titles as keys and their counts as values.
        min_occurrences (int): Minimum number of occurrences to keep a title.
    
    Returns:
        list: A list of titles that occur at least min_occurrences times.
    """
    filtered_titles = [title for title, count in title_counts.items() if count > min_occurrences]
    return filtered_titles
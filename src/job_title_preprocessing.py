from src.data_loader import preprocess_text
import pandas as pd

def filter_non_unique_job_titles(titles):
    """
    Filters the list of job titles to include only non-unique titles (those that occur more than once).
    
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

if __name__ == "__main__":
    # Example usage
    titles = ["Manager", "Developer", "Manager", "Engineer", "Developer", "CEO"]
    non_unique_titles = filter_non_unique_job_titles(titles)
    print(non_unique_titles)
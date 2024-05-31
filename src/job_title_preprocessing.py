from data_loader import preprocess_text

def filter_titles_by_occurrences_and_preprocess(title_counts, min_occurrences=2):
    """
    Filters titles by their occurrences and preprocesses them.
    
    Args:
        title_counts (dict): A dictionary with titles as keys and their counts as values.
        min_occurrences (int): Minimum number of occurrences to keep a title.
    
    Returns:
        list: A list of preprocessed titles that occur at least min_occurrences times.
    """
    # Filter titles by occurrences
    filtered_titles = [title for title, count in title_counts.items() if count >= min_occurrences]
    
    # Preprocess titles
    preprocessed_titles = preprocess_text(filtered_titles)
    
    return preprocessed_titles

if __name__ == "__main__":
    # Example usage
    title_counts = {
        "Manager": 3,
        "Developer": 1,
        "Engineer": 2,
        "CEO": 1
    }
    min_occurrences = 2
    filtered_preprocessed_titles = filter_titles_by_occurrences_and_preprocess(title_counts, min_occurrences)
    print(filtered_preprocessed_titles)
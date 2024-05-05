from sentence_transformers import SentenceTransformer
import os

def generate_embeddings(titles):
    """
    Generates embeddings for a list of preprocessed job titles using a pre-trained model.
    
    Args:
        titles (list): A list of preprocessed job titles.
    
    Returns:
        numpy.ndarray: The embeddings for the given job titles.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(titles)

if __name__ == "__main__":
    # Define the path components
    directory = "sample_data"
    filename = "titles.csv"
    file_path = os.path.join(directory, filename)

    # This part assumes that data_loader.py and embedding_generator.py are in the same directory
    from data_loader import preprocess_text, read_and_flatten_data

    titles = read_and_flatten_data(file_path, test_mode=True)
    preprocessed_titles = preprocess_text(titles)
    embeddings = generate_embeddings(preprocessed_titles)
    print(embeddings)
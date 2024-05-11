from sentence_transformers import SentenceTransformer
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def find_most_similar(input_embedding, embeddings, n=5):
    """
    Finds the top n most similar embeddings to the input embedding.
    
    Args:
        input_embedding (numpy.ndarray): The input embedding.
        embeddings (numpy.ndarray): The set of embeddings to query against.
        n (int): The number of most similar embeddings to return.
    
    Returns:
        list: The indices of the top n most similar embeddings.
    """
    similarities = cosine_similarity(input_embedding.reshape(1, -1), embeddings)
    top_n_indices = np.argsort(similarities[0])[-n:]
    return top_n_indices.tolist()

def query_similar_job_titles(job_title, n=5):
    """
    Queries the most similar job titles to the given job title.
    
    Args:
        job_title (str): The job title to query.
        n (int): The number of most similar job titles to return.
    
    Returns:
        list: The indices of the top n most similar job titles.
    """
    # Generate an embedding for the job title
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_embedding = model.encode([job_title])

    # Load the embeddings from the .npy file
    embeddings = np.load('embeddings.npy')

    # Find the most similar embeddings
    most_similar_indices = find_most_similar(input_embedding, embeddings, n=n)

    # Load the titles
    directory = "sample-data"
    filename = "titles.csv"
    file_path = os.path.join(directory, filename)
    from data_loader import read_and_flatten_data
    titles = read_and_flatten_data(file_path, test_mode=True)

    # Return the most similar titles
    return [titles[i] for i in most_similar_indices]

if __name__ == "__main__":
    # Define the path components
    directory = "sample-data"
    filename = "titles.csv"
    file_path = os.path.join(directory, filename)

    # This part assumes that data_loader.py and embedding_generator.py are in the same directory
    from data_loader import preprocess_text, read_and_flatten_data

    titles = read_and_flatten_data(file_path, test_mode=True)
    preprocessed_titles = preprocess_text(titles)
    embeddings = generate_embeddings(preprocessed_titles)

    # Save the embeddings to a .npy file
    np.save('embeddings.npy', embeddings)
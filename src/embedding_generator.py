from sentence_transformers import SentenceTransformer
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from data_loader import read_and_flatten_data, filter_non_unique_titles

def generate_embeddings(titles):
    """
    Generates embeddings for a list of preprocessed job titles using a pre-trained model.
    
    Args:
        titles (list): A list of preprocessed job titles.
    
    Returns:
        tuple: A tuple containing the embeddings for the given job titles and counts of successful and failed embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    embeddings = []
    success_count = 0
    fail_count = 0

    print(f"Total job titles to embed: {len(titles)}")
    
    for title in tqdm(titles, desc="Generating embeddings"):
        try:
            embedding = model.encode([title])
            embeddings.append(embedding[0])
            success_count += 1
        except Exception as e:
            print(f"Failed to embed title '{title}': {e}")
            fail_count += 1

    return np.array(embeddings), success_count, fail_count

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

def query_similar_job_titles(job_title, embeddings_path, titles_path, n=5):
    """
    Queries the most similar job titles to the given job title.
    
    Args:
        job_title (str): The job title to query.
        embeddings_path (str): The file path to the embeddings .npy file.
        titles_path (str): The file path to the titles .npy file.
        n (int): The number of most similar job titles to return.
    
    Returns:
        list: The most similar job titles.
    """
    # Generate an embedding for the job title
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_embedding = model.encode([job_title])

    # Load the embeddings from the .npy file
    embeddings = np.load(embeddings_path)

    # Find the most similar embeddings
    most_similar_indices = find_most_similar(input_embedding, embeddings, n=n)

    # Load the titles
    all_titles = np.load(titles_path)

    # Return the most similar titles
    return [all_titles[i] for i in most_similar_indices]

if __name__ == "__main__":

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Define the path components
    directory = "sample-data"
    titles_filename = "kaggle_histories_titles.csv"
    ipod_filename = "IPOD_titles.csv"
    titles_file_path = os.path.join(directory, titles_filename)
    ipod_file_path = os.path.join(directory, ipod_filename)
    max_records = 50000

    # Load data from both files
    titles = read_and_flatten_data(titles_file_path, max_records)
    ipod_titles = read_and_flatten_data(ipod_file_path, max_records)
    
    # Combine the lists of titles
    combined_titles = list(titles) + list(ipod_titles)
    
    # Filter to non-unique titles
    non_unique_titles = filter_non_unique_titles(combined_titles)
    
    # Print the total number of job titles to be embedded
    print(f"Total number of non-unique job titles to process: {len(non_unique_titles)}")

    # Generate embeddings for non-unique titles
    embeddings, success_count, fail_count = generate_embeddings(non_unique_titles)

    # Save the embeddings to a .npy file
    np.save('embeddings.npy', embeddings)
    np.save('titles.npy', np.array(non_unique_titles))

    # Print the counts of successful and failed embeddings
    print(f"Number of job titles successfully embedded: {success_count}")
    print(f"Number of job titles failed to embed: {fail_count}")
import numpy as np
from data_loader import preprocess_text, read_and_flatten_data
from embedding_generator import generate_embeddings, find_most_similar
from llm_prompter import generate_prompt, call_openai
from sentence_transformers import SentenceTransformer
import os

def embed_dataset(data_path, save_path):
    """
    Loads data from the given path, generates embeddings for all titles, and stores the result in a local file.
    
    Args:
        data_path (str): The path to the data.
        save_path (str): The path to save the embeddings.
    """
    titles = read_and_flatten_data(data_path, test_mode=True)
    preprocessed_titles = preprocess_text(titles)
    embeddings = generate_embeddings(preprocessed_titles)

    # Save the embeddings to a .npy file
    np.save(save_path, embeddings)

def query_single_title(job_title, embeddings_path, data_path, n=5):
    """
    Queries the most similar job titles to the given job title.
    
    Args:
        job_title (str): The job title to query.
        embeddings_path (str): The path to the saved embeddings.
        data_path (str): The path to the data.
        n (int): The number of most similar job titles to return.
    
    Returns:
        list: The indices of the top n most similar job titles.
    """
    # Generate an embedding for the job title
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_embedding = model.encode([job_title])

    # Load the embeddings from the .npy file
    embeddings = np.load(embeddings_path)

    # Find the most similar embeddings
    most_similar_indices = find_most_similar(input_embedding, embeddings, n=n)

    # Load the titles
    titles = read_and_flatten_data(data_path, test_mode=True)

    # Return the most similar titles
    return [titles[i] for i in most_similar_indices]

def get_variations(job_title, embeddings_path, data_path, n=5, model_name="gpt-3.5-turbo-0125"):
    """
    Gets the variations of a job title by querying similar job titles and calling the OpenAI API.
    
    Args:
        job_title (str): The job title to get variations for.
        embeddings_path (str): The path to the saved embeddings.
        data_path (str): The path to the data.
        n (int): The number of most similar job titles to return.
        model_name (str): The name of the model to use. Defaults to "gpt-3.5-turbo-0125".
    
    Returns:
        list: The variations of the job title.
    """
    # Query the most similar job titles
    similar_titles = query_single_title(job_title, embeddings_path, data_path, n=n)

    # Generate prompts for the similar job titles
    prompt = generate_prompt(job_title, similar_titles)

    # Call the OpenAI API to get the variations
    variations = call_openai(prompt, model_name)

    return variations

if __name__ == "__main__":
    # Call the get_variations function with some example parameters
    variations = get_variations("Developer", "embeddings.npy", "sample-data/titles.csv")
    print(variations)
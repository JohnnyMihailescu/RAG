import numpy as np
from data_loader import preprocess_text, read_and_flatten_data
from embedding_generator import generate_embeddings, find_most_similar, query_similar_job_titles
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

def get_variations(job_title, embeddings_path, titles_path, n=5, model_name="gpt-3.5-turbo"):
    """
    Gets the variations of a job title by querying similar job titles and calling the OpenAI API.
    
    Args:
        job_title (str): The job title to get variations for.
        embeddings_path (str): The path to the saved embeddings.
        titles_path (str): The path to the saved titles.
        n (int): The number of most similar job titles to return.
        model_name (str): The name of the model to use. Defaults to "gpt-3.5-turbo".
    
    Returns:
        list: The variations of the job title.
    """
    # Query the most similar job titles
    similar_titles = query_similar_job_titles(job_title, embeddings_path, titles_path, n=n)

    # Generate prompts for the similar job titles
    prompt = generate_prompt(job_title, similar_titles)

    # Call the OpenAI API to get the variations
    variations = call_openai(prompt, model_name)

    return variations

if __name__ == "__main__":
    # Call the get_variations function with some example parameters
    print(os.getcwd())
    embeddings_path = os.path.join('embeddings.npy')
    titles_path = os.path.join('titles.npy')
    model_name = "gpt-3.5-turbo"
    variations = get_variations("Radar technician expert", embeddings_path, titles_path, n=10, model_name=model_name)
    print(variations)
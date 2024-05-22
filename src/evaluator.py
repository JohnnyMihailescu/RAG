import os
import numpy as np
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,
)
from data_loader import preprocess_text, read_and_flatten_data
from embedding_generator import query_similar_job_titles
from llm_prompter import generate_prompt, call_openai, extract_normalized_title
import random
import logging
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables. 
load_dotenv('C:\git\RAG\project.env')
# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Print the API key to verify it is loaded correctly
if api_key:
    print(f"OpenAI API key loaded successfully, length: {len(api_key)}")
else:
    print("Failed to load OpenAI API key.")

def load_combined_unique_titles(titles_file_path, ipod_file_path):
    """
    Loads and combines unique job titles from two datasets.
    
    Args:
        titles_file_path (str): Path to the first dataset.
        ipod_file_path (str): Path to the second dataset.
    
    Returns:
        list: List of unique job titles combined from both datasets.
    """
    # Load data from both files
    titles = read_and_flatten_data(titles_file_path)
    ipod_titles = read_and_flatten_data(ipod_file_path)
    
    # Combine and preprocess the lists of titles
    combined_titles = list(titles) + list(ipod_titles)
    preprocessed_titles = preprocess_text(combined_titles)
    
    # Create a DataFrame to get unique titles
    df = pd.DataFrame(preprocessed_titles, columns=['title'])
    title_counts = df['title'].value_counts()
    
    # Filter unique titles
    unique_titles = title_counts[title_counts == 1].index.tolist()
    
    return unique_titles
    
# Evaluate the RAG pipeline
def evaluate_rag_pipeline(sample_titles, embeddings_path, titles_path, number_of_similar_titles, model_name="gpt-3.5-turbo"):
    results = []
    contexts = []
    
    for job_title in sample_titles:
        normalized_title, similar_titles = normalize_job_title(job_title, embeddings_path, 
                                                               titles_path, number_of_similar_titles,  model_name)
        results.append({'original_job_title': job_title, 'normalized_job_title': normalized_title})
        contexts.append(similar_titles)
    
    return pd.DataFrame(results), contexts

def normalize_job_title(job_title, embeddings_path, titles_path, n=5, model_name="gpt-3.5-turbo"):
    """
    Normalizes a job title by querying similar job titles and calling the OpenAI API.
    
    Args:
        job_title (str): The job title to normalize.
        embeddings_path (str): The path to the saved embeddings.
        titles_path (str): The path to the saved titles.
        n (int): The number of most similar job titles to return.
        model_name (str): The name of the model to use. Defaults to "gpt-3.5-turbo".
    
    Returns:
        str: The normalized job title.
    """
    # Query the most similar job titles
    similar_titles = query_similar_job_titles(job_title, embeddings_path, titles_path, n=n)

    # Generate prompts for the similar job titles
    prompt = generate_prompt(job_title, similar_titles)

    # Call the OpenAI API to get the variations
    response = call_openai(prompt, model_name)

    # Extract the normalized job title from the response
    normalized_title = extract_normalized_title(response)

    return normalized_title, similar_titles

if __name__ == "__main__":
    # Define the path components
    directory = "sample-data"
    titles_filename = "kaggle_histories_titles.csv"
    ipod_filename = "IPOD_titles.csv"
    titles_file_path = os.path.join(directory, titles_filename)
    ipod_file_path = os.path.join(directory, ipod_filename)
    embeddings_path = os.path.join('embeddings.npy')
    titles_path = os.path.join('titles.npy')
    results_output_path = os.path.join(directory, 'normalization_results.csv')
    ragas_output_path = os.path.join(directory, 'ragas_evaluation_results.csv')
    test_dataset_sample = 10
    number_of_similar_titles = 10

    # Load combined unique job titles
    unique_titles = load_combined_unique_titles(titles_file_path, ipod_file_path)
    
    # Sample 1000 random titles from the unique titles
    sample_titles = random.sample(unique_titles, test_dataset_sample)
    
    # Evaluate the RAG pipeline on the sampled titles
    results_df, contexts = evaluate_rag_pipeline(sample_titles, embeddings_path, titles_path, 
                                                 number_of_similar_titles, model_name="gpt-3.5-turbo")
    results_df.to_csv(results_output_path, index=False)
    
    # Prepare dataset for evaluation
    questions = sample_titles
    ground_truths = []  # Add ground truth data if available
    answers = results_df['normalized_job_title'].tolist()

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    dataset = Dataset.from_dict(data)

    # Evaluate using RAGAs
    result = evaluate(
        dataset=dataset,
        metrics=[context_utilization, faithfulness, answer_relevancy],
    )
    ragas_df = result.to_pandas()


    # Save the ragas results to a CSV file
    ragas_df.to_csv(ragas_output_path, index=False)
    
    print(f"Evaluation results saved to {ragas_output_path}")
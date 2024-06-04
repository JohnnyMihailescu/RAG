import os
import pandas as pd
import random
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,
)
from app.data_loader import read_and_flatten_data
from app.embedding_generator import query_similar_items
from app.job_title_prompter import generate_job_title_prompt, extract_normalized_title
from app.llm_prompter import call_openai

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
    
    # Combine the lists of titles
    combined_titles = list(titles) + list(ipod_titles)
    
    # Filter non-unique job titles
    non_unique_titles = filter_non_unique_job_titles(combined_titles)
    
    return non_unique_titles
    
# Evaluate the RAG pipeline
def evaluate_rag_pipeline(sample_titles, embeddings_path, titles_path, number_of_similar_titles, model_name, prompt_template):
    results = []
    contexts = []
    
    for job_title in sample_titles:
        normalized_title, similar_titles = normalize_job_title(job_title, embeddings_path, 
                                                               titles_path, number_of_similar_titles, model_name, prompt_template)
        results.append({'original_job_title': job_title, 'normalized_job_title': normalized_title})
        contexts.append(similar_titles)
    
    return pd.DataFrame(results), contexts

def normalize_job_title(job_title, embeddings_path, titles_path, n, model_name, prompt_template):
    """
    Normalizes a job title by querying similar job titles and calling the OpenAI API.
    
    Args:
        job_title (str): The job title to normalize.
        embeddings_path (str): The path to the saved embeddings.
        titles_path (str): The path to the saved titles.
        n (int): The number of most similar job titles to return.
        model_name (str): The name of the model to use.
        prompt_template (str): The prompt template to use.
    
    Returns:
        str: The normalized job title.
    """
    # Query the most similar job titles
    similar_titles = query_similar_items(job_title, embeddings_path, titles_path, n=n)

    # Generate prompts for the similar job titles
    prompt = generate_job_title_prompt(prompt_template, job_title, similar_titles)

    # Call the OpenAI API to get the variations
    response = call_openai(prompt, model_name)

    # Extract the normalized job title from the response
    normalized_title = extract_normalized_title(response)

    return normalized_title, similar_titles

def evaluate_rag_pipeline(config):
    # Define the path components
    directory = "data"
    titles_filename = config['titles_filename']
    ipod_filename = config['ipod_filename']
    titles_file_path = os.path.join(directory, titles_filename)
    ipod_file_path = os.path.join(directory, ipod_filename)
    embeddings_path = config['embedding_path']
    titles_path = config['titles_path']
    results_output_path = config['results_output_path']
    ragas_output_path = config['ragas_output_path']
    test_dataset_sample = config['test_dataset_sample']
    number_of_similar_titles = config['number_of_similar_titles']
    model_name = config['model_name']
    prompt_template = config['prompt_template']

    # Load combined unique job titles
    unique_titles = load_combined_unique_titles(titles_file_path, ipod_file_path)
    
    # Sample random titles from the unique titles
    sample_titles = random.sample(unique_titles, test_dataset_sample)
    
    # Evaluate the RAG pipeline on the sampled titles
    results_df, contexts = evaluate_rag_pipeline(sample_titles, embeddings_path, titles_path, 
                                                 number_of_similar_titles, model_name, prompt_template)
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

    # Save the RAGAs results to a CSV file
    ragas_df.to_csv(ragas_output_path, index=False)
    
    print(f"Evaluation results saved to {ragas_output_path}")
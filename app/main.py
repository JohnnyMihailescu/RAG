import numpy as np
import os
import yaml
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv
from app.evaluator import normalize_job_title
from app.embedding_generator import generate_embeddings, bulk_insert_embeddings_to_elasticsearch
from app.job_title_preprocessing import filter_titles_by_occurrences_and_preprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from config.yaml
config_file_path = "config.yaml"
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

logger.info("Config loaded.")
directory = "data"
titles_filename = config['titles_filename']
ipod_filename = config['ipod_filename']
titles_file_path = os.path.join(directory, titles_filename)
ipod_file_path = os.path.join(directory, ipod_filename)
embeddings_path = config['embedding_path']
titles_path = config['titles_path']
results_output_path = config['results_output_path']
ragas_output_path = config['ragas_output_path']
test_dataset_sample_size = config['test_dataset_sample_size']
number_of_similar_titles = config['number_of_similar_titles']
model_name = config['model_name']
prompt_template = config['prompt_template']

# Define Request Models. 
class JobTitleRequest(BaseModel):
    job_title: str
    n_similar: int = number_of_similar_titles

class EvaluationRequest(BaseModel):
    sample_size: int = test_dataset_sample_size


class GenerateEmbeddingsRequest(BaseModel):
    min_occurrences: Optional[int] = 2

# Load local data for lifespan of the app.
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load config and data from local files. 
    # Define the path components
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load env variables. 
    load_dotenv('C:\git\RAG\project.env')
    # Get the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")

    # Print the API key to verify it is loaded correctly
    if api_key:
        logger.info(f"OpenAI API key loaded successfully, length: {len(api_key)}")
    else:
        logger.info("Failed to load OpenAI API key.")
    yield


# Initialize FastAPI app. 
app = FastAPI(lifespan=lifespan)

# Normalize single job title endpoint. 
@app.post("/normalize_job_title")
async def api_normalize_job_title(job_title_request: JobTitleRequest):
    print 
    job_title = job_title_request.job_title
    n_similar = job_title_request.n_similar
    normalized_title, similar_title = normalize_job_title(job_title, embeddings_path, titles_path, 
                                                          n_similar, model_name, prompt_template)
    return {"normalized_title": normalized_title, "similar_titles": similar_title}

# Generate Embeddings endpoint.
@app.post("/generate_embeddings")
def api_generate_embeddings(yaml_file: UploadFile = File(...), request: GenerateEmbeddingsRequest = Depends()):
    
    yaml_data = yaml.safe_load(yaml_file.file)

    # Filter titles by occurrences
    min_occurrences = request.min_occurrences
    filtered_preprocessed_titles = filter_titles_by_occurrences_and_preprocess(yaml_data, min_occurrences=min_occurrences)
    
    # Generate embeddings for the filtered titles
    embeddings, gen_success_count, gen_fail_count = generate_embeddings(filtered_preprocessed_titles)


    # Bulk insert embeddings into Elasticsearch
    bulk_insert_embeddings_to_elasticsearch(embeddings)

    return {
        "message": "Embeddings generated and stored successfully.",
        "generation_success_count": gen_success_count,
        "generation_fail_count": gen_fail_count,
        "filtered_titles_count": len(filtered_preprocessed_titles)
    }

from sentence_transformers import SentenceTransformer
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from elasticsearch import Elasticsearch, helpers

def connect_to_elasticsearch():
    """
    Connects to the Elasticsearch server.
    
    Returns:
        Elasticsearch: The Elasticsearch client.
    """
    es_client = Elasticsearch("https://localhost:9200", 
                          api_key="YlpkcjQ0OEJfUTA5YlhhTUpCblU6OWJ5OUVsb2ZSNU9wWlY3MUlpSkpvdw==", 
                          verify_certs=False,
                          ssl_show_warn=False)
    return es_client


def generate_embeddings(items, model_name='all-MiniLM-L6-v2'):
    """
    Generates embeddings for a list of items using a pre-trained model.
    
    Args:
        items (list): A list of items to embed.
        model_name (str): The name of the pre-trained model to use. Defaults to 'all-MiniLM-L6-v2'.
    
    Returns:
        tuple: A tuple containing a list of embeddings and counts of successful and failed embeddings.
    """
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    embeddings = []
    success_count = 0
    fail_count = 0

    print(f"Total items to embed: {len(items)}")

    for item in tqdm(items, desc="Generating embeddings"):
        try:
            embedding = model.encode([item])[0]
            embeddings.append({
                'title': item,
                'embedding': embedding
            })
            success_count += 1
        except Exception as e:
            print(f"Failed to embed item '{item}': {e}")
            fail_count += 1
    
    return embeddings, success_count, fail_count

def bulk_insert_embeddings_to_elasticsearch(embeddings, index_name='job_titles'):
    """
    Bulk inserts embeddings into Elasticsearch.
    
    Args:
        embeddings (list): A list of dictionaries containing items and their embeddings.
        index_name (str): The name of the Elasticsearch index. Defaults to 'job_titles'.
    """
    es_client = connect_to_elasticsearch()

    actions = [
        {
            "_index": index_name,
            "_source": {
                "title": embedding['title'],
                "embedding": embedding['embedding']
            }
        }
        for embedding in embeddings
    ]

    helpers.bulk(es_client, actions)

def elastic_query_similar_items(item, model_name='all-MiniLM-L6-v2', index_name='job_titles', n=5):           
    """
    Queries the most similar items to the given item in Elasticsearch.
    
    Args:
        item (str): The item to query.
        model_name (str): The name of the pre-trained model to use. Defaults to 'all-MiniLM-L6-v2'.
        index_name (str): The name of the Elasticsearch index. Defaults to 'items'.
        n (int): The number of most similar items to return.
    
    Returns:
        list: The most similar items.
    """
    es_client = connect_to_elasticsearch()
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    input_embedding = model.encode([item])[0].tolist()

    query = {
        "knn": {
            "field": "embedding",
            "query_vector": input_embedding,
            "k" : n,
            "num_candidates": 1000
        } 
    }

    response = es_client.search(index=index_name, body=query)
    hits = response['hits']['hits']
    similar_items = [hit['_source']['title'] for hit in hits[:n]]

    return similar_items

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

def query_similar_items(item, embeddings_path, items_path, n=5, model_name='all-MiniLM-L6-v2'):
    """
    Queries the most similar items to the given item.
    
    Args:
        item (str): The item to query.
        embeddings_path (str): The file path to the embeddings .npy file.
        items_path (str): The file path to the items .npy file.
        n (int): The number of most similar items to return.
        model_name (str): The name of the pre-trained model to use. Defaults to 'all-MiniLM-L6-v2'.
    
    Returns:
        list: The most similar items.
    """
    # Generate an embedding for the item
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_embedding = model.encode([item])

    # Load the embeddings from the .npy file
    embeddings = np.load(embeddings_path)

    # Find the most similar embeddings
    most_similar_indices = find_most_similar(input_embedding, embeddings, n=n)

    # Load the items
    all_items = np.load(items_path)

    # Return the most similar items
    return [all_items[i] for i in most_similar_indices]

# Example usage
if __name__ == "__main__":

    items = ["Senior Software Engineer", "Data Scientist", "Product Manager", "Graphic Designer"]
    embeddings = generate_embeddings(items)
    
    # Insert the embeddings into Elasticsearch
    bulk_insert_embeddings_to_elasticsearch(embeddings)

    item = "Software Developer"
    similar_items = elastic_query_similar_items(item, n=3)
    print("Similar items:", similar_items)
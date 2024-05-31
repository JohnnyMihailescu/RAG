import logging
from llm_prompter import generate_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_job_title_prompt(prompt_template, job_title, similar_titles):
    """
    Generates a prompt for a job title and a set of similar titles.
    
    Args:
        prompt_template (str): The prompt template.
        job_title (str): The job title to generate a prompt for.
        similar_titles (list): The similar job titles.
    
    Returns:
        str: The prompt for the job title and the similar titles.
    """
    return generate_prompt(prompt_template, job_title, similar_titles, entity_type="job title")

def extract_normalized_title(response):
    """
    Extracts the normalized job title from the OpenAI API response.
    
    Args:
        response (str): The response from the OpenAI API.
    
    Returns:
        str: The normalized job title.
    """
    if 'normalized_job_title:' in response:
        normalized_title = response.split('normalized_job_title:')[1].strip().strip('"')
        return normalized_title
    return response
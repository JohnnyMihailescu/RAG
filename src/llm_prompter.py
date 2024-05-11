import openai
import logging

PROMPT_TEMPLATE = "{} is a job title. Given the similar titles {}, which of these are variations of the job title?"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_prompt(title, similar_titles):
    """
    Generates a prompt for a job title and a set of similar titles.
    
    Args:
        title (str): The job title to generate a prompt for.
        similar_titles (list): The similar job titles.
    
    Returns:
        str: The prompt for the job title and the similar titles.
    """
    # Join the similar titles with commas
    similar_titles_str = ", ".join(similar_titles)
    
    # Format the PROMPT_TEMPLATE with the title and the similar titles
    return PROMPT_TEMPLATE.format(title, similar_titles_str)

def load_openai_key():
    """
    Loads the OpenAI API key from a text file.
    
    Returns:
        str: The OpenAI API key.
    """
    with open("api_key.txt", "r") as file:
        openai_key = file.read().strip()
    return openai_key


def call_openai(prompt, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=4000):
    """
    Calls the OpenAI API for each prompt and returns the responses.
    
    Args:
        prompts (list): The prompts to send to the OpenAI API.
        model_name (str): The name of the model to use. Defaults to "gpt-3.5-turbo".
        temperature (float): The temperature parameter for generating responses. Defaults to 0.5.
        max_tokens (int): The maximum number of tokens in the generated response. Defaults to 100.
    
    Returns:
        list: The responses from the OpenAI API.
    """
    api_key = load_openai_key()
    client = openai.OpenAI(api_key=api_key)

    responses = []
    logger.info(f"Sending prompt to OpenAI: {prompt}")
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    logger.info(f"Received response from OpenAI: {response.choices[0].message.content}")
    responses.append(response.choices[0].message.content.strip())

    return responses
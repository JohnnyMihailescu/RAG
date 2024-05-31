import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_prompt(template, entity, context, entity_type="entity"):
    """
    Generates a prompt for a given entity and a set of context items.
    
    Args:
        template (str): The prompt template.
        entity (str): The entity to generate a prompt for.
        context (list): The context items related to the entity.
        entity_type (str): The type of entity. Defaults to "entity".
    
    Returns:
        str: The formatted prompt.
    """
    # Join the context items with newline and space indentation for YAML list format
    context_str = '\n'.join([f'    - "{item}"' for item in context])
    
    # Format the template with the entity and the context items
    return template.format(query=entity, context=context_str)


def call_openai(prompt, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=4000):
    """
    Calls the OpenAI API for each prompt and returns the responses.
    
    Args:
        prompt (str): The prompt to send to the OpenAI API.
        model_name (str): The name of the model to use. Defaults to "gpt-3.5-turbo".
        temperature (float): The temperature parameter for generating responses. Defaults to 0.5.
        max_tokens (int): The maximum number of tokens in the generated response. Defaults to 4000.
    
    Returns:
        str: The response from the OpenAI API.
    """
    api_key = load_openai_key()
    client = openai.OpenAI(api_key=api_key)

    logger.info(f"Sending prompt to OpenAI: {prompt}")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    logger.info(f"Received response from OpenAI: {response.choices[0].message.content}")
    
    return response.choices[0].message.content.strip()

def load_openai_key():
    """
    Loads the OpenAI API key from a text file.
    
    Returns:
        str: The OpenAI API key.
    """
    with open("../api_key.txt", "r") as file:
        openai_key = file.read().strip()
    return openai_key

def extract_normalized_title(response):
    """
    Extracts the normalized job title from the OpenAI API response.
    
    Args:
        response (str): The response from the OpenAI API.
    
    Returns:
        str: The normalized job title.
    """
    if 'normalized_title:' in response:
        normalized_title = response.split('normalized_title:')[1].strip().strip('"')
        return normalized_title
    return response
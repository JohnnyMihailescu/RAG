import openai
import logging

PROMPT_TEMPLATE = """
task: Normalize the given job title by selecting the most common acceptable form from the provided similar titles. 
Make sure the normalized title retains all valuable information that makes the input title a distinct role/responsibility. 
Extraneous information that does not signify a distinct role and responsibility should not be retained in the normalized title, 
this includes things like the company names, department name, location names. 
Output using the YAML format specified in the example below without any additional text. 

  Input Example:
    job_title: "Senior Software Engineer at Google"
    similar_titles:
      - "Senior Developer at Google"
      - "Lead Software Engineer"
      - "Software Engineer at Google"
      - "software engineer" 
      - "software developer" 
      - "Senior Engineer" 
  Output Example:
    normalized_title: "Software Engineer"

Here is the input job title and similar titles:
  job_title: "{title}"
  similar_titles:
{similar_titles}
"""

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
    # Join the similar titles with newline and space indentation for YAML list format
    similar_titles_str = '\n'.join([f'    - "{t}"' for t in similar_titles])
    
    # Format the PROMPT_TEMPLATE with the title and the similar titles
    return PROMPT_TEMPLATE.format(title=title, similar_titles=similar_titles_str)

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
    with open("api_key.txt", "r") as file:
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
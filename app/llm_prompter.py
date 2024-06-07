import boto3
import logging
import os
from botocore.exceptions import ClientError

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


def call_bedrock(prompt, model_id="anthropic.claude-3-haiku-20240307-v1:0", temperature=0.5, max_tokens=4000):
    """
    Calls the AWS Bedrock API for each prompt and returns the responses.
    
    Args:
        prompt (str): The prompt to send to the AWS Bedrock API.
        model_id (str): The ID of the model to use. Defaults to "anthropic.claude-3-haiku-20240307-v1:0".
        temperature (float): The temperature parameter for generating responses. Defaults to 0.5.
        max_tokens (int): The maximum number of tokens in the generated response. Defaults to 4000.
    
    Returns:
        str: The response from the AWS Bedrock API.
    """
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',  # Replace with your Bedrock region
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

    logger.info(f"Sending prompt to Bedrock: {prompt}")

    conversation = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": 0.9},
        )
        response_text = response["output"]["message"]["content"][0]["text"]
        logger.info(f"Received response from Bedrock: {response_text}")
        return response_text.strip()

    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return f"ERROR: Can't invoke '{model_id}'. Reason: {e}"


def extract_normalized_title(response):
    """
    Extracts the normalized job title from the Bedrock API response.
    
    Args:
        response (str): The response from the Bedrock API.
    
    Returns:
        str: The normalized job title.
    """
    if 'normalized_title:' in response:
        normalized_title = response.split('normalized_title:')[1].strip().strip('"')
        return normalized_title
    return response

# Example usage
if __name__ == "__main__":
    template = "Generate a normalized job title for the entity '{query}' based on the following context:\n{context}"
    entity = "Senior Software Engineer"
    context = ["Software Development", "Engineering", "Senior Level"]
    
    prompt = generate_prompt(template, entity, context)
    response = call_bedrock(prompt)
    normalized_title = extract_normalized_title(response)
    
    print(f"Normalized Title: {normalized_title}")
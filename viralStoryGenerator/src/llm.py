# viralStoryGenerator/llm.py

import requests
import json
import re
import time
from viralStoryGenerator.src.storyboard import generate_storyboard
from viralStoryGenerator.prompts.prompts import get_system_instructions, get_user_prompt, get_fix_prompt
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as appconfig

# Precompiled regex patterns for efficiency
STORY_PATTERN = re.compile(r"(?s)### Story Script:\s*(.*?)\n### Video Description:")
DESC_PATTERN = re.compile(r"### Video Description:\s*(.*)$")
THINK_PATTERN = re.compile(r'(<think>.*?</think>)', re.DOTALL)

def _reformat_text(raw_text, endpoint, model, temperature):
    """
    Makes a second LLM call to reformat 'raw_text' if the first attempt was off-format.
    """
    fix_prompt = get_fix_prompt(raw_text)
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": fix_prompt.strip()}],
        "temperature": temperature,
        "max_tokens": appconfig.llm.MAX_TOKENS,
        "stream": False # To be implemented in the future
    }
    try:
        _logger.debug("Reformatting text using LLM...")
        response = requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=appconfig.httpOptions.TIMEOUT) #timeout to be a variable

        # if deevelopment, dump result to console
        if appconfig.ENVIRONMENT == "development":
            _logger.debug(f"Reformatting response: {response.text}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        _logger.error(f"An error occurred while calling the LLM for reformatting: {e}")
        return raw_text
    response_json = response.json()

    try:
        response_json = response.json()
    except json.JSONDecodeError as e:
        _logger.error(f"JSON decode error during reformatting: {e}")
        return raw_text

    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        _logger.error(f"Unexpected response structure: {e}")
        return raw_text

    return content

def _check_format(completion_text):
    """
    Validates that the text has the expected sections:
      ### Story Script:
      ### Video Description:
    Returns (story, description) if valid, else (None, None).
    """
    story_match = STORY_PATTERN.search(completion_text)
    desc_match = DESC_PATTERN.search(completion_text)

    if story_match and desc_match:
        story = story_match.group(1).strip()
        description = desc_match.group(1).strip()
        return story, description
    return None, None

def _extract_chain_of_thought(text):
    """
    Extracts chain-of-thought from text using THINK_PATTERN.
    Returns a tuple (clean_text, chain_of_thought).
    """
    match = THINK_PATTERN.search(text)
    chain = ""
    if match:
        chain = match.group(1)
        text = text.replace(chain, "").strip()
    return text, chain

def process_with_llm(topic: str, cleansed_content: str, temperature: float) -> str:
    """
    Process the given topic and cleansed content using the LLM to generate a story script.

    Args:
        topic (str): The topic for the story.
        cleansed_content (str): The preprocessed content to use for story generation.
        temperature (float): The temperature setting for the LLM.

    Returns:
        str: The generated story script.
    """
    _logger.debug(f"Processing with LLM for topic: {topic}")

    # Prepare the payload for the LLM API
    data = {
        "model": appconfig.llm.MODEL,
        "messages": [
            {"role": "system", "content": get_system_instructions()},
            {"role": "user", "content": get_user_prompt(topic, cleansed_content)}
        ],
        "temperature": temperature,
        "max_tokens": appconfig.llm.MAX_TOKENS,
        "stream": False
    }

    try:
        response = requests.post(appconfig.llm.ENDPOINT, json=data, timeout=appconfig.httpOptions.TIMEOUT)
        response.raise_for_status()
        response_json = response.json()
        story_script = response_json["choices"][0]["message"]["content"]
        _logger.debug(f"LLM response received for topic: {topic}")
        return story_script
    except requests.exceptions.RequestException as e:
        _logger.error(f"Error during LLM API call: {e}")
        raise
    except (KeyError, IndexError) as e:
        _logger.error(f"Unexpected response structure from LLM: {e}")
        raise

def generate_story_script(topic: str,
                          sources: str,
                          endpoint: str,
                          model: str,
                          temperature=float,
                          show_thinking: bool = False) -> dict:
    """
    Sends a request to a locally hosted LLM to generate the story script and video description.
    Attempts to ensure the output meets the format:
        ### Story Script:
        ### Video Description:
    """
    system_instructions = get_system_instructions()
    user_prompt = get_user_prompt(topic, sources).strip()

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": appconfig.llm.MAX_TOKENS,
        "stream": False
    }

    start_time = time.time()
    try:
        response = requests.post(endpoint, json=data, timeout=appconfig.httpOptions.TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        _logger.error(f"Error calling the LLM: {e}")
        return {
            "story": "",
            "video_description": "",
            "thinking": "",
            "generation_time": 0,
            "usage": {}
        }
    generation_time = time.time() - start_time

    try:
        response_json = response.json()
    except json.JSONDecodeError as e:
        _logger.error(f"Error decoding JSON response: {e}")
        return {
            "story": "",
            "video_description": "",
            "thinking": "",
            "generation_time": generation_time,
            "usage": {}
        }

    usage_info = response_json.get("usage", {})
    try:
        completion_text = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        _logger.error(f"Unexpected response structure: {e}")
        return {
            "story": "",
            "video_description": "",
            "thinking": "",
            "generation_time": generation_time,
            "usage": usage_info
        }

    # Check if the thinking part is present in the response
    if show_thinking:
        _logger.debug("Thinking is enabled; extracting chain-of-thought...")
        # if reasoning_content is found in response, extract it
        if response_json["choices"][0]["message"].get("reasoning_content") and response_json["choices"][0]["message"].get("reasoning_content") != "":
            thinking = response_json["choices"][0]["message"]["reasoning_content"]
        else:
            completion_text, thinking = _extract_chain_of_thought(completion_text)
    else:
        thinking = ""

    # Extract chain-of-thought if present

    # Check if format is correct
    story, description = _check_format(completion_text)

    # Generate the storyboard
    if story and story.strip():
        try:
            _logger.info("Generating storyboard based on the story script...")

            # storyboard = generate_storyboard(
            #     story=story,
            #     topic=topic,
            #     llm_endpoint=endpoint,
            #     model=model,
            #     temperature=temperature,
            #     voice_id=appconfig.elevenLabs.VOICE_ID
            # )
            storyboard = None

            _logger.info("Storyboard generation successful.")

        except Exception as e:
            _logger.error(f"Storyboard generation failed: {e}")
            return {
                "story": "",
                "video_description": "",
                "thinking": "",
                "generation_time": 0,
                "usage": {},
                "storyboard": ""
            }

    if story is None or description is None:
        fixed_text = _reformat_text(completion_text, endpoint, model, temperature)
        match = re.search(r'(<think>.*?</think>)', fixed_text, re.DOTALL)
        if match and not thinking:
            thinking = match.group(1)
            fixed_text = fixed_text.replace(match.group(1), "").strip()
        story, description = _check_format(fixed_text)
        if story is None or description is None:
            return {
                "story": completion_text,
                "video_description": "",
                "thinking": thinking,
                "generation_time": generation_time,
                "usage": usage_info,
                "storyboard": ""
            }
        else:
            completion_text = fixed_text
    return {
        "story": story,
        "video_description": description,
        "thinking": thinking,
        "generation_time": generation_time,
        "usage": usage_info,
        "storyboard": storyboard
    }
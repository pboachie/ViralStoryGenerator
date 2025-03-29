# viralStoryGenerator/main.py

import argparse
import requests
import json
import re
import time
import logging
from viralStoryGenerator.prompts.prompts import get_system_instructions, get_user_prompt, get_fix_prompt
from viralStoryGenerator.src.cli import cli_main

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Precompiled regex patterns for efficiency
THINK_PATTERN = re.compile(r'(<think>.*?</think>)', re.DOTALL)
STORY_PATTERN = re.compile(r"(?s)### Story Script:\s*(.*?)\n### Video Description:")
DESC_PATTERN = re.compile(r"### Video Description:\s*(.*)$")

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

def _reformat_text(raw_text, endpoint, model, temperature=0.7):
    """
    Makes a second LLM call to reformat 'raw_text' if the first attempt was off-format.
    """
    fix_prompt = get_fix_prompt(raw_text)
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": fix_prompt.strip()},
        ],
        "temperature": temperature,
        "max_tokens": 8192,
        "stream": False
    }

    try:
        logging.info("Reformatting text using LLM...")
        response = requests.post(endpoint, json=data, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while calling the LLM for reformatting: {e}")
        return raw_text  # fallback: just return the original

    try:
        response_json = response.json()
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error during reformatting: {e}")
        return raw_text

    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logging.error(f"Unexpected response structure: {e}")
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

def generate_story_script(topic,
                          sources,
                          endpoint="http://192.168.1.190:1234/v1/chat/completions",
                          model="deepseek-r1-distill-qwen-14b@q4_k_m",
                          temperature=0.7,
                          show_thinking=False):
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
        "max_tokens": 8192,
        "stream": False
    }

    start_time = time.time()
    try:
        response = requests.post(endpoint, json=data, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling the LLM: {e}")
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
        logging.error(f"Error decoding JSON response: {e}")
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
        logging.error(f"Unexpected response structure: {e}")
        return {
            "story": "",
            "video_description": "",
            "thinking": "",
            "generation_time": generation_time,
            "usage": usage_info
        }

    # Extract chain-of-thought if present
    completion_text, thinking = _extract_chain_of_thought(completion_text)

    # Check if format is correct
    story, description = _check_format(completion_text)
    if story is None or description is None:
        logging.info("Initial completion was off-format. Attempting reformatting...")
        fixed_text = _reformat_text(completion_text, endpoint, model, temperature)
        fixed_text, extra_thinking = _extract_chain_of_thought(fixed_text)
        if not thinking and extra_thinking:
            thinking = extra_thinking
        story, description = _check_format(fixed_text)
        if story is None or description is None:
            logging.warning("Reformatting did not produce the expected format; returning raw output.")
            return {
                "story": completion_text,
                "video_description": "",
                "thinking": thinking,
                "generation_time": generation_time,
                "usage": usage_info
            }
        else:
            completion_text = fixed_text

    return {
        "story": story,
        "video_description": description,
        "thinking": thinking,
        "generation_time": generation_time,
        "usage": usage_info
    }


if __name__ == "__main__":
    cli_main()

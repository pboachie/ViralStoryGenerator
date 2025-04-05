# viralStoryGenerator/llm.py

import requests
import json
import re
import time
from typing import Tuple, Dict, Optional

from viralStoryGenerator.prompts.prompts import get_system_instructions, get_user_prompt, get_fix_prompt
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils import config

appconfig = config.config

STORY_PATTERN = re.compile(r"(?s)### Story Script:\s*(.*?)\n### Video Description:")
DESC_PATTERN = re.compile(r"### Video Description:\s*(.*)$")
THINK_PATTERN = re.compile(r'(<think>.*?</think>)', re.DOTALL)

# Define a user agent for HTTP requests
APP_USER_AGENT = f"{appconfig.APP_TITLE}/{appconfig.VERSION}"

def _make_llm_request(endpoint: str, model: str, messages: list, temperature: float, max_tokens: int) -> requests.Response:
    """Helper function to make the actual HTTP request to the LLM."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": APP_USER_AGENT
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(data),
            timeout=appconfig.httpOptions.TIMEOUT
        )
        response.raise_for_status()
        return response
    except requests.exceptions.Timeout:
        _logger.error(f"LLM request timed out after {appconfig.httpOptions.TIMEOUT} seconds to {endpoint}.")
        raise
    except requests.exceptions.ConnectionError as e:
         _logger.error(f"LLM connection error to {endpoint}: {e}")
         raise
    except requests.exceptions.HTTPError as e:
         _logger.error(f"LLM HTTP error: {e.status_code} {e.response.text[:200]}...")
         raise
    except requests.exceptions.RequestException as e:
        _logger.error(f"An unexpected error occurred during the LLM request: {e}")
        raise


def _reformat_text(raw_text: str, endpoint: str, model: str, temperature: float) -> Optional[str]:
    """
    Makes a second LLM call to reformat 'raw_text' if the first attempt was off-format.
    Returns the reformatted text or None if reformatting fails.
    """
    _logger.warning("Attempting to reformat LLM output due to incorrect initial format.")
    fix_prompt = get_fix_prompt(raw_text).strip()
    messages = [{"role": "user", "content": fix_prompt}]

    try:
        response = _make_llm_request(endpoint, model, messages, temperature, appconfig.llm.MAX_TOKENS)
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        _logger.info("LLM output successfully reformatted.")
        return content
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        _logger.error(f"Failed to reformat LLM output: {e}")
        return None


def _check_format(completion_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validates that the text has the expected sections using compiled regex.
    Returns (story, description) if valid, else (None, None).
    """
    story_match = STORY_PATTERN.search(completion_text)
    desc_match = DESC_PATTERN.search(completion_text)

    if story_match and desc_match:
        story = story_match.group(1).strip()
        description = desc_match.group(1).strip()
        if story and description:
            return story, description
    _logger.debug("LLM output format check failed.")
    return None, None


def _extract_chain_of_thought(text: str) -> Tuple[str, str]:
    """
    Extracts chain-of-thought block (<think>...</think>) from text.
    Returns a tuple (clean_text, chain_of_thought).
    """
    match = THINK_PATTERN.search(text)
    chain = ""
    clean_text = text
    if match:
        chain = match.group(1)
        # Remove the matched block and surrounding whitespace
        clean_text = THINK_PATTERN.sub('', text).strip()
        _logger.debug("Extracted chain-of-thought block.")
    return clean_text, chain


def process_with_llm(topic: str, cleansed_content: str, temperature: float) -> str:
    """
    Process the given topic and cleansed content using the LLM to generate a story script.
    This is used by the api_worker.

    Args:
        topic: The topic for the story.
        cleansed_content: The preprocessed content to use for story generation.
        temperature: The temperature setting for the LLM.

    Returns:
        The generated story script (raw output from LLM).

    Raises:
        requests.exceptions.RequestException: If the LLM request fails.
        ValueError: If the LLM response is malformed.
    """
    _logger.debug(f"Processing with LLM for topic: {topic}")

    # Basic input validation (more complex validation might be needed depending on LLM)
    if not topic:
        _logger.error("LLM processing request failed: Topic cannot be empty.")
        raise ValueError("Topic cannot be empty")
    if not appconfig.llm.MODEL:
         _logger.error("LLM processing request failed: LLM_MODEL is not configured.")
         raise ValueError("LLM Model not configured")
    if not appconfig.llm.ENDPOINT:
         _logger.error("LLM processing request failed: LLM_ENDPOINT is not configured.")
         raise ValueError("LLM Endpoint not configured")


    # Consider input length limit based on model (e.g., estimate token count)
    # Simplified check: Warn if content is very long
    MAX_INPUT_LEN_APPROX = 131072 # Very rough character limit guess
    if len(cleansed_content) > MAX_INPUT_LEN_APPROX:
        _logger.warning(f"Input content length ({len(cleansed_content)} chars) is large, may exceed LLM context window.")
        cleansed_content = cleansed_content[:MAX_INPUT_LEN_APPROX]

    # Prepare messages for the LLM API
    system_prompt = get_system_instructions()
    user_prompt = get_user_prompt(topic, cleansed_content)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = _make_llm_request(
            appconfig.llm.ENDPOINT,
            appconfig.llm.MODEL,
            messages,
            temperature,
            appconfig.llm.MAX_TOKENS
        )
        response_json = response.json()

        # Extract content safely
        story_script = response_json["choices"][0]["message"]["content"]
        _logger.debug(f"LLM response received successfully for topic: {topic}")
        return story_script

    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        _logger.error(f"LLM API call failed for topic '{topic}': {e}")
        raise
    except (KeyError, IndexError) as e:
        _logger.error(f"Unexpected response structure from LLM for topic '{topic}': {e}. Response: {response_json}")
        raise ValueError("Malformed response from LLM") from e

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
                          temperature: float,
                          show_thinking: bool = False) -> dict:
    """
    Sends a request to an LLM to generate the story script and video description.
    Attempts to ensure the output meets the specified format and extracts metadata.
    This version is used by the CLI/direct calls.

    Args:
        topic: The topic for the story.
        sources: Cleansed source material.
        endpoint: LLM API endpoint URL.
        model: Name of the LLM model.
        temperature: Sampling temperature (0.0-1.0).
        show_thinking: Whether to extract and include the <think> block.

    Returns:
        A dictionary containing:
        - story: Extracted story script (str).
        - video_description: Extracted video description (str).
        - thinking: Extracted chain-of-thought (str, empty if not found/disabled).
        - generation_time: Request duration in seconds (float).
        - usage: Token usage dictionary from LLM response (dict).
        - storyboard: Generated storyboard structure (dict, or None if failed - Placeholder).
        - raw_output: Original full completion text from LLM (str).
    """
    _logger.info(f"Generating story script for topic: '{topic}' using model '{model}'")

    # Prepare messages
    system_instructions = get_system_instructions()
    user_prompt = get_user_prompt(topic, sources).strip()
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_prompt}
    ]

    # Initialize result structure
    result = {
        "story": "",
        "video_description": "",
        "thinking": "",
        "generation_time": 0.0,
        "usage": {},
        "storyboard": None,
        "raw_output": ""
    }

    try:
        response = _make_llm_request(endpoint, model, messages, temperature, appconfig.llm.MAX_TOKENS)
        result["generation_time"] = time.time() - start_time
        response_json = response.json()
        result["usage"] = response_json.get("usage", {})
        completion_text = response_json["choices"][0]["message"]["content"]
        result["raw_output"] = completion_text

    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        result["generation_time"] = time.time() - start_time # Record time even on failure
        _logger.error(f"LLM request failed for story generation: {e}")
        return result


    # Extract thinking block if enabled
    thinking = ""
    clean_completion_text = completion_text
    if show_thinking:
        _logger.debug("Thinking is enabled; extracting chain-of-thought...")
        # if reasoning_content is found in response, extract it
        if response_json["choices"][0]["message"]["reasoning_content"] and response_json["choices"][0]["message"]["reasoning_content"] != "":
            thinking = response_json["choices"][0]["message"]["reasoning_content"]
        else:
            # Fallback to regex extraction
            clean_completion_text, thinking = _extract_chain_of_thought(completion_text)
    result["thinking"] = thinking

    # Check format of the (potentially cleaned) text
    story, description = _check_format(clean_completion_text)

    # Generate the storyboard
    if story.strip():
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
        _logger.warning("Initial LLM output format incorrect, attempting reformat...")
        fixed_text = _reformat_text(clean_completion_text, endpoint, model, temperature)

        if fixed_text:
            # Re-check format and potentially re-extract thinking
            clean_fixed_text = fixed_text
            if show_thinking and not thinking: # Extract thinking only if not already found
                clean_fixed_text, thinking = _extract_chain_of_thought(fixed_text)
                result["thinking"] = thinking # Update thinking if found in reformatted text

            story, description = _check_format(clean_fixed_text)
            if story and description:
                 _logger.info("Successfully reformatted LLM output.")
                 result["raw_output"] = fixed_text # Update raw output to the reformatted version
            else:
                _logger.error("Reformatting failed to produce the correct structure.")
                # Keep the original (cleaned) completion text in the story field as fallback
                story = clean_completion_text
                description = ""
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

    # if story is None or description is None:
    #     _logger.info("Initial completion was off-format. Attempting reformatting...")
    #     fixed_text = _reformat_text(completion_text, endpoint, model, temperature)
    #     fixed_text, extra_thinking = _extract_chain_of_thought(fixed_text)
    #     if not thinking and extra_thinking:
    #         thinking = extra_thinking
    #     story, description = _check_format(fixed_text)
    #     if story is None or description is None:
    #         _logger.warning("Reformatting did not produce the expected format; returning raw output.")
    #         return {
    #             "story": completion_text,
    #             "video_description": "",
    #             "thinking": thinking,
    #             "generation_time": generation_time,
    #             "usage": usage_info
    #         }
    #     else:
    #         completion_text = fixed_text

    # return {
    #     "story": story,
    #     "video_description": description,
    #     "thinking": thinking,
    #     "generation_time": generation_time,
    #     "usage": usage_info
    # }
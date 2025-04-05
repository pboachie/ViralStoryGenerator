# viralStoryGenerator/llm.py

import requests
import json
import re
import time
from typing import Tuple, Dict, Optional, List

from viralStoryGenerator.prompts.prompts import get_system_instructions, get_user_prompt, get_fix_prompt
from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as appconfig


STORY_PATTERN = re.compile(r"(?s)### Story Script:\s*(.*?)\n### Video Description:")
DESC_PATTERN = re.compile(r"### Video Description:\s*(.*)$")
THINK_PATTERN = re.compile(r'(<think>.*?</think>)', re.DOTALL)

# Define a user agent for HTTP requests
APP_USER_AGENT = f"{appconfig.APP_TITLE}/{appconfig.VERSION}"

def _make_llm_request(endpoint: str, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> requests.Response:
    """Helper function to make the actual HTTP request to the LLM."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": APP_USER_AGENT
    }
    effective_max_tokens = min(max_tokens, 8192) # Cap at 8192, adjust if needed
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
        "stream": False
    }
    _logger.debug(f"Sending request to LLM: {endpoint}, Model: {model}, Temp: {temperature}, MaxTokens: {effective_max_tokens}")
    response = requests.post(
        endpoint,
        headers=headers,
        json=data,
        timeout=appconfig.httpOptions.TIMEOUT
    )
    response.raise_for_status()
    return response

def _reformat_text(raw_text: str, endpoint: str, model: str, temperature: float) -> Optional[str]:
    """Attempts to reformat LLM output if it doesn't match the expected structure."""
    _logger.warning("Attempting to reformat LLM output.")
    messages = [
        {"role": "system", "content": get_system_instructions()},
        {"role": "user", "content": get_fix_prompt(raw_text)}
    ]
    try:
        response = _make_llm_request(endpoint, model, messages, temperature, appconfig.llm.MAX_TOKENS)
        response_json = response.json()
        fixed_text = response_json["choices"][0]["message"]["content"]
        return fixed_text
    except Exception as e:
        _logger.error(f"Reformatting request failed: {e}")
        return None

def _check_format(completion_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Checks if the text contains the required sections."""
    story_match = STORY_PATTERN.search(completion_text)
    desc_match = DESC_PATTERN.search(completion_text)
    story = story_match.group(1).strip() if story_match else None
    description = desc_match.group(1).strip() if desc_match else None
    return story, description

def _extract_chain_of_thought(text: str) -> Tuple[str, str]:
    """Extracts <think> block and returns cleaned text and thinking block."""
    think_match = THINK_PATTERN.search(text)
    if think_match:
        thinking = think_match.group(1)
        # Remove the thinking block from the text
        cleaned_text = THINK_PATTERN.sub('', text).strip()
        return cleaned_text, thinking
    return text, ""

def process_with_llm(topic: str, relevant_content: str, temperature: float) -> str:
    """
    Process the given topic and relevant content using the LLM to generate a story script.
    This version is used by the api_worker with RAG.

    Args:
        topic: The topic for the story.
        relevant_content: Relevant content snippets retrieved via RAG.
        temperature: The temperature setting for the LLM.

    Returns:
        The generated story script (including description, separated by markers).

    Raises:
        requests.exceptions.RequestException: If the LLM request fails.
        ValueError: If the LLM response is malformed or required config is missing.
    """
    _logger.debug(f"Processing with LLM for topic: {topic} using RAG content.")

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

    # Prepare messages for the LLM API using the RAG-aware prompt
    system_prompt = get_system_instructions()
    # Pass relevant_content to get_user_prompt
    user_prompt = get_user_prompt(topic, relevant_content)
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
        completion_text = response_json["choices"][0]["message"]["content"]
        _logger.debug(f"LLM response received successfully for topic: {topic}")

        # --- Format Checking and Potential Reformatting ---
        clean_completion_text, thinking = _extract_chain_of_thought(completion_text) # Extract thinking first
        story, description = _check_format(clean_completion_text)

        if story is None or description is None:
            _logger.warning("Initial LLM output format incorrect, attempting reformat...")
            fixed_text = _reformat_text(clean_completion_text, appconfig.llm.ENDPOINT, appconfig.llm.MODEL, temperature)
            if fixed_text:
                clean_fixed_text, _ = _extract_chain_of_thought(fixed_text) # Re-extract thinking if needed
                story_fixed, description_fixed = _check_format(clean_fixed_text)
                if story_fixed and description_fixed:
                    _logger.info("Successfully reformatted LLM output.")
                    # Return the reformatted text which includes both parts
                    return clean_fixed_text
                else:
                    _logger.error("Reformatting failed to produce the correct structure. Returning original cleaned output.")
                    return clean_completion_text # Fallback to original cleaned text
            else:
                _logger.error("Reformatting request failed. Returning original cleaned output.")
                return clean_completion_text # Fallback to original cleaned text
        else:
            return clean_completion_text # Return text without thinking block

    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        _logger.error(f"LLM API call failed for topic '{topic}': {e}", exc_info=True)
        raise
    except (KeyError, IndexError) as e:
        _logger.error(f"Unexpected response structure from LLM for topic '{topic}': {e}. Response: {response_json if 'response_json' in locals() else 'N/A'}")
        raise ValueError("Malformed response from LLM") from e
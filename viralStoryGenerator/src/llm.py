# viralStoryGenerator/llm.py

import requests
import json
import re
import time
from typing import Tuple, Dict, Optional, List, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, wait_fixed

from viralStoryGenerator.prompts.prompts import get_system_instructions, get_user_prompt, get_fix_prompt, get_clean_markdown_prompt
import logging
from viralStoryGenerator.utils.config import config as appconfig

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

STORY_PATTERN = re.compile(r"(?s)### Story Script:\s*(.*?)\n### Video Description:")
DESC_PATTERN = re.compile(r"### Video Description:\s*(.*)$")
THINK_PATTERN = re.compile(r'(<think>.*?</think>)', re.DOTALL)
APP_USER_AGENT = f"{appconfig.APP_TITLE}/{appconfig.VERSION}"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10), # Wait 1s, 2s, 4s... up to 10s
    retry=retry_if_exception_type(
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError)
    ),
    before_sleep=lambda retry_state: _logger.warning(f"Retrying LLM request (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")
)
def _make_llm_request(endpoint: str, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, timeout: int) -> requests.Response:
    """
    Helper function to make the actual HTTP request to the LLM with retry logic.

    Args:
        endpoint: The API endpoint URL.
        model: The LLM model name.
        messages: The list of messages for the conversation.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        The requests.Response object on success.

    Raises:
        requests.exceptions.RequestException: If the request fails after retries.
        requests.exceptions.HTTPError: Specifically for non-2xx responses after retries.
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": APP_USER_AGENT
        # "Authorization": f"Bearer {appconfig.llm.API_KEY}"
    }
    max_output_tokens = getattr(appconfig.llm, 'MAX_OUTPUT_TOKENS', 32768)
    effective_max_tokens = min(max_tokens, max_output_tokens)
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
        "stream": False
        # todo: add provider-specific parameters if needed (e.g., top_p)
    }
    _logger.debug(f"Sending request to LLM: {endpoint}, Model: {model}, Temp: {temperature}, MaxTokens: {effective_max_tokens}, Timeout: {timeout}")
    try:
        _logger.debug(f"LLM Request Metadata: {len(messages)} messages, roles: {[m['role'] for m in messages]}")
        # _logger.debug(f"LLM Request Payload: {json.dumps(data, indent=2)}")
    except Exception as log_err:
        _logger.error(f"Error preparing log message: {log_err}")

    response = requests.post(
        endpoint,
        headers=headers,
        json=data,
        timeout=timeout
    )
    response.raise_for_status()
    _logger.debug(f"LLM request successful (Status: {response.status_code})")
    return response

@retry(wait=wait_fixed(1), stop=stop_after_attempt(2), retry=retry_if_exception_type(requests.exceptions.RequestException), after=lambda retry_state: _logger.warning(f"Retrying LLM request (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}"))
def _make_llm_request(
    messages: List[Dict[str, str]],
    model_name: str,
    temperature: float,
    max_tokens: int,
    llm_api_endpoint: str,
    stream_response: bool = False,
    timeout: int = 120
) -> requests.Response:
    _logger.debug(f"Original model name received: {model_name}")

    if not isinstance(llm_api_endpoint, str):
        _logger.warning(f"llm_api_endpoint was not a string (type: {type(llm_api_endpoint)}, value: {llm_api_endpoint}). Converting to string.")
        llm_api_endpoint = str(llm_api_endpoint)

    cleaned_model_name = model_name
    if isinstance(model_name, str) and "#" in model_name:
        cleaned_model_name = model_name.split("#")[0].strip()
        _logger.debug(f"Cleaned model name to: {cleaned_model_name}")

    effective_llm_api_endpoint = llm_api_endpoint
    if not llm_api_endpoint.endswith("/v1/chat/completions") and llm_api_endpoint.endswith("/"):
        effective_llm_api_endpoint = llm_api_endpoint.rstrip("/")

    _logger.debug(f"Sending request to LLM: {effective_llm_api_endpoint}, Model: {cleaned_model_name}, Temp: {temperature}, MaxTokens: {max_tokens}, Timeout: {timeout}")

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": cleaned_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream_response
    }
    try:
        response = requests.post(
            effective_llm_api_endpoint,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        _logger.debug(f"LLM request successful (Status: {response.status_code})")
        return response
    except requests.exceptions.RequestException as e:
        _logger.error(f"LLM request to {effective_llm_api_endpoint} failed: {e}", exc_info=True)
        raise

def _pre_process_markdown(markdown: str) -> str:
    """
    Applies deterministic pre-processing rules to remove obvious clutter,
    focusing on extracting the main article content from news-like sources.
    """
    _logger.debug("Applying improved markdown pre-processing rules...")
    processed = markdown

    # 1. Remove all image tags and their immediate captions/credits on the same line
    # Handles: ![alt](url) Caption Credit/Agency
    # Handles: ![alt](data:image/...)
    # Also removes lines that are *just* image URLs, which can happen after bad conversions
    processed = re.sub(r'^\s*https?://(?:media\.com|i\.ytimg\.com).*?(?:\.jpg|\.png|\.gif|\.webp)?\s*$', '', processed, flags=re.MULTILINE | re.IGNORECASE)
    processed = re.sub(r'!\[.*?\]\((?:data:image|https?://)[^)]*?\)\s*(?:[^\n]*)', '', processed)
    # Handles: [image: placeholder text]
    processed = re.sub(r'\[image:\s*.*?\]', '', processed, flags=re.IGNORECASE)


    # 2. Remove empty links and links to '#'
    processed = re.sub(r'\[\s*\]\(\s*\)\s*', '', processed)
    processed = re.sub(r'\[[^]]*?\]\(#\)\s*', '', processed) # Links to '#'
    processed = re.sub(r'\[\s*([^]]*?)\s*\]\((https?://[^)]*?\s*)\)\s*', r'\1 \2', processed) # Keep text and URL if link is alone


    # 3. Remove potential leftover script/style tags and HTML comments
    processed = re.sub(r'<script.*?</script>', '', processed, flags=re.DOTALL | re.IGNORECASE)
    processed = re.sub(r'<style.*?</style>', '', processed, flags=re.DOTALL | re.IGNORECASE)
    processed = re.sub(r'<!--.*?-->', '', processed, flags=re.DOTALL)

    # 4. Normalize whitespace and remove excessive newlines early
    lines = processed.splitlines()
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in lines]
    # Filter out empty lines created by previous steps
    lines = [line for line in lines if line]
    processed = "\n".join(lines)
    # Remove excessive newlines (more than 2 consecutive)
    processed = re.sub(r'\n{3,}', '\n\n', processed)


    # 5. Remove specific news/blog clutter patterns (line-based removal)
    clutter_line_patterns = [
        # Timestamps and relative times (e.g., "22 hr 16 min ago", "9:48 p.m. PDT, May 6, 2025")
        re.compile(r'^\s*(?:\d{1,2}\s+(?:hr|hrs|min|mins|sec|secs)\s+ago)\s*$', re.IGNORECASE),
        re.compile(r'^\s*(?:\d{1,2}:\d{2}\s*(?:[ap]\.?m\.?)?\s*(?:[A-Z]{2,4})?(?:,\s+\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?)?)\s*$', re.IGNORECASE), # PDT, EST etc.
        # "Link Copied!"
        re.compile(r'^\s*Link Copied!\s*$', re.IGNORECASE),
        # Standalone image credits (e.g., "Punit Paranjpe/AFP/Getty Images", "Manish Swarup/AP")
        # This one is tricky because it can match actual names if not specific enough.
        # Focus on patterns ending with known agencies or /File.
        re.compile(r'^\s*(?:[A-Z][A-Za-z\s.’-]+(?:,\s*[A-Z][A-Za-z\s.’-]+)*\s*/\s*)?(?:AFP|AP|Reuters|Getty Images|File|EPA|EFE|Shutterstock|Hindustan Times|Press Trust of India|ANI)(?:/(?:Getty Images|AP|AFP|Reuters))?\s*$', re.IGNORECASE),
        re.compile(r'^\s*[A-Z][a-z\s]+[A-Z][a-z]+(?:/[A-Z]+)?(?:/Getty Images)?\s*$', re.IGNORECASE), # Slightly more generic for "Name Name/Agency"
        re.compile(r'^\s*Image credit:.*$', re.IGNORECASE),
        re.compile(r'^\s*Photo by.*$', re.IGNORECASE),
        re.compile(r'^\s*Source:\s*(?:Reuters|AP|AFP|Getty Images|AP Photo|Bloomberg|X|FlightRadar24\.com)\s*$', re.IGNORECASE), # Line that is just "Source: XYZ"
        re.compile(r'^\s*\d{2}:\d{2}\s+-\s+Source:\s+\[.*?\]\(.*?\)\s*$', re.IGNORECASE), # For "02:09 - Source: [Test](...)"

        # Update notices
        re.compile(r'^\s*_(This post has been updated.*?)_\s*$', re.IGNORECASE),
        re.compile(r'^\s*Correction: This post has been updated.*?$', re.IGNORECASE),
        re.compile(r'^\s*Editor’s Note:.*$', re.IGNORECASE),
        re.compile(r'^\s*This story has been updated to reflect.*$', re.IGNORECASE),

        # Common boilerplate lines/phrases
        re.compile(r"^\s*advertisement\s*$", re.IGNORECASE),
        re.compile(r"^\s*share this(?: article| story| post)?\s*$", re.IGNORECASE),
        re.compile(r"^\s*(?:click to )?print\s*$", re.IGNORECASE),
        re.compile(r"^\s*(?:click to )?email\s*$", re.IGNORECASE),
        re.compile(r"^\s*related posts:?\s*$", re.IGNORECASE),
        re.compile(r"^\s*comments\s*$", re.IGNORECASE),
        re.compile(r"^\s*navigation menu\s*$", re.IGNORECASE),
        re.compile(r"^\s*skip to content\s*$", re.IGNORECASE),
        re.compile(r"^\s*log in\s*/\s*register\s*$", re.IGNORECASE),
        re.compile(r"^\s*follow us on.*", re.IGNORECASE),
        re.compile(r"^\s*subscribe to our newsletter.*", re.IGNORECASE),
        re.compile(r"^\s*posted in.*", re.IGNORECASE),
        re.compile(r"^\s*tags:.*", re.IGNORECASE),
        re.compile(r"^\s*leave a reply.*", re.IGNORECASE),
        re.compile(r"^\s*your email address will not be published.*", re.IGNORECASE),
        re.compile(r"^\s*required fields are marked.*", re.IGNORECASE),
        re.compile(r"^\s*Ad Feedback\s*$", re.IGNORECASE),
        re.compile(r"^\s*Subscribe\s*$", re.IGNORECASE),
        re.compile(r"^\s*Sign in\s*$", re.IGNORECASE),
        re.compile(r"^\s*My Account\s*$", re.IGNORECASE),

        # Orphaned image context (often bolded)
        re.compile(r'^\s*\*\*(?:Before|After) the strikes on .*?:\s*$', re.IGNORECASE),
        re.compile(r'^\s*\*\*(?:After) the strikes, taken on .*?:\s*$', re.IGNORECASE),
        re.compile(r'^\s*Photo:\s*.*$', re.IGNORECASE), # Lines starting with Photo:

        # "Here's what we know" type intros for summaries
        re.compile(r'^\s*_(?:Here’s|Here is) what we know(?: so far)?:?_\s*$', re.IGNORECASE),
        re.compile(r'^\s*_(?:Here’s|Here is) the latest(?: on what we know)?:?_\s*$', re.IGNORECASE),
        re.compile(r'^\s*_(?:Here’s|Here is) where things stand:?_\s*$', re.IGNORECASE),
        re.compile(r'^\s*_(?:Here’s|Here is) a quick recap:?_\s*$', re.IGNORECASE),
        re.compile(r'^\s*_(?:Here’s|Here is) what you need to know:?_\s*$', re.IGNORECASE),

        re.compile(r'^\s*\*\s+\[\s*(?:Settings|Newsletters|Topics you follow|Sign out)\s*\]\(.*?\)\s*$', re.IGNORECASE),
        re.compile(r'^\s*For privacy options, please see our privacy policy:.*?$', re.IGNORECASE),
        re.compile(r'^\s*### Cookie List\s*$', re.IGNORECASE),
        re.compile(r'^\s*Back Button\s*$', re.IGNORECASE), # From cookie consent
        re.compile(r'^\s*Search Icon\s*Filter Icon\s*Clear\s*$', re.IGNORECASE),
        re.compile(r'^\s*checkbox label label\s*$', re.IGNORECASE), # From cookie consent
        re.compile(r'^\s*Apply Cancel\s*$', re.IGNORECASE), # From cookie consent
        re.compile(r'^\s*Consent Leg\.Interest\s*$', re.IGNORECASE), # From cookie consent
        re.compile(r'^\s*Close\s*$', re.IGNORECASE), # From cookie consent
        re.compile(r'^\s*\[!\[Powered by Onetrust\]\(.*?\)\]\(https?://www.onetrust.com/products/cookie-consent/\)\s*$', re.IGNORECASE), # Onetrust logo
        re.compile(r'^\s*\[\s*\]\((?:https?://(?:facebook|twitter|instagram|tiktok|linkedin)\.com/.*?)\s*"Visit us on .*?"\)\s*$', re.IGNORECASE), # Social media icons

        # Lines that are just URLs (often remnants)
        re.compile(r'^\s*(https?://[^\s]+)\s*$', re.IGNORECASE), # If a line is ONLY a URL

        # Lines that are likely just noise from HTML conversion or short non-content markers
        re.compile(r"^\s*[-_*]{3,}\s*$"),  # Horizontal rules
        re.compile(r"^\s*•\s*$"), # Stray bullets
        # re.compile(r"^\s*([A-Z]{2,5})\s*$", re.IGNORECASE), # Very short ALL CAPS words - careful with this one
    ]

    # State variable for summary removal
    in_summary_list = False
    temp_cleaned_lines = []

    lines = processed.splitlines()
    for line_idx, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            if not temp_cleaned_lines or temp_cleaned_lines[-1] != "": # Avoid multiple blank lines
                 temp_cleaned_lines.append("")
            continue

        is_clutter = False
        for pattern in clutter_line_patterns:
            if pattern.fullmatch(line_stripped):
                is_clutter = True
                _logger.debug(f"Removing line due to pattern '{pattern.pattern}': {line_stripped[:100]}")
                # If this was a summary intro, set state for next lines
                if "_Here’s what we know" in pattern.pattern or "_Here is what we know" in pattern.pattern:
                    in_summary_list = True
                break
        if is_clutter:
            continue

        # Logic for removing list items if we are in a "Here's what we know" summary
        if in_summary_list:
            if line_stripped.startswith(("* ", "- ", "• ")) or re.match(r"^\d+\.\s+", line_stripped):
                # Check if this list item looks like a question (often valuable)
                if '?' in line_stripped and len(line_stripped) > 30:
                    temp_cleaned_lines.append(line) # Keep question-like list items
                    _logger.debug(f"Keeping summary list item (question): {line_stripped[:100]}")
                else:
                    _logger.debug(f"Removing summary list item: {line_stripped[:100]}")
                    continue # Skip this summary list item
            else:
                # No longer in a list format, or it's a heading, so assume summary ended
                in_summary_list = False
                if line_stripped.startswith("##"): # if it's a new heading, keep it and reset
                     temp_cleaned_lines.append(line)
                     continue

        temp_cleaned_lines.append(line)

    # Filter out multiple consecutive blank lines that might have been created
    cleaned_lines = []
    for i, l in enumerate(temp_cleaned_lines):
        if l == "" and i > 0 and temp_cleaned_lines[i-1] == "":
            continue
        cleaned_lines.append(l)

    processed = "\n".join(cleaned_lines)


    # 6. Normalize whitespace again and trim overall
    processed = re.sub(r'\n{3,}', '\n\n', processed) # Consolidate newlines
    lines = processed.splitlines()
    processed = "\n".join(line.strip() for line in lines if line.strip()) # Strip and remove empty lines
    processed = processed.strip()

    # 7. Remove any remaining list items that might be navigation if they are very short
    # This is more aggressive and might need adjustment - applied AFTER main cleaning
    lines = processed.splitlines()
    potentially_cleaned_lines = []
    for line in lines:
        is_short_nav_item = False
        # Matches lines like "* Item" or "- Item" or "1. Item" where Item is short
        if re.match(r"^\s*[-*]\s+[\w\s.&']{1,25}\s*$", line) or \
           re.match(r"^\s*\d+\.\s+[\w\s.&']{1,25}\s*$", line):
            # Check if it's a common section header that we want to keep, or contains a verb
            is_content_like = any(kw in line.lower() for kw in ['introduction', 'conclusion', 'summary', 'overview', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']) or re.search(r'\b(is|are|was|were|has|have|had|do|does|did|will|can|could|may|might|must|should|would)\b', line.lower())
            if not is_content_like and len(line.split()) < 5 : # if it has few words and no verb/keyword
                _logger.debug(f"Removing potential short list item/nav: {line[:100]}")
                is_short_nav_item = True

        if not is_short_nav_item:
            potentially_cleaned_lines.append(line)

    processed = "\n".join(potentially_cleaned_lines)
    processed = re.sub(r'\n{3,}', '\n\n', processed) # Clean newlines again
    processed = processed.strip()


    _logger.debug(f"Pre-processing complete. Original length {len(markdown)}, Processed length {len(processed)}")
    return processed

def _post_process_llm_output(llm_output: str) -> str:
    """
    Cleans the raw output from the LLM, removing common artifacts.
    """
    _logger.debug("Applying post-processing rules to LLM markdown output...")
    processed = llm_output.strip()

    # Remove common LLM preamble/apologies (case-insensitive start)
    common_preambles = [
        "Here is the cleaned markdown content:",
        "Here's the cleaned markdown content:",
        "Here is the cleaned markdown:",
        "Here's the cleaned markdown:",
        "Here is the cleaned text:",
        "Here's the cleaned text:",
        "Okay, here is the cleaned version:",
        "Okay, here's the cleaned version:",
        "Here is the extracted article content:",
        "Below is the cleaned markdown:",
        "The cleaned markdown is:",
        "Cleaned Markdown Output:", # Match the prompt's ending
        "```markdown",
        "```"
    ]
    common_preambles.sort(key=len, reverse=True)

    processed_lower = processed.lower()
    for preamble in common_preambles:
        if processed_lower.startswith(preamble.lower()):
            processed = processed[len(preamble):].lstrip(" \n")
            processed_lower = processed.lower()

    # Remove common LLM postamble/notes (case-insensitive end)
    common_postambles = [
        "I hope this helps!",
        "Let me know if you need further adjustments.",
        "Feel free to ask if you need more cleaning.",
        "This output contains only the core article content.",
        "```" # Remove trailing code block marker
    ]
    common_postambles.sort(key=len, reverse=True)

    processed_lower = processed.lower()
    for postamble in common_postambles:
         if processed_lower.endswith(postamble.lower()):
            processed = processed[:-len(postamble)].rstrip(" \n")
            processed_lower = processed.lower()

    processed = processed.strip()

    _logger.debug(f"Post-processing complete. Final length: {len(processed)}")
    return processed


def _generate_cleaning_prompt(markdown_to_clean: str, max_chars: int = 20000) -> str:
    """
    Generates a detailed and robust prompt for the LLM cleaning task.

    Args:
        markdown_to_clean: The markdown content (potentially pre-processed).
        max_chars: Approximate maximum characters to include in the prompt to avoid token limits.
                   Adjust based on typical input size and model context window.

    Returns:
        The formatted user prompt string.
    """
    # todo: This is a simple truncation. A better approach for very long docs might be chunking & merging.
    original_length = len(markdown_to_clean)
    if original_length > max_chars:
        _logger.warning(f"Input markdown length ({original_length}) exceeds max_chars ({max_chars}). Truncating.")
        truncated_markdown = markdown_to_clean[:max_chars]
        # Try to avoid cutting mid-word/sentence if possible
        last_space = truncated_markdown.rfind(' ')
        if last_space != -1:
            truncated_markdown = truncated_markdown[:last_space]
        markdown_to_clean = truncated_markdown + "\n... [Content Truncated Due To Length]"
        _logger.warning(f"Truncated markdown length: {len(markdown_to_clean)}")

    prompt = get_clean_markdown_prompt(markdown_to_clean)
    return prompt

def clean_markdown_with_llm(raw_markdown: str, temperature: float = 0.95) -> Optional[str]:
    """
    Uses pre-processing and an LLM to clean raw markdown scraped from the web.
    Aims to remove boilerplate, ads, navigation, etc., and format as a clean article.

    Args:
        raw_markdown: The raw markdown string to clean.
        temperature: The temperature setting for the LLM (lower for more deterministic cleaning).

    Returns:
        The cleaned markdown string, or None if cleaning fails after all attempts.

    Raises:
        ValueError: If required LLM configuration (MODEL, ENDPOINT) is missing.
    """
    _logger.info(f"Attempting to clean markdown content (initial length: {len(raw_markdown)}).")

    # 1. Basic Input Validation
    if not raw_markdown:
        _logger.warning("clean_markdown_with_llm called with empty input.")
        return ""
    if not appconfig.llm.MODEL:
         _logger.error("LLM cleaning request failed: LLM_MODEL is not configured.")
         raise ValueError("LLM Model not configured")
    if not appconfig.llm.ENDPOINT:
         _logger.error("LLM cleaning request failed: LLM_ENDPOINT is not configured.")
         raise ValueError("LLM Endpoint not configured")

    # 2. Pre-processing (Deterministic Cleaning)
    pre_processed_markdown = _pre_process_markdown(raw_markdown)
    _logger.debug(f"Pre-processed markdown length: {len(pre_processed_markdown)}")

    if not pre_processed_markdown.strip():
        _logger.info("Markdown content is empty after pre-processing. Skipping LLM cleaning.")
        return pre_processed_markdown

    if len(pre_processed_markdown) < appconfig.llm.MIN_MARKDOWN_LENGTH_FOR_LLM_CLEANING:
        _logger.info(f"Markdown content (length: {len(pre_processed_markdown)}) is shorter than minimum ({appconfig.llm.MIN_MARKDOWN_LENGTH_FOR_LLM_CLEANING}) for LLM cleaning. Returning pre-processed content.")
        return pre_processed_markdown

    # 3. Prepare LLM Request
    # todo: Some models work better with the instructions primarily in the *user* prompt.
    system_prompt = "You are an AI assistant specialized in extracting and cleaning web article and meaningful content from raw markdown."
    # Generate the detailed user prompt including the pre-processed markdown
    max_prompt_chars = appconfig.llm.CLEANING_MAX_PROMPT_CHARS
    user_prompt = _generate_cleaning_prompt(pre_processed_markdown, max_chars=max_prompt_chars)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Configuration for the cleaning task
    cleaning_temperature = temperature # Use the provided temperature
    # todo: Set based on expected *cleaned* article length, plus some buffer.
    max_tokens_for_cleaning = appconfig.llm.CLEANING_MAX_OUTPUT_TOKENS
    request_timeout = appconfig.httpOptions.TIMEOUT

    response_json: Optional[Dict[str, Any]] = None
    try:
        response = _make_llm_request(
            messages=messages,
            model_name=appconfig.llm.MODEL_SMALL,
            temperature=cleaning_temperature,
            max_tokens=max_tokens_for_cleaning,
            llm_api_endpoint=appconfig.llm.ENDPOINT,
            timeout=request_timeout
        )
        response_json = response.json()

        if not response_json or "choices" not in response_json or not response_json["choices"]:
             _logger.error(f"LLM response missing 'choices' array. Response: {response_json}")
             return None

        # Handle potential variations in response structure (e.g., message/content keys)
        first_choice = response_json["choices"][0]
        if "message" in first_choice and "content" in first_choice["message"]:
            raw_llm_output = first_choice["message"]["content"]
        elif "text" in first_choice:
             raw_llm_output = first_choice["text"]
        else:
            _logger.error(f"Could not find 'content' or 'text' in LLM response choice. Choice: {first_choice}")
            return None

        if not raw_llm_output:
             _logger.warning("LLM returned empty content for cleaning.")
             return ""

        # 5. Post-processing LLM Output
        cleaned_text = _post_process_llm_output(raw_llm_output)

        _logger.info(f"Successfully cleaned markdown. Original length: {len(raw_markdown)}, Pre-processed length: {len(pre_processed_markdown)}, Final length: {len(cleaned_text)}")
        return cleaned_text

    # Handle exceptions that tenacity didn't retry or final failure after retries
    except requests.exceptions.RequestException as e:
        _logger.error(f"LLM cleaning API call failed definitively after retries: {e}", exc_info=True)
        if hasattr(e, 'response') and e.response is not None:
            _logger.error(f"Failed Response Status: {e.response.status_code}, Body: {e.response.text[:500]}...") # Log snippet
        return None
    except json.JSONDecodeError as e:
        _logger.error(f"Failed to decode JSON response from LLM: {e}", exc_info=True)
        if 'response' in locals() and hasattr(response, 'text'):
             _logger.error(f"Raw LLM Response Text (snippet): {response.text[:500]}...")
        return None
    except (KeyError, IndexError) as e:
        _logger.error(f"Unexpected response structure from LLM during cleaning: {e}. Response JSON: {response_json}")
        return None
    except Exception as e:
        _logger.exception(f"An unexpected error occurred during LLM cleaning: {e}")
        return None



def _reformat_text(raw_text: str, endpoint: str, model: str, temperature: float) -> Optional[str]:
    """Attempts to reformat LLM output if it doesn't match the expected structure."""
    _logger.warning("Attempting to reformat LLM output for story/description.")
    messages = [
        {"role": "system", "content": get_system_instructions()},
        {"role": "user", "content": get_fix_prompt(raw_text)}
    ]
    try:
        response = _make_llm_request(
            messages=messages,
            model_name=model,
            temperature=temperature,
            max_tokens=appconfig.llm.MAX_TOKENS,
            llm_api_endpoint=endpoint,
            timeout=appconfig.httpOptions.TIMEOUT
        )
        response_json = response.json()

        if not response_json or "choices" not in response_json or not response_json["choices"]:
             _logger.error(f"Reformatting response missing 'choices' array. Response: {response_json}")
             return None
        first_choice = response_json["choices"][0]
        if "message" in first_choice and "content" in first_choice["message"]:
            fixed_text = first_choice["message"]["content"]
            return fixed_text
        else:
             _logger.error(f"Could not find 'content' in reformatting response choice. Choice: {first_choice}")
             return None

    except requests.exceptions.RequestException as e:
        _logger.error(f"LLM Reformatting API call failed: {e}", exc_info=True)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
         _logger.error(f"Error processing reformatting response: {e}. Response JSON: {response_json if 'response_json' in locals() else 'N/A'}")
         return None
    except Exception as e:
        _logger.exception(f"An unexpected error occurred during LLM reformatting: {e}")
        return None

def _check_format(completion_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Checks if the text contains the required sections. Tries to extract story and description even if markers are missing."""
    _logger.debug("Checking format of completion text for story/description markers.")
    story_match = STORY_PATTERN.search(completion_text)
    desc_match = DESC_PATTERN.search(completion_text)

    _logger.debug(f"Regex - Story match found: {bool(story_match)}, Description match found: {bool(desc_match)}")

    story = story_match.group(1).strip() if story_match else None
    description = desc_match.group(1).strip() if desc_match else None

    # Fallback logic if standard markers are missing
    if story is None or description is None:
        _logger.debug("Standard markers (### Story Script:, ### Video Description:) not found or incomplete. Attempting fallbacks.")
        if '### Video Description:' in completion_text:
             parts = completion_text.split('### Video Description:', 1)
             story_candidate = parts[0].replace('### Story Script:', '').strip()
             desc_candidate = parts[1].strip() if len(parts) > 1 else None
             if story is None: story = story_candidate
             if description is None: description = desc_candidate
             _logger.debug(f"Fallback using '### Video Description:': Story='{story[:50]}...', Desc='{description[:50]}...'")

        elif '[Description]' in completion_text:
            _logger.debug("Fallback using '[Description]' marker detected.")
            parts = completion_text.split('[Description]', 1)
            story_candidate = parts[0].replace('### Story Script:', '').strip()
            desc_candidate = parts[1].strip() if len(parts) > 1 else None
            if story is None: story = story_candidate
            if description is None: description = desc_candidate
            _logger.debug(f"Fallback using '[Description]': Story='{story[:50]}...', Desc='{description[:50]}...'")

        # Fallback to double newline split as a last resort
        elif '\n\n' in completion_text and (story is None or description is None):
            _logger.debug("Fallback using double newline split.")
            if story_match is None and desc_match is None:
                parts = completion_text.split('\n\n', 1)
                story_candidate = parts[0].strip()
                desc_candidate = parts[1].strip() if len(parts) > 1 else None
                if story is None: story = story_candidate
                if description is None: description = desc_candidate
                _logger.debug(f"Fallback using '\\n\\n': Story='{story[:50]}...', Desc='{description[:50]}...'")
            else:
                 _logger.debug("Skipping double newline fallback as standard markers were partially present.")

        if story is None and description is None:
            _logger.debug("No clear separators found, treating entire text as story.")
            story = completion_text.strip()
            description = None

    elif story is None and description is not None:
         _logger.warning("Description found but story is missing. This might indicate unexpected formatting.")
         desc_index = completion_text.find('### Video Description:')
         if desc_index > 0 :
             story = completion_text[:desc_index].replace('### Story Script:', '').strip()
             _logger.debug(f"Recovered story based on description position: Story='{story[:50]}...'")

    return story, description

def _extract_chain_of_thought(text: str) -> Tuple[str, str]:
    """Extracts <think> block and returns cleaned text and thinking block."""
    think_match = THINK_PATTERN.search(text)
    if think_match:
        thinking = think_match.group(1)
        # Remove the thinking block from the text
        cleaned_text = THINK_PATTERN.sub('', text).strip()
        _logger.debug("Extracted chain-of-thought block.")
        return cleaned_text, thinking
    _logger.debug("No chain-of-thought block found.")
    return text, ""

def process_with_llm(topic: str, temperature: float, model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Process the given topic and relevant content using the specified LLM model to generate a story script.
    This version is used by the api_worker with RAG.

    Args:
        topic: The topic for the story.
        temperature: The temperature setting for the LLM.
        model: The name of the LLM model to use.
        system_prompt: The system prompt for the LLM.
        user_prompt: The user prompt for the LLM.

    Returns:
        The generated story script (potentially including description, separated by markers).

    Raises:
        requests.exceptions.RequestException: If the LLM request fails after retries.
        ValueError: If the LLM response is malformed or required config is missing.
        Exception: For other unexpected errors during processing.
    """
    _logger.info(f"Processing with LLM model '{model}' for topic: '{topic}' using RAG content.")

    # Basic input validation
    if not topic:
        _logger.error("LLM processing request failed: Topic cannot be empty.")
        raise ValueError("Topic cannot be empty")
    if not model:
         _logger.error("LLM processing request failed: Model name cannot be empty.")
         raise ValueError("LLM Model name cannot be empty")
    if not appconfig.llm.ENDPOINT:
         _logger.error("LLM processing request failed: LLM_ENDPOINT is not configured.")
         raise ValueError("LLM Endpoint not configured")

    # Prepare messages for the LLM API using the RAG-aware prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response_json: Optional[Dict[str, Any]] = None
    try:
        response = _make_llm_request(
            messages=messages,
            model_name=model,
            temperature=temperature,
            max_tokens=appconfig.llm.MAX_TOKENS,
            llm_api_endpoint=appconfig.llm.ENDPOINT,
            timeout=appconfig.httpOptions.TIMEOUT
        )
        response_json = response.json()

        if not response_json or "choices" not in response_json or not response_json["choices"]:
             _logger.error(f"LLM response missing 'choices' array for topic '{topic}'. Response: {response_json}")
             raise ValueError("Malformed response from LLM (missing choices)")
        first_choice = response_json["choices"][0]
        if "message" in first_choice and "content" in first_choice["message"]:
            completion_text = first_choice["message"]["content"]
        else:
             _logger.error(f"Could not find 'content' in LLM response choice for topic '{topic}'. Choice: {first_choice}")
             raise ValueError("Malformed response from LLM (missing content)")

        if not completion_text:
            _logger.warning(f"LLM returned empty content for topic '{topic}'.")
            return ""

        _logger.debug(f"Raw LLM output for topic '{topic}' (length {len(completion_text)}): {completion_text[:150]}...")
        clean_completion_text, thinking = _extract_chain_of_thought(completion_text)
        if thinking:
             _logger.info(f"Extracted thinking block for topic '{topic}': {thinking[:100]}...")

        story, description = _check_format(clean_completion_text)

        if story is None or description is None:
            _logger.warning(f"Initial LLM output format incorrect for topic '{topic}'. Attempting reformat...")
            fixed_text = _reformat_text(clean_completion_text, appconfig.llm.ENDPOINT, appconfig.llm.MODEL_LARGE, temperature)

            if fixed_text:
                _logger.debug(f"Reformatted text received (length {len(fixed_text)}): {fixed_text[:150]}...")
                clean_fixed_text, fixed_thinking = _extract_chain_of_thought(fixed_text)
                if fixed_thinking:
                    _logger.info(f"Extracted thinking block from *reformatted* text: {fixed_thinking[:100]}...")

                story_fixed, description_fixed = _check_format(clean_fixed_text)

                if story_fixed and description_fixed:
                    _logger.info(f"Successfully reformatted LLM output for topic '{topic}'.")
                    return clean_fixed_text
                else:
                    _logger.error(f"Reformatting failed to produce the correct structure for topic '{topic}'. Returning original cleaned output.")
                    return clean_completion_text
            else:
                _logger.error(f"Reformatting request failed for topic '{topic}'. Returning original cleaned output.")
                return clean_completion_text
        else:
            _logger.info(f"LLM output format correct for topic '{topic}'. Story and Description found.")
            return clean_completion_text

    except requests.exceptions.RequestException as e:
        _logger.error(f"LLM API call failed definitively for topic '{topic}': {e}", exc_info=True)
        raise
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
        _logger.error(f"Error processing LLM response for topic '{topic}': {e}. Response JSON: {response_json}")
        raise ValueError(f"Malformed response or processing error for LLM: {e}") from e
    except Exception as e:
        _logger.exception(f"An unexpected error occurred during LLM processing for topic '{topic}': {e}")
        raise
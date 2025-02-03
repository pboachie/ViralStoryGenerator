# viralStoryGenerator/main.py

import argparse
import requests
import json
import re
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _reformat_text(raw_text, endpoint, model, temperature=0.7):
    """
    Makes a second LLM call to reformat 'raw_text' if the first attempt was off-format.
    """
    fix_prompt = f"""
You provided the following text, but it doesn't follow the required format:

{raw_text}

Reformat this text to exactly include two sections:
1) ### Story Script:
2) ### Video Description:

No additional text or sections.
The video description must be a single line with a maximum of 100 characters.
"""

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": fix_prompt.strip()},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": False
    }

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the LLM for reformatting: {e}")
        return raw_text  # fallback: just return the original

    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]

def _check_format(completion_text):
    """
    Validates that the text has the expected sections:
      ### Story Script:
      ### Video Description:
    Returns (story, description) if valid, else (None, None).
    """
    # Use regex to capture the sections
    story_pattern = r"(?s)### Story Script:\s*(.*?)\n### Video Description:"
    desc_pattern = r"### Video Description:\s*(.*)$"

    story_match = re.search(story_pattern, completion_text)
    desc_match = re.search(desc_pattern, completion_text)

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

    # System-level or top-level instructions (reinforce format)
    system_instructions = (
        "You are a helpful assistant that strictly follows formatting rules.\n\n"
        "Rules:\n"
        "1. Do NOT add extra commentary or disclaimers.\n"
        "2. Output MUST contain exactly two sections in this order:\n"
        "   - \"### Story Script:\" followed by the story\n"
        "   - \"### Video Description:\" followed by the description\n"
        "3. The video description must be a single line (<= 100 characters).\n"
    )

    user_prompt = f"""
Below are several sources and articles with notes about {topic}. Using the provided information,
please generate a short, engaging story script that is about 1.5 minutes long when narrated.
The script should be informal, conversational, and suitable for a casual video update.
Make sure to highlight the key points, include any 'spicy' or controversial details mentioned in
the notes, and explain why {topic} hasn't been working recently, while also weaving in any speculations
or rumors as appropriate.

Additionally, please generate a video description that is a maximum of 100 characters long.
The description should include creatively placed hashtags related to the subject of the story.

Here are the sources and notes:
{sources}

Now, please produce the narrated story script followed by the video description.
""".strip()

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            # System message:
            {"role": "system", "content": system_instructions},
            # User message:
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 4096,
        "stream": False
    }

    # Start timer for generation
    start_time = time.time()
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
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

    response_json = response.json()
    # Capture token usage if available
    usage_info = response_json.get("usage", {})

    completion_text = response_json["choices"][0]["message"]["content"]

    # Always try to extract chain-of-thought and remove it from final text
    thinking = ""
    match = re.search(r'(<think>.*?</think>)', completion_text, re.DOTALL)
    if match:
        thinking = match.group(1)
        # Always remove it from the final text
        completion_text = completion_text.replace(thinking, "").strip()

    # 2) Check if format is correct
    story, description = _check_format(completion_text)
    if story is None or description is None:
        logging.info("Initial completion was off-format. Attempting reformatting...")
        # Attempt to reformat
        fixed_text = _reformat_text(completion_text, endpoint, model, temperature)

        # Check again for chain-of-thought in the re-formatted text
        match = re.search(r'(<think>.*?</think>)', fixed_text, re.DOTALL)
        if match:
            # If we didn't capture anything before, store it now
            if not thinking:
                thinking = match.group(1)
            # Remove from final text
            fixed_text = fixed_text.replace(match.group(1), "").strip()

        story, description = _check_format(fixed_text)
        if story is None or description is None:
            # Return partially if still not formatted
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

from viralStoryGenerator.cli import cli_main

if __name__ == "__main__":
    cli_main()

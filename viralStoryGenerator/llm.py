# viralStoryGenerator/llm.py

import requests
import json
import re
import time
import logging
from viralStoryGenerator.storyboard import generate_storyboard
from viralStoryGenerator.prompts.prompts import get_system_instructions, get_user_prompt, get_fix_prompt

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _reformat_text(raw_text, endpoint, model, temperature=0.7):
    # Use get_fix_prompt from prompts module
    fix_prompt = get_fix_prompt(raw_text)
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": fix_prompt.strip()}],
        "temperature": temperature,
        "max_tokens": 8192,
        "stream": False
    }
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the LLM for reformatting: {e}")
        return raw_text
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]

def _check_format(completion_text):
    # ...existing _check_format code...
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
                          temperature=0.7):
    system_instructions = get_system_instructions()
    user_prompt = get_user_prompt(topic, sources)

    headers = {"Content-Type": "application/json"}
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
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling the LLM: {e}")
        return {
            "story": "",
            "video_description": "",
            "thinking": "",
            "generation_time": 0,
            "usage": {},
            "storyboard": ""
        }
    generation_time = time.time() - start_time
    response_json = response.json()
    usage_info = response_json.get("usage", {})
    completion_text = response_json["choices"][0]["message"]["content"]
    thinking = ""
    match = re.search(r'(<think>.*?</think>)', completion_text, re.DOTALL)
    if match:
        thinking = match.group(1)
        completion_text = completion_text.replace(thinking, "").strip()

    story, description = _check_format(completion_text)
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

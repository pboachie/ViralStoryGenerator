import pytest
from viralStoryGenerator.prompts import prompts

def test_get_system_instructions():
    instructions = prompts.get_system_instructions()
    assert isinstance(instructions, str)
    assert "Viral Content Specialist AI" in instructions
    assert "## Task" in instructions
    assert "## Output Format" in instructions

def test_get_user_prompt_with_relevant_chunks():
    topic = "Test Topic"
    relevant_chunks = "Chunk 1. Chunk 2."
    prompt = prompts.get_user_prompt(topic, relevant_chunks)
    assert isinstance(prompt, str)
    assert f"'{topic}'" in prompt
    assert "## Relevant Information Snippets:" in prompt
    assert relevant_chunks in prompt
    assert "Generate the required output now." in prompt

def test_get_user_prompt_without_relevant_chunks():
    topic = "Test Topic No Chunks"
    relevant_chunks = ""
    prompt = prompts.get_user_prompt(topic, relevant_chunks)
    assert isinstance(prompt, str)
    assert f"'{topic}'" in prompt
    assert "No specific context snippets were retrieved." in prompt
    assert "Generate the required output now." in prompt

def test_get_user_prompt_with_whitespace_chunks():
    topic = "Test Topic Whitespace"
    relevant_chunks = "   \n  "
    prompt = prompts.get_user_prompt(topic, relevant_chunks)
    assert isinstance(prompt, str)
    assert f"'{topic}'" in prompt
    assert "No specific context snippets were retrieved." in prompt
    assert "Generate the required output now." in prompt

def test_get_fix_prompt():
    raw_text = "This is some faulty output."
    prompt = prompts.get_fix_prompt(raw_text)
    assert isinstance(prompt, str)
    assert "## Error Analysis" in prompt
    assert "## Correction Instructions" in prompt
    assert raw_text in prompt
    assert "Generate the corrected version now." in prompt

def test_get_clean_markdown_prompt():
    raw_markdown = "# Title\nSome content with *markdown*."
    prompt = prompts.get_clean_markdown_prompt(raw_markdown)
    assert isinstance(prompt, str)
    assert "**CRUCIAL OVERARCHING PRINCIPLE: VERBATIM EXTRACTION, NOT GENERATION.**" in prompt
    assert "<RAW_MARKDOWN_START>" in prompt
    assert raw_markdown in prompt
    assert "<RAW_MARKDOWN_END>" in prompt
    assert "**Cleaned Markdown Output:**" in prompt
    assert "YOUR RESPONSE MUST CONSIST *SOLELY* AND *EXCLUSIVELY* OF THE CLEANED MARKDOWN TEXT" in prompt

def test_get_storyboard_prompt():
    story = "This is a test story for storyboarding. It has multiple sentences."
    prompt = prompts.get_storyboard_prompt(story)
    assert isinstance(prompt, str)
    assert "## Role" in prompt
    assert "Storyboard Planner" in prompt
    assert "## Task" in prompt
    assert "Output ONLY the valid JSON object" in prompt
    assert "## Input Story" in prompt
    assert story in prompt
    assert "Ensure every scene includes a valid 'scene_start_marker'." in prompt
    assert '"scene_start_marker": "Exact first 5-10 words of the scene..."' in prompt

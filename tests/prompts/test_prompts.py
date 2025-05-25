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
    relevant_chunks_str = "Chunk 1. Chunk 2."
    prompt = prompts.get_user_prompt(topic, relevant_chunks_str)
    assert isinstance(prompt, str)
    assert f"'{topic}'" in prompt
    assert "## Relevant Information Snippets:" in prompt
    assert relevant_chunks_str in prompt
    assert "Generate the required output now." in prompt

def test_get_user_prompt_without_relevant_chunks():
    topic = "Test Topic No Chunks"
    relevant_chunks_empty_str = "" 
    prompt = prompts.get_user_prompt(topic, relevant_chunks_empty_str)
    assert isinstance(prompt, str)
    assert f"'{topic}'" in prompt
    assert "No specific context snippets were retrieved." in prompt
    assert "Generate the required output now." in prompt

def test_get_user_prompt_with_none_chunks_as_input_type():
    # The function expects relevant_chunks: str.
    # If None is passed, the `all` check might behave unexpectedly if not handled.
    # The current code: `if not relevant_chunks or all(...)` handles None correctly due to `not relevant_chunks` being true.
    topic = "Test Topic None Chunks"
    relevant_chunks_none = None
    prompt = prompts.get_user_prompt(topic, relevant_chunks_none) # type: ignore 
    # Ignoring type error as we are testing this specific path.
    assert isinstance(prompt, str)
    assert f"'{topic}'" in prompt
    assert "No specific context snippets were retrieved." in prompt # This path should be taken
    assert "Generate the required output now." in prompt

def test_get_user_prompt_with_whitespace_chunks_string():
    topic = "Test Topic Whitespace String"
    relevant_chunks_whitespace_str = "   \n  " 
    prompt = prompts.get_user_prompt(topic, relevant_chunks_whitespace_str)
    assert isinstance(prompt, str)
    assert f"'{topic}'" in prompt
    # This covers the `all((not chunk or str(chunk).isspace()) for chunk in relevant_chunks)`
    # where relevant_chunks is a string of whitespace characters.
    assert "No specific context snippets were retrieved." in prompt
    assert "Generate the required output now." in prompt

def test_get_user_prompt_empty_topic():
    topic = ""
    relevant_chunks = "Some relevant content."
    prompt = prompts.get_user_prompt(topic, relevant_chunks)
    assert isinstance(prompt, str)
    assert f"'{topic}'" in prompt 
    assert "## Relevant Information Snippets:" in prompt
    assert relevant_chunks in prompt
    assert "Generate the required output now." in prompt

def test_get_fix_prompt():
    raw_text = "This is some faulty output."
    prompt = prompts.get_fix_prompt(raw_text)
    assert isinstance(prompt, str)
    assert "## Error Analysis" in prompt
    assert "## Correction Instructions" in prompt
    assert raw_text in prompt
    assert "Generate the corrected version now." in prompt

def test_get_fix_prompt_empty_raw_text():
    raw_text = ""
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

def test_get_clean_markdown_prompt_empty_raw_markdown():
    raw_markdown = ""
    prompt = prompts.get_clean_markdown_prompt(raw_markdown)
    assert isinstance(prompt, str)
    assert "**CRUCIAL OVERARCHING PRINCIPLE: VERBATIM EXTRACTION, NOT GENERATION.**" in prompt
    assert "<RAW_MARKDOWN_START>" in prompt
    assert raw_markdown in prompt 
    assert "<RAW_MARKDOWN_END>" in prompt
    assert "**Cleaned Markdown Output:**" in prompt

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
    assert '"scene_start_marker": "Exact first 4 to 7 words of the scene..."' in prompt

def test_get_storyboard_prompt_empty_story():
    story = ""
    prompt = prompts.get_storyboard_prompt(story)
    assert isinstance(prompt, str)
    assert "## Role" in prompt
    assert "Storyboard Planner" in prompt
    assert "## Input Story" in prompt
    assert story in prompt 
    assert "Generate valid JSON output now." in prompt

# Test for the internal logic of get_user_prompt regarding how it handles string `relevant_chunks`
# that might be non-whitespace but very short, or contain mixed whitespace/non-whitespace.
def test_get_user_prompt_various_string_relevant_chunks():
    topic = "Test Topic Various Chunks"
    
    # Test with a single non-whitespace character
    prompt_single_char = prompts.get_user_prompt(topic, "a")
    assert "## Relevant Information Snippets:" in prompt_single_char
    assert "No specific context snippets were retrieved." not in prompt_single_char
    assert " a " not in prompt_single_char # f-string adds space around {relevant_chunks}
    assert "\na\n" in prompt_single_char # Check actual formatting from f-string

    # Test with a string containing mixed leading/trailing whitespace
    prompt_mixed_whitespace = prompts.get_user_prompt(topic, "  b  ")
    assert "## Relevant Information Snippets:" in prompt_mixed_whitespace
    assert "No specific context snippets were retrieved." not in prompt_mixed_whitespace
    assert "\n  b  \n" in prompt_mixed_whitespace

    # Test with string that is effectively empty after potential stripping if that was the logic
    # (current logic doesn't strip, but good to be robust)
    prompt_empty_after_strip_hypothetical = prompts.get_user_prompt(topic, "")
    assert "No specific context snippets were retrieved." in prompt_empty_after_strip_hypothetical

    # Test with a list of strings that are not all whitespace (if relevant_chunks were a list of strings)
    # This is to ensure the `all((not chunk or str(chunk).isspace()) for chunk in relevant_chunks)`
    # part of the condition in `get_user_prompt` is robustly tested.
    # The function's type hint for relevant_chunks is `str`.
    # If a list is passed, it's converted to its string representation by the f-string.
    # e.g. `relevant_chunks = ["a", " "]; str(relevant_chunks)` is "['a', ' ']"
    # This string "['a', ' ']" is not empty and not all whitespace, so it takes the "else" path.
    list_chunks_as_string = str(["a", " "]) # "['a', ' ']"
    prompt_list_input_str = prompts.get_user_prompt(topic, list_chunks_as_string)
    assert "## Relevant Information Snippets:" in prompt_list_input_str
    assert "No specific context snippets were retrieved." not in prompt_list_input_str
    assert list_chunks_as_string in prompt_list_input_str

    # Test with a list of strings that ARE all whitespace/empty (if relevant_chunks were a list of strings)
    # e.g. `relevant_chunks = ["", " "]; str(relevant_chunks)` is "['', ' ']"
    # This string "['', ' ']" is not empty and not all whitespace itself.
    list_empty_chunks_as_string = str(["", " "]) # "['', ' ']"
    prompt_list_empty_input_str = prompts.get_user_prompt(topic, list_empty_chunks_as_string)
    assert "## Relevant Information Snippets:" in prompt_list_empty_input_str
    assert "No specific context snippets were retrieved." not in prompt_list_empty_input_str
    assert list_empty_chunks_as_string in prompt_list_empty_input_str

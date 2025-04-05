import pytest
from my_project.main import generate_story_script

def test_generate_story_script():
    # Basic test to ensure function runs without error
    output = generate_story_script("TestTopic", "TestSources")
    assert "LLM" not in output  # or some other logic

import pytest
from unittest.mock import patch, MagicMock

# Assuming the module is viralStoryGenerator.utils.text_processing
from viralStoryGenerator.utils import text_processing as tp_module

# --- Global Mocks & Fixtures (if any needed later) ---

@pytest.fixture
def mock_tp_logger():
    """Fixture to mock the _logger in text_processing.py if it exists and is used."""
    # Check if the module has a logger; if not, this patch isn't strictly needed
    # but doesn't hurt to have if one is added later to the function.
    if hasattr(tp_module, '_logger'):
        with patch('viralStoryGenerator.utils.text_processing._logger') as mock_logger:
            yield mock_logger
    else:
        yield MagicMock() # Dummy mock if no logger

# --- Tests for split_text_into_chunks ---

# Scenario 1: Basic splitting & Scenario 6: Hard split
@pytest.mark.parametrize("text, max_chunk_size, expected_num_chunks, expected_first_chunk_len_approx", [
    ("This is a long text that needs to be split into multiple chunks based on the max size. This part is for chunk one. This part is for chunk two.", 50, 3, 50),
    ("Short text", 100, 1, 10), # Text shorter than max_chunk_size
    ("NoSpacesHereJustOneLongWordThatExceedsTheChunkSizeLimit", 20, 3, 20), # Scenario 7: Long word
    ("Hard split test. " + "A"*60, 50, 2, 50), # Hard split after first sentence
])
def test_split_text_basic_and_hard_splitting(
    text, max_chunk_size, expected_num_chunks, expected_first_chunk_len_approx, mock_tp_logger
):
    chunks = tp_module.split_text_into_chunks(text, max_chunk_size)
    assert len(chunks) == expected_num_chunks
    if chunks:
        # Check first chunk length (approximate for hard splits, exact for sentence boundaries)
        # For very long words, the split might be exactly at max_chunk_size
        if "NoSpacesHere" in text:
             assert len(chunks[0]) == expected_first_chunk_len_approx
        # For other cases, it's more about being near max_chunk_size or at a boundary
        # This assertion might need refinement based on how "approximate" is defined.
        # For now, if it's not the long word case, we assume it's reasonably split.
        # A more robust check would be to ensure no chunk (except last) greatly exceeds max_chunk_size
        # and that splits occur at reasonable places.
        for i, chunk in enumerate(chunks):
            if i < len(chunks) -1 : # For all but the last chunk
                 assert len(chunk) <= max_chunk_size + 20 # Allow some leeway for boundary seeking
            else: # Last chunk can be shorter
                 assert len(chunk) <= max_chunk_size
            assert chunk.strip() == chunk # Ensure chunks are stripped


# Scenario 2: Respecting sentence boundaries & paragraph breaks
@pytest.mark.parametrize("text, max_chunk_size, expected_chunks", [
    ("First sentence. Second sentence. Third sentence here.", 30, 
     ["First sentence.", "Second sentence.", "Third sentence here."]),
    ("Sentence one! Sentence two? Sentence three followed by more text to make it longer than chunk size.", 40,
     ["Sentence one!", "Sentence two?", "Sentence three followed by more text to make it longer than chunk size."]), # Last chunk might be longer if it's the remainder
    ("Paragraph one ends here.\n\nParagraph two starts. This should be a new chunk if possible.", 50,
     ["Paragraph one ends here.", "Paragraph two starts. This should be a new chunk if possible."]),
    ("Short first sentence.\n\nThis is a much much much much much much much much much much much much much much much much much much much much longer second paragraph that will definitely exceed the max chunk size and should be split.", 60,
     ["Short first sentence.", "This is a much much much much much much much much much much much much much much much much much much much much longer second paragraph that will definitely exceed the max chunk size and should be split."]), # Second chunk might be split further based on its own logic or hard split.
                                                                                                                                                                                          # The test here is that the initial split respects the paragraph.
    ("Text before newline.\nText after newline but not a full paragraph break. This is still going.", 50,
     ["Text before newline.\nText after newline but not a full paragraph break. This is still going."]), # Single newline might not be a strong break
])
def test_split_text_respects_boundaries(text, max_chunk_size, expected_chunks, mock_tp_logger):
    chunks = tp_module.split_text_into_chunks(text, max_chunk_size)
    # Normalize expected chunks for comparison if they contain intentional extra spaces for readability in test
    normalized_expected_chunks = [re.sub(r'\s+', ' ', chunk.strip()) for chunk in expected_chunks]
    normalized_actual_chunks = [re.sub(r'\s+', ' ', chunk.strip()) for chunk in chunks]
    
    assert normalized_actual_chunks == normalized_expected_chunks
    for chunk in chunks:
        assert chunk.strip() == chunk


# Scenario 3: Text shorter than max_chunk_size (already covered by parametrize in basic splitting)
# Scenario 4: Empty text input
def test_split_text_empty_input(mock_tp_logger):
    assert tp_module.split_text_into_chunks("", 100) == []

# Scenario 5: Zero or negative max_chunk_size
@pytest.mark.parametrize("max_chunk_size_invalid", [0, -10])
def test_split_text_invalid_max_chunk_size(max_chunk_size_invalid, mock_tp_logger):
    assert tp_module.split_text_into_chunks("Some text.", max_chunk_size_invalid) == []
    # Check for log if specified (current code doesn't log for this, returns empty)

# Scenario 7: Text with very long words (covered by basic splitting)

# Scenario 8: Text with only whitespace
def test_split_text_only_whitespace(mock_tp_logger):
    assert tp_module.split_text_into_chunks("   \n \t  \n  ", 100) == []

# Scenario 9: Consecutive whitespace/newlines
def test_split_text_consecutive_whitespace(mock_tp_logger):
    text = "Sentence one.  \n\n  Sentence two. \t Sentence three.\n\n\nSentence four."
    max_chunk_size = 30 
    # Expected: Should split by sentences, and each chunk should be stripped.
    # No empty chunks should result from the multiple spaces/newlines.
    expected = ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four."]
    
    chunks = tp_module.split_text_into_chunks(text, max_chunk_size)
    
    # Normalize for comparison just in case of subtle space differences due to split points
    normalized_actual = [re.sub(r'\s+', ' ', chunk).strip() for chunk in chunks]
    normalized_expected = [re.sub(r'\s+', ' ', chunk).strip() for chunk in expected]

    assert normalized_actual == normalized_expected
    for chunk in chunks:
        assert chunk.strip() == chunk # Ensure individual chunks are stripped
        assert chunk != "" # Ensure no empty string chunks

# Test for a very long sentence with no natural breaks before max_chunk_size
def test_split_text_long_sentence_hard_split(mock_tp_logger):
    long_sentence = "Thisisasentencethatisverylongandhasnospacesorpunctuationformanycharacterssoitmustbehardstoppedatachunkboundaryeventhoughitlooksugly"
    max_chunk_size = 50
    chunks = tp_module.split_text_into_chunks(long_sentence, max_chunk_size)
    assert len(chunks) > 1
    assert len(chunks[0]) == max_chunk_size # First chunk should be exactly max_chunk_size
    assert chunks[0] == long_sentence[:max_chunk_size]
    assert chunks[1] == long_sentence[max_chunk_size:]

# Test for text slightly longer than max_chunk_size but ending with a sentence.
def test_split_text_slightly_longer_ends_with_sentence(mock_tp_logger):
    text = "This is a sentence that is just a tiny bit longer than the max chunk size of fifty. It should split here."
    max_chunk_size = 50
    # Expected: "This is a sentence that is just a tiny bit longer than the max chunk size of fifty."
    #           "It should split here."
    # The first part is > 50. The split logic prefers sentence boundaries.
    # If it finds "." before 50, it splits. If the sentence itself is > 50, it might take the whole sentence
    # or hard split.
    # Current logic: `find_best_split_point` looks back from `end`.
    # `end = min(start + max_chunk_size, len(text))`
    # If `text[start:end]` contains a boundary, it uses it. Otherwise, hard split at `max_chunk_size`.
    
    chunks = tp_module.split_text_into_chunks(text, max_chunk_size)
    # print(chunks) # For debugging test
    assert len(chunks) == 2
    assert chunks[0] == "This is a sentence that is just a tiny bit longer than the max chunk size of fifty."
    assert chunks[1] == "It should split here."

# Test for text where a paragraph break comes much earlier than max_chunk_size
def test_split_text_early_paragraph_break(mock_tp_logger):
    text = "Short first paragraph.\n\nThis is the second paragraph which is much longer and could have been part of the first chunk if not for the paragraph break but we want to respect it if possible."
    max_chunk_size = 100
    chunks = tp_module.split_text_into_chunks(text, max_chunk_size)
    # print(chunks)
    assert len(chunks) == 2
    assert chunks[0] == "Short first paragraph."
    assert chunks[1] == "This is the second paragraph which is much longer and could have been part of the first chunk if not for the paragraph break but we want to respect it if possible."

# Test to ensure chunk overlap logic (if any, not explicitly in current code but good to consider)
# The current code does not have explicit chunk_overlap parameter like in Langchain.
# It tries to find the "best" split point.

# Test with unicode sentence boundaries
def test_split_text_unicode_sentence_boundaries(mock_tp_logger):
    text = "第一句话结束了。第二句话也来了！第三句话呢？" # Chinese punctuation
    max_chunk_size = 10 # Force splits
    # Expected: ["第一句话结束了。", "第二句话也来了！", "第三句话呢？"]
    # Note: len("第一句话结束了。") is 8. len("第二句话也来了！") is 8. len("第三句话呢？") is 6.
    # These are all < 10. So, it should be one chunk if boundaries are not prioritized over length.
    # The current logic prioritizes splitting if a boundary is found *within* the initial chunk guess.
    # If the whole text is < max_chunk_size, it's one chunk.
    
    # If text is shorter than max_chunk_size, it returns as one chunk
    chunks_long_max = tp_module.split_text_into_chunks(text, 100)
    assert len(chunks_long_max) == 1
    assert chunks_long_max[0] == text

    # If max_chunk_size forces splits
    chunks_short_max = tp_module.split_text_into_chunks(text, 7) # "第一句话结束了。" is 8
    # print(chunks_short_max)
    assert len(chunks_short_max) == 3
    assert chunks_short_max[0] == "第一句话结束了" # The period is a boundary, so it's split before it if it exceeds.
                                          # Current regex `[.!?]\s+` requires a space after.
                                          # Chinese punctuation usually doesn't have space.
                                          # Let's update regex in mind or test as is.
                                          # The code uses `[.!?\u3002\uff01\uff1f]` which includes Chinese full stops.
                                          # And `\s*` so space is optional.
    assert chunks_short_max[0] == "第一句话结束了。" # With updated regex understanding
    assert chunks_short_max[1] == "第二句话也来了！"
    assert chunks_short_max[2] == "第三句话呢？"

# Final check on regex in `_find_best_split_point`
# `boundary_regex = r"(\n\n+|[.!?\u3002\uff01\uff1f]\s*)"`
# This looks for double newlines OR sentence ending punctuation followed by optional space.
# It then tries to find the *last* such boundary within `max_lookahead` from `preferred_split`.

# Retest a case that might be tricky for the boundary logic
def test_split_text_boundary_interaction_with_length(mock_tp_logger):
    text = "Short. This part is longer and goes beyond the max_chunk_size limit slightly."
    max_chunk_size = 10
    # "Short." (6) - first chunk
    # "This part is longer and goes beyond the max_chunk_size limit slightly." (70) - needs splitting
    #   "This part " (10) - second chunk (hard split)
    #   "is longer " (10) - third
    #   "and goes b" (10)
    #   "eyond the " (10)
    #   "max_chunk_" (10)
    #   "size limit" (10)
    #   " slightly." (10)
    chunks = tp_module.split_text_into_chunks(text, max_chunk_size)
    # print(f"Chunks for boundary_interaction: {chunks}")
    assert chunks[0] == "Short."
    assert len(chunks[1]) <= max_chunk_size
    assert "".join(chunks) == text.replace("  ", " ") # Allow for some space normalization if any

    text2 = "This is a veryverylongword. Next sentence is short."
    max_chunk_size = 20
    # "This is a veryverylo" (20)
    # "ngword. Next sentenc" (20) # or "ngword." (7) then "Next sentence is sho" (20)
    # "e is short." (11)
    # The logic: find end=start+max_chunk. Look back for boundary.
    # 1. start=0, end=20. Text="This is a veryverylo". No boundary. Hard split at 20. Chunk1="This is a veryverylo"
    # 2. start=20. Text="ngword. Next sentence is short.". end=20+20=40. Sub="ngword. Next sentenc". Boundary at "ngword.". Split there.
    #    Chunk2="ngword."
    # 3. start=20+len("ngword.") = 20+7=27. Text=" Next sentence is short.". end=27+20=47. Whole remaining.
    #    Chunk3=" Next sentence is short." (stripped to "Next sentence is short.")
    chunks2 = tp_module.split_text_into_chunks(text2, max_chunk_size)
    # print(f"Chunks for text2: {chunks2}")
    assert chunks2[0] == "This is a veryverylo"
    assert chunks2[1] == "ngword."
    assert chunks2[2] == "Next sentence is short."
    assert "".join(chunks2) == text2 # Ensure all content is preserved

# Test with a custom max_lookahead (though not a param of split_text_into_chunks directly)
# This is more for understanding the internal _find_best_split_point if it were exposed or configurable.
# Since it's not, this test is conceptual for now.

# Test for multiple paragraph breaks close together
def test_split_text_multiple_close_paragraph_breaks(mock_tp_logger):
    text = "Para1.\n\nPara2.\n\nPara3 is a bit longer.\n\nPara4."
    max_chunk_size = 15 # Small enough to force splits
    # Expected: each paragraph becomes a chunk if possible.
    # "Para1." (7)
    # "Para2." (7)
    # "Para3 is a bit longer." (22) -> might split if boundary logic is strict about length first
    #   Current logic: `end = min(start + max_chunk_size, len(text))`. Looks for boundary in `text[start:end]`.
    #   If start=0, text="Para1.\n\nPara2...", end=15. `text[0:15]` = "Para1.\n\nPara2.". Boundary at "Para1.\n\n".
    #   Chunk1 = "Para1."
    #   Next start = after "Para1.\n\n" (index 8). text="Para2.\n\nPara3...". end=8+15=23. `text[8:23]`="Para2.\n\nPara3 i". Boundary at "Para2.\n\n".
    #   Chunk2 = "Para2."
    #   Next start = after "Para2.\n\n" (index 8+8=16). text="Para3 is a bit longer.\n\nPara4.". end=16+15=31. `text[16:31]`="Para3 is a bit ". No boundary. Hard split?
    #      Or, if boundary search window (max_lookahead) is large enough, it might find "\n\n" after "longer.".
    #      The regex `(\n\n+|[.!?\u3002\uff01\uff1f]\s*)` includes `\n\n+`.
    #      _find_best_split_point search_window = text[start : start + max_chunk_size + max_lookahead]
    #      If "Para3 is a bit longer.\n\n" is within this window, it will be found.
    #      max_lookahead = max_chunk_size // 2 = 7.
    #      start=16. text[16 : 16+15+7] = text[16:38] = "Para3 is a bit longer.\n\nPa". Boundary at "longer.\n\n".
    #      So, Chunk3 = "Para3 is a bit longer."
    #   Next start = after "longer.\n\n" (index 16 + 24 = 40). text="Para4.". end=40+15=55. Whole remaining.
    #   Chunk4 = "Para4."
    
    chunks = tp_module.split_text_into_chunks(text, max_chunk_size)
    # print(f"Chunks for multiple_close_paragraph_breaks: {chunks}")
    assert chunks == ["Para1.", "Para2.", "Para3 is a bit longer.", "Para4."]

# Test case from the module docstring
def test_split_text_docstring_example(mock_tp_logger):
    text = "This is the first sentence. This is the second sentence, which is a bit longer.\n\nThis is a new paragraph. It also has two sentences. The final one."
    max_chunk_size = 60
    expected = [
        "This is the first sentence.",
        "This is the second sentence, which is a bit longer.", # Length 53
        "This is a new paragraph. It also has two sentences.", # Length 51
        "The final one." # Length 14
    ]
    # Re-evaluating based on code:
    # 1. start=0, end=60. text[0:60] = "This is the first sentence. This is the second sentence, wh". Boundary at "sentence. ".
    #    Chunk1 = "This is the first sentence."
    # 2. start=28. end=28+60=88. text[28:88] = " This is the second sentence, which is a bit longer.\n\nThis ". Boundary at "longer.\n\n".
    #    Chunk2 = "This is the second sentence, which is a bit longer." (after strip)
    # 3. start=28+54=82. end=82+60=142. text[82:142] = "This is a new paragraph. It also has two sentences. The fi". Boundary at "sentences. ".
    #    Chunk3 = "This is a new paragraph. It also has two sentences."
    # 4. start=82+52=134. end=134+60=194. text[134:end] = " The final one.". Whole remaining.
    #    Chunk4 = "The final one."
    
    chunks = tp_module.split_text_into_chunks(text, max_chunk_size)
    # print(f"Chunks for docstring_example: {chunks}")
    assert chunks == expected

# viralStoryGenerator/utils/text_processing.py
"""Text processing utilities."""

import re
from typing import List

from viralStoryGenerator.src.logger import logger as _logger

def split_text_into_chunks(text: str, max_chunk_size: int) -> List[str]:
    """
    Splits text into chunks, trying to respect sentence boundaries.

    Args:
        text: The input text to split.
        max_chunk_size: The approximate maximum size for each chunk.

    Returns:
        A list of text chunks.
    """
    if not text or max_chunk_size <= 0:
        return []

    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        end_pos = min(current_pos + max_chunk_size, text_len)

        # If we are not at the end of the text, try to find a sentence boundary
        if end_pos < text_len:
            # Look for sentence-ending punctuation (. ! ?) followed by space or newline
            sentence_end_match = re.search(r'[.!?]\s+', text[current_pos:end_pos])
            if sentence_end_match:
                # Find the last sentence end within the chunk
                last_sentence_end = -1
                for match in re.finditer(r'[.!?]\s+', text[current_pos:end_pos]):
                    last_sentence_end = match.end()

                if last_sentence_end != -1:
                    end_pos = current_pos + last_sentence_end
                else:
                    # If no sentence end found, try paragraph break (double newline)
                    paragraph_end_match = re.search(r'\n\n', text[current_pos:end_pos])
                    if paragraph_end_match:
                         last_paragraph_end = -1
                         for match in re.finditer(r'\n\n', text[current_pos:end_pos]):
                              last_paragraph_end = match.end()
                         if last_paragraph_end != -1:
                              end_pos = current_pos + last_paragraph_end
                    # else: fall back to hard split at max_chunk_size

            else:
                 # If no sentence end found, try paragraph break (double newline)
                 paragraph_end_match = re.search(r'\n\n', text[current_pos:end_pos])
                 if paragraph_end_match:
                      last_paragraph_end = -1
                      for match in re.finditer(r'\n\n', text[current_pos:end_pos]):
                           last_paragraph_end = match.end()
                      if last_paragraph_end != -1:
                           end_pos = current_pos + last_paragraph_end
                 # else: fall back to hard split at max_chunk_size


        chunk = text[current_pos:end_pos].strip()
        if chunk: # Avoid adding empty chunks
            chunks.append(chunk)
        current_pos = end_pos

        # Skip potential whitespace between chunks
        while current_pos < text_len and text[current_pos].isspace():
            current_pos += 1

    _logger.debug(f"Split text (length {text_len}) into {len(chunks)} chunks with max size {max_chunk_size}.")
    return chunks


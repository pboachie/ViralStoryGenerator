# viralStoryGenerator/source_cleanser.py

import requests
import json
import logging
import re
import hashlib
import shelve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define a persistent cache database filename.
CACHE_DB = "chunk_summary_cache.db"

def get_cache_key(text, model, temperature):
    """
    Returns a SHA256 hash as a cache key that uniquely identifies the request.
    The key incorporates the text, model, and temperature.
    """
    hasher = hashlib.sha256()
    hasher.update(text.encode('utf-8'))
    hasher.update(model.encode('utf-8'))
    hasher.update(str(temperature).encode('utf-8'))
    return hasher.hexdigest()


def cleanse_sources(raw_sources, endpoint, model, temperature=0.7):
    """
    Calls your local LLM to produce a cleaned up summary of raw_sources.
    (This is the same as your original function.)
    """
    system_prompt = (
        "You are a helpful assistant that merges multiple notes or sources into one cohesive summary ensuring the story is coherent and easy to understand.\n"
        "1. Summarize all major points or controversies.\n"
        "2. Remove duplicates or confusion.\n"
        "3. Return a concise but complete summary.\n"
        "4. No disclaimers or extraneous commentary—just the final summary.\n"
    )

    user_prompt = f"""
Below are several pieces of text (sources, notes, bullet points).
Please unify them into a short summary that accurately reflects the key points.
Clean up the language so it's coherent and easy to understand.

Sources:
{raw_sources}
""".strip()

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 8192,
        "stream": False
    }

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling the LLM for source cleansing: {e}")
        # Fallback: just return raw sources if something went wrong
        return raw_sources

    response_json = response.json()
    summary = response_json["choices"][0]["message"]["content"].strip()

    return summary


def cleanse_sources_cached(raw_sources, endpoint, model, temperature=0.7):
    """
    Wraps cleanse_sources() with a persistent cache.
    If the summary for the given raw_sources (with these parameters) has already been computed,
    it will be returned from the cache.
    """
    cache_key = get_cache_key(raw_sources, model, temperature)
    with shelve.open(CACHE_DB) as cache:
        if cache_key in cache:
            logging.info("Cache hit for the current source text. Using cached summary.")
            return cache[cache_key]
        else:
            logging.info("Cache miss. Calling LLM for source cleansing.")
            summary = cleanse_sources(raw_sources, endpoint, model, temperature)
            cache[cache_key] = summary
            return summary


def _chunk_text_by_words(text, chunk_size=1500):
    """
    Splits the text into roughly equal chunks by word count.
    chunk_size is how many words per chunk.
    Returns a list of chunk strings.
    """
    words = text.split()
    chunks = []
    current_chunk_words = []

    for word in words:
        current_chunk_words.append(word)
        # Once we hit chunk_size, finalize this chunk
        if len(current_chunk_words) >= chunk_size:
            chunk_text = " ".join(current_chunk_words)
            chunks.append(chunk_text)
            current_chunk_words = []
    # Append any leftover words
    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunks.append(chunk_text)

    return chunks


def chunkify_and_summarize(raw_sources, endpoint, model,
                           temperature=0.7, chunk_size=1500):
    """
    1) Split raw_sources into smaller chunks (default ~1500 words each).
    2) Summarize each chunk individually via cleanse_sources_cached().
    3) Merge those mini-summaries into one final summary
       (calling cleanse_sources_cached() again on the concatenated chunk summaries).
    Returns a single final summary string.
    """
    # Split the sources text if it is large
    chunks = _chunk_text_by_words(raw_sources, chunk_size=chunk_size)

    # If there's only one chunk, no need for multi-step summarization
    if len(chunks) == 1:
        return cleanse_sources_cached(chunks[0], endpoint, model, temperature)

    # Summarize each chunk individually
    logging.info(f"Splitting sources into {len(chunks)} chunks (chunk_size={chunk_size} words).")
    partial_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        logging.info(f"Summarizing chunk {i} of {len(chunks)}...")
        chunk_summary = cleanse_sources_cached(chunk, endpoint, model, temperature)
        partial_summaries.append(chunk_summary)

    # Now unify all chunk-level summaries into one final text
    logging.info("Merging chunk summaries into one final summary...")
    all_partial_text = "\n\n".join(partial_summaries)
    final_summary = cleanse_sources_cached(all_partial_text, endpoint, model, temperature)

    return final_summary

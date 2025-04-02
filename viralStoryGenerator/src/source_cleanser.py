# viralStoryGenerator/src/source_cleanser.py

import requests
import json
import re
import hashlib
import shelve
import time
from typing import List, Optional

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as appconfig

# Define cache database filename
CACHE_DB = "chunk_summary_cache.db"
APP_USER_AGENT = f"{appconfig.APP_TITLE}/{appconfig.VERSION}"

def get_cache_key(text: str, model: str, temperature: float) -> str:
    """Returns a SHA256 hash as a cache key."""
    hasher = hashlib.sha256()
    hasher.update(text.encode('utf-8'))
    hasher.update(model.encode('utf-8'))
    hasher.update(f"{temperature:.2f}".encode('utf-8'))
    return hasher.hexdigest()

def cleanse_sources(raw_sources: str, endpoint: str, model: str, temperature: float = 0.7) -> Optional[str]:
    """Calls LLM to produce a cleaned-up summary of raw_sources. Returns None on failure."""
    if not endpoint or not model:
        _logger.error("LLM endpoint or model not configured for cleanse_sources.")
        return None
    if not raw_sources or raw_sources.isspace():
         _logger.warning("cleanse_sources called with empty input.")
         return ""

    # Define prompts
    system_prompt = (
        "You are a helpful assistant that merges multiple notes or sources into one cohesive summary ensuring the story is coherent and easy to understand.\n"
        "1. Summarize all major points or controversies.\n"
        "2. Remove duplicates or confusion.\n"
        "3. Return a concise but complete summary.\n"
        "4. No disclaimers or extraneous commentary—just the final summary.\n"
        "5. Do not include any instructions or system prompts in the output.\n"
        "6. Do not include any references to the LLM or its capabilities and its training data.\n"
    )
    user_prompt = f"Below are several pieces of text (sources, notes, bullet points, articles).\nPlease unify them into a short summary that accurately reflects the key points.\nClean up the language so it's coherent and easy to understand.\n\nSources:\n{raw_sources}\n\nSummary:"

    headers = {
        "Content-Type": "application/json",
        "User-Agent": APP_USER_AGENT
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": appconfig.llm.MAX_TOKENS,
        "stream": False
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(data),
            timeout=appconfig.httpOptions.TIMEOUT
            )
        response.raise_for_status()
        response_json = response.json()
        summary = response_json["choices"][0]["message"]["content"].strip()
        return summary
    except requests.exceptions.Timeout:
        _logger.error(f"LLM request timed out during source cleansing to {endpoint}.")
        return None
    except requests.exceptions.RequestException as e:
        _logger.error(f"Error calling LLM for source cleansing: {e}")
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
         _logger.error(f"Failed to parse LLM response during source cleansing: {e}")
         return None


def cleanse_sources_cached(raw_sources: str, endpoint: str, model: str, temperature: float = 0.7) -> Optional[str]:
    """Wraps cleanse_sources() with a persistent cache. Returns None on failure."""
    cache_key = get_cache_key(raw_sources, model, temperature)
    summary = None
    try:
        # Use context manager for shelve
        with shelve.open(CACHE_DB) as cache:
            if cache_key in cache:
                _logger.info(f"Cache hit for source cleansing (key: {cache_key[:8]}...).")
                summary = cache[cache_key]
            else:
                _logger.info("Cache miss for source cleansing. Calling LLM.")
                summary = cleanse_sources(raw_sources, endpoint, model, temperature)
                if summary is not None:
                    cache[cache_key] = summary
                    _logger.debug(f"Cached cleansing result for key {cache_key[:8]}...")
                else:
                    _logger.error("LLM cleansing failed, not caching result.")
    except Exception as e:
        _logger.error(f"Failed to open or access cache file '{CACHE_DB}': {e}")
        _logger.warning("Falling back to non-cached source cleansing due to cache error.")
        summary = cleanse_sources(raw_sources, endpoint, model, temperature)

    return summary


def _chunk_text_by_words(text: str, chunk_size: int = 1500) -> List[str]:
    """Splits text into chunks by word count."""
    if chunk_size <= 0:
        _logger.warning(f"Invalid chunk_size {chunk_size}, defaulting to 1500.")
        chunk_size = 1500
    words = text.split()
    if not words:
        return []

    chunks = []
    current_chunk_words = []
    for word in words:
        current_chunk_words.append(word)
        if len(current_chunk_words) >= chunk_size:
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words = []
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    return chunks


def chunkify_and_summarize(raw_sources: str, endpoint: str, model: str,
                           temperature: float = 0.7, chunk_size: int = 3000) -> Optional[str]:
    """
    Splits sources, summarizes chunks (cached), merges summaries (cached).
    Returns a single final summary string, or None on failure.
    """
    if not raw_sources or raw_sources.isspace():
        _logger.info("chunkify_and_summarize called with empty input.")
        return ""

    # Split the sources text
    chunks = _chunk_text_by_words(raw_sources, chunk_size=chunk_size)
    if not chunks:
         _logger.warning("Source text resulted in zero chunks.")
         return ""

    # If only one chunk, summarize directly
    if len(chunks) == 1:
        _logger.debug("Single chunk detected, performing direct summary.")
        return cleanse_sources_cached(chunks[0], endpoint, model, temperature)

    # Summarize each chunk individually
    _logger.info(f"Splitting sources into {len(chunks)} chunks (size ~{chunk_size} words). Summarizing chunks...")
    partial_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        _logger.info(f"Summarizing chunk {i}/{len(chunks)}...")
        chunk_summary = cleanse_sources_cached(chunk, endpoint, model, temperature)
        if chunk_summary is None:
            _logger.error(f"Failed to summarize chunk {i}. Aborting multi-chunk summarization.")
            return None
        partial_summaries.append(chunk_summary)

    # Unify chunk summaries
    _logger.info("Merging chunk summaries into final summary...")
    if not partial_summaries:
        _logger.error("No partial summaries generated, cannot merge.")
        return None

    all_partial_text = "\n\n".join(partial_summaries)
    final_summary = cleanse_sources_cached(all_partial_text, endpoint, model, temperature)

    if final_summary is None:
         _logger.error("Failed to merge partial summaries.")
         return None

    _logger.info("Multi-chunk summarization complete.")
    return final_summary
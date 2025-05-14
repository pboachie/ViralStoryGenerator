"""
Utility module for processing API jobs.
"""

import asyncio
import time
import json
import datetime
import logging
from typing import Dict, Any

from viralStoryGenerator.utils.redis_manager import RedisMessageBroker
from viralStoryGenerator.src.llm import _extract_chain_of_thought, process_with_llm, clean_markdown_with_llm
from viralStoryGenerator.src.storyboard import generate_storyboard
from viralStoryGenerator.utils.storage_manager import storage_manager
from viralStoryGenerator.prompts.prompts import get_system_instructions, get_user_prompt
from viralStoryGenerator.utils.config import config as app_config
from viralStoryGenerator.utils.crawl4ai_scraper import queue_scrape_request, get_scrape_result
from viralStoryGenerator.utils.vector_db_manager import add_chunks_to_collection, query_collection
from viralStoryGenerator.utils.text_processing import split_text_into_chunks
import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

async def process_api_job(job_data: Dict[str, Any], consumer_name: str, group_name: str, message_broker) -> bool:
    """Process a single API job."""
    start_time = time.time()
    job_id = job_data.get("job_id", "unknown")
    story_script_info = None
    storyboard_info = None
    final_metadata_info = None

    _logger.debug(f"Processing job {job_id} via consumer '{consumer_name}' in group '{group_name}'")

    if not message_broker:
        _logger.error(f"Cannot process job {job_id}: Redis message broker unavailable")
        return False

    # Update job status to processing
    message_broker.track_job_progress(job_id, "processing", {"message": "Job processing started"})

    try:
        job_type = job_data.get("job_type", "unknown")

        if job_type == "generate_story":
            urls = job_data.get("urls", [])
            if isinstance(urls, str):
                try:
                    urls = json.loads(urls)
                except Exception:
                    urls = [urls]
            if not isinstance(urls, list) or not urls or not all(isinstance(u, str) and u.strip() for u in urls):
                _logger.warning(f"Message {job_id} has no valid URLs, acknowledging and skipping")
                message_broker.track_job_progress(job_id, "failed", {"error": "No valid URLs provided"})
                return False
            topic = job_data.get("topic", None)
            temperature = job_data.get("temperature", app_config.llm.TEMPERATURE)
            voice_id = job_data.get("voice_id")

            # --- Scraping --- >
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Scraping content from provided URLs", "progress": 10}
            )
            scraped_content = []
            scrape_failed = False
            if urls:
                _logger.info(f"Job {job_id}: Queuing scrape request for {len(urls)} URLs.")
                scrape_job_id = await queue_scrape_request(
                    urls,
                    user_query_for_bm25=topic,
                    wait_for_result=True,
                    timeout=app_config.httpOptions.TIMEOUT
                )
                if scrape_job_id:
                    _logger.info(f"Job {job_id}: Scrape job {scrape_job_id} queued. Retrieving result...")
                    scrape_result_data = await get_scrape_result(scrape_job_id)
                    if scrape_result_data:
                        scraped_content = [item.markdown_content for item in scrape_result_data if item and item.markdown_content]
                        if not scraped_content:
                            _logger.warning(f"Job {job_id}: Scrape job {scrape_job_id} completed but returned no content.")
                            scrape_failed = True
                        else:
                             _logger.info(f"Job {job_id}: Successfully retrieved scraped content ({len(scraped_content)} items)." )
                    else:
                        _logger.error(f"Job {job_id}: Failed to retrieve result for scrape job {scrape_job_id} after waiting (timed out or failed).")
                        scrape_failed = True
                else:
                    _logger.error(f"Job {job_id}: Scrape request {scrape_job_id or 'unknown'} failed or timed out.")
                    scrape_failed = True

            if scrape_failed:
                message_broker.track_job_progress(job_id, "failed", {"error": "Scraping step failed or timed out"})
                return False
            # < --- End Scraping ---

            # --- Clean Scraped contents --->
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Cleaning scraped content with LLM", "progress": 20}
            )
            cleaned_scraped_content = []
            if scraped_content:
                for content_item in scraped_content:
                    cleaned_data = clean_markdown_with_llm(content_item)
                    if cleaned_data:
                        cleaned_data, thinking = _extract_chain_of_thought(cleaned_data)
                        _logger.debug(f"Extracted thinking block: {thinking[:100]}...")
                    cleaned_scraped_content.append(cleaned_data)

            if not cleaned_scraped_content and urls:
                message_broker.track_job_progress(job_id, "failed", {"error": "Content cleaning step failed for all items."})
                return False

            _logger.info(f"Job {job_id}: Content cleaning completed. {len(cleaned_scraped_content)} items processed.")
            # < --- End Clean Scraped contents --- >

            rag_context = cleaned_scraped_content

            if len(rag_context) >= app_config.llm.MAX_TOKENS:
                _logger.warning(f"Job {job_id}: Content length exceeds LLM token limit. Using Chunking with vector DB (RAG) instead.")

                # --- Chunking & Vector DB Storage (RAG) ---
                message_broker.track_job_progress(
                    job_id,
                    "processing",
                    {"message": "Chunking cleaned content and storing in vector DB (RAG)", "progress": 30}
                )
                all_chunks = []
                chunk_metadatas = []
                chunk_size = app_config.rag.CHUNK_SIZE
                for idx, doc in enumerate(cleaned_scraped_content):
                    if not doc:
                        continue
                    chunks = split_text_into_chunks(doc, chunk_size)
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        chunk_metadatas.append({
                            "job_id": job_id,
                            "source_idx": idx,
                            "chunk_idx": i
                        })
                collection_name = f"job_{job_id}"
                if all_chunks:
                    add_chunks_to_collection(collection_name, all_chunks, metadatas=chunk_metadatas)
                # < --- End Chunking & Vector DB Storage ---

                # --- RAG Retrieval: Query relevant chunks for LLM prompt ---
                relevant_chunks = []
                if all_chunks:
                    query_results = query_collection(
                        collection_name,
                        query_texts=[topic],
                        n_results=app_config.rag.RELEVANT_CHUNKS_COUNT
                    )
                    relevant_chunks = query_results.get("documents", [])[0] if query_results.get("documents") else []
                if relevant_chunks:
                    rag_context = "\n\n".join(relevant_chunks)
                else:
                    rag_context = "\n\n".join(cleaned_scraped_content)
                # < --- End RAG Retrieval ---

            # --- LLM Processing --- >
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Generating story script with LLM", "progress": 40}
            )
            _logger.debug(f"Job {job_id}: Processing scraped content with LLM. Scraped content: {rag_context[:25]}")

            llm_result = process_with_llm(
                topic=topic,
                temperature=temperature,
                model=app_config.llm.MODEL,
                system_prompt=get_system_instructions(),
                user_prompt=get_user_prompt(topic, rag_context)
            )
            if not llm_result:
                 _logger.error(f"Job {job_id}: LLM processing returned empty result.")
                 message_broker.track_job_progress(job_id, "failed", {"error": "LLM processing failed or returned empty."})
                 return False
            # < --- End LLM Processing ---

            # --- Store Story Script --- >
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Storing story script", "progress": 50}
            )
            story_script_filename = f"{job_id}_story.txt"
            try:
                story_script_info = storage_manager.store_file(
                    file_data=llm_result,
                    file_type="story",
                    filename=story_script_filename,
                    content_type="text/plain"
                )
                if "error" in story_script_info:
                    _logger.error(f"Job {job_id}: Failed to store story script: {story_script_info.get('error')}")
                    message_broker.track_job_progress(job_id, "failed", {"error": f"Failed to store story script: {story_script_info.get('error')}"})
                    return False
                else:
                    _logger.info(f"Job {job_id}: Story script stored: {story_script_info.get('file_path')}")
            except Exception as store_err:
                 _logger.exception(f"Job {job_id}: Exception storing story script: {store_err}")
                 message_broker.track_job_progress(job_id, "failed", {"error": f"Exception storing story script: {store_err}"})
                 return False
            # < --- End Store Story Script ---

            # Respect ENABLE_IMAGE_GENERATION and ENABLE_AUDIO_GENERATION in job processing
            if not app_config.ENABLE_IMAGE_GENERATION:
                _logger.info("Image generation is disabled. Skipping image-related processing.")
                image_generation_enabled = False

            if not app_config.ENABLE_AUDIO_GENERATION:
                _logger.info("Audio generation is disabled. Skipping audio-related processing.")

            if not app_config.storyboard.ENABLE_STORYBOARD_GENERATION:
                _logger.info("Storyboard generation is disabled. Skipping storyboard-related processing.")

            # --- Storyboard Generation --- >
            storyboard_data = None
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Generating storyboard (including images/audio if enabled)", "progress": 60}
            )
            try:
                storyboard_data = generate_storyboard(
                    story=llm_result,
                    topic=topic,
                    task_id=job_id,
                    llm_endpoint=app_config.llm.ENDPOINT,
                    temperature=temperature,
                    voice_id=voice_id
                )
                if storyboard_data:
                    _logger.info(f"Job {job_id}: Storyboard generation completed.")
                    storyboard_info = {
                        "file_path": storyboard_data.get("storyboard_file"),
                        "url": storyboard_data.get("storyboard_url"),
                        "provider": storage_manager.provider # Assume same provider
                    }
                else:
                    _logger.warning(f"Job {job_id}: Storyboard generation returned None. Proceeding without storyboard.")
                    message_broker.track_job_progress(job_id, "processing", {"message": "Storyboard generation failed or skipped", "progress": 90})

            except Exception as sb_err:
                 _logger.exception(f"Job {job_id}: Error during storyboard generation: {sb_err}")
                 message_broker.track_job_progress(job_id, "processing", {"message": f"Storyboard generation failed: {sb_err}", "progress": 90})
            # < --- End Storyboard Generation ---

            # --- Final Metadata Aggregation & Storage --- >
            message_broker.track_job_progress(
                job_id,
                "processing",
                {"message": "Aggregating and storing final metadata", "progress": 95}
            )
            created_at_ts = job_data.get("request_time") or start_time
            try:
                created_at_float = float(created_at_ts)
            except (ValueError, TypeError):
                _logger.warning(f"Job {job_id}: Could not convert created_at_ts '{created_at_ts}' to float. Using current time as fallback.")
                created_at_float = start_time

            updated_at = time.time()

            _logger.debug(f"Job {job_id}: Final check before metadata creation. urls = {urls}, type = {type(urls)}")

            final_metadata = {
                "job_id": job_id,
                "topic": topic,
                "status": "completed",
                "message": "Job completed successfully.",
                "created_at": datetime.datetime.fromtimestamp(created_at_float, tz=datetime.timezone.utc).isoformat() if created_at_float else None,
                "updated_at": datetime.datetime.fromtimestamp(updated_at, tz=datetime.timezone.utc).isoformat(),
                "processing_time_seconds": round(updated_at - start_time, 2),
                "story_script_file": story_script_info.get("file_path") if story_script_info else None,
                "story_script_url": story_script_info.get("url") if story_script_info else None,
                "storyboard_file": storyboard_info.get("file_path") if storyboard_info else None,
                "storyboard_url": storyboard_info.get("url") if storyboard_info else None,
                "audio_file": storyboard_data.get("audio_file") if storyboard_data else None,
                "audio_url": storyboard_data.get("audio_url") if storyboard_data else None,
                "sources": urls if urls is not None else [],
                "story_script_llm_model": app_config.llm.MODEL,
                "storyboard_llm_model": app_config.llm.MODEL_MULTI,
                "llm_temperature": temperature,
                "voice_id": voice_id,
            }

            if not storyboard_info and 'sb_err' in locals():
                 final_metadata["message"] = f"Job completed, but storyboard generation failed: {sb_err}"
                 final_metadata["error"] = f"Storyboard generation failed: {sb_err}"
            elif not storyboard_info and not 'sb_err' in locals():
                 final_metadata["message"] = "Job completed, but storyboard generation returned no data."

            metadata_filename = f"{job_id}_metadata.json"
            try:
                metadata_json_str = json.dumps(final_metadata, indent=2)
                final_metadata_info = storage_manager.store_file(
                    file_data=metadata_json_str,
                    file_type="metadata",
                    filename=metadata_filename,
                    content_type="application/json"
                )
                if "error" in final_metadata_info:
                     _logger.error(f"Job {job_id}: Failed to store final metadata: {final_metadata_info.get('error')}")
                     message_broker.track_job_progress(job_id, "failed", {"error": f"Failed to store final metadata: {final_metadata_info.get('error')}"})
                     return False
                else:
                     _logger.info(f"Job {job_id}: Final metadata stored: {final_metadata_info.get('file_path')}")
            except Exception as meta_err:
                 _logger.exception(f"Job {job_id}: Exception storing final metadata: {meta_err}")
                 message_broker.track_job_progress(job_id, "failed", {"error": f"Exception storing final metadata: {meta_err}"})
                 return False
            # < --- End Final Metadata ---

            # --- Final Redis Update --- >
            redis_final_payload = {
                "message": final_metadata["message"],
                "story_script_ref": story_script_info.get("file_path") if story_script_info else None,
                "storyboard_ref": storyboard_info.get("file_path") if storyboard_info else None,
                "metadata_ref": final_metadata_info.get("file_path") if final_metadata_info else None,
                "processing_time": round(updated_at - start_time, 2),
                "created_at": final_metadata["created_at"],
                "updated_at": final_metadata["updated_at"],
            }
            if "error" in final_metadata:
                 redis_final_payload["error"] = final_metadata["error"]

            message_broker.track_job_progress(job_id, "completed", redis_final_payload)
            # < --- End Final Redis Update ---

            _logger.info(f"Job {job_id} completed successfully in {time.time() - start_time:.2f}s")
            return True
        else:
            # Unknown job type
            message_broker.track_job_progress(
                job_id,
                "failed",
                {"error": f"Unknown job type: {job_type}"}
            )
            _logger.warning(f"Unknown job type for {job_id}: {job_type}")
            return False

    except Exception as e:
        _logger.exception(f"Error processing job {job_id}: {e}")
        try:
            message_broker.track_job_progress(
                job_id,
                "failed",
                {
                    "error": f"Processing error: {str(e)}",
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                 }
            )
        except Exception as final_track_err:
             _logger.error(f"Job {job_id}: CRITICAL - Failed to update final status to failed in Redis: {final_track_err}")
        return False
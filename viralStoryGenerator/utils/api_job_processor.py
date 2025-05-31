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
    start_time = time.time()
    job_id = job_data.get("job_id", "unknown")
    story_script_info = None
    storyboard_info = None
    final_metadata_info = None

    _logger.debug(f"Processing job {job_id} via consumer '{consumer_name}' in group '{group_name}'")

    created_at_ts = job_data.get("request_time", start_time)
    try:
        created_at_float = float(created_at_ts)
    except (ValueError, TypeError):
        _logger.warning(f"Job {job_id}: Could not convert created_at_ts '{created_at_ts}' to float. Using job processing start_time as fallback.")
        created_at_float = start_time
    created_at_iso = datetime.datetime.fromtimestamp(created_at_float, tz=datetime.timezone.utc).isoformat()

    async def _track_progress(status: str, details: Dict[str, Any]):
        payload = details.copy()
        payload["created_at"] = created_at_iso
        payload["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        await message_broker.track_job_progress(job_id, status, payload)

    if not message_broker:
        _logger.error(f"Cannot process job {job_id}: Redis message broker unavailable")
        return False

    await _track_progress("processing", {"message": "Job processing started"})

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
                await _track_progress("failed", {"error": "No valid URLs provided"})
                return False

            topic_val = job_data.get("topic")
            topic_str: str
            if topic_val is None:
                _logger.warning(f"Job {job_id}: 'topic' is missing or None in job_data. Using default 'General Topic'.")
                topic_str = "General Topic"
            elif not isinstance(topic_val, str):
                _logger.warning(f"Job {job_id}: 'topic' is not a string (type: {type(topic_val)} value: '{topic_val}'). Converting to string.")
                topic_str = str(topic_val)
            else:
                topic_str = topic_val

            raw_temp = job_data.get("temperature")
            temperature: float
            default_temp_from_config = app_config.llm.TEMPERATURE
            final_default_temp = 0.9

            if default_temp_from_config is not None:
                try:
                    final_default_temp = float(default_temp_from_config)
                except (ValueError, TypeError):
                    _logger.warning(f"Job {job_id}: Invalid default temperature in config '{default_temp_from_config}'. Using {final_default_temp}.")

            if raw_temp is not None:
                try:
                    temperature = float(raw_temp)
                except (ValueError, TypeError):
                    _logger.warning(f"Job {job_id}: Invalid temperature value '{raw_temp}' in job_data. Defaulting to {final_default_temp}.")
                    temperature = final_default_temp
            else:
                temperature = final_default_temp

            voice_id = job_data.get("voice_id")

            llm_model_script = app_config.llm.MODEL
            if llm_model_script is None:
                _logger.error(f"Job {job_id}: LLM_MODEL for script generation is not configured in app_config.")
                await _track_progress("failed", {"error": "LLM_MODEL for script is not configured."})
                return False

            await _track_progress(
                "processing",
                {"message": "Scraping content from provided URLs", "progress": 10}
            )
            scraped_content = []
            scrape_failed = False
            if urls:
                _logger.debug(f"Job {job_id}: Queuing scrape request for {len(urls)} URLs.")
                scrape_job_id = await queue_scrape_request(
                    urls,
                    user_query_for_bm25=topic_str,
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
                             _logger.debug(f"Job {job_id}: Retrieved scraped content ({len(scraped_content)} items)")
                    else:
                        _logger.error(f"Job {job_id}: Failed to retrieve result for scrape job {scrape_job_id} after waiting (timed out or failed).")
                        scrape_failed = True
                else:
                    _logger.error(f"Job {job_id}: Scrape request {scrape_job_id or 'unknown'} failed or timed out.")
                    scrape_failed = True

            if scrape_failed:
                await _track_progress("failed", {"error": "Scraping step failed or timed out"})
                return False

            await _track_progress(
                "processing",
                {"message": "Cleaning scraped content with LLM", "progress": 20}
            )
            cleaned_scraped_content = []
            if scraped_content:
                for content_item in scraped_content:
                    if not content_item or not isinstance(content_item, str):
                        continue
                    cleaned_data_str = clean_markdown_with_llm(content_item)
                    if cleaned_data_str:
                        cleaned_data_text, thinking = _extract_chain_of_thought(cleaned_data_str)
                        _logger.debug(f"Extracted thinking block for item: {thinking[:100]}...")
                        if cleaned_data_text and isinstance(cleaned_data_text, str):
                           cleaned_scraped_content.append(cleaned_data_text)

            _logger.debug(f"Job {job_id}: Content cleaning completed ({len(cleaned_scraped_content)} items)")

            rag_context_str: str

            estimated_chars = sum(len(item) for item in cleaned_scraped_content)
            max_chars_before_rag = (app_config.llm.MAX_TOKENS or 8000) * 2.5

            _logger.debug(f"Job {job_id}: Estimated characters in cleaned content: {estimated_chars}. Max chars before RAG: {max_chars_before_rag}.")

            if cleaned_scraped_content and estimated_chars > max_chars_before_rag:
                _logger.warning(f"Job {job_id}: Estimated content characters ({estimated_chars}) exceeds threshold ({max_chars_before_rag}). Using Chunking with vector DB (RAG).")

                await _track_progress(
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

                relevant_chunks = []
                if all_chunks:
                    query_results = query_collection(
                        collection_name,
                        query_texts=[topic_str],
                        n_results=app_config.rag.RELEVANT_CHUNKS_COUNT
                    )
                    docs_from_query = query_results.get("documents")
                    if docs_from_query and isinstance(docs_from_query, list) and len(docs_from_query) > 0 and isinstance(docs_from_query[0], list):
                        relevant_chunks = [str(chunk) for chunk in docs_from_query[0] if chunk is not None]

                if relevant_chunks:
                    rag_context_str = "\n\n".join(relevant_chunks)
                    _logger.debug(f"Job {job_id}: Using {len(relevant_chunks)} relevant chunks from RAG")
                elif cleaned_scraped_content:
                    _logger.debug(f"Job {job_id}: No RAG chunks, using all cleaned content")
                    rag_context_str = "\n\n".join(cleaned_scraped_content)
                else:
                    rag_context_str = ""
            else:
                if cleaned_scraped_content:
                    rag_context_str = "\n\n".join(cleaned_scraped_content)
                    _logger.debug(f"Job {job_id}: Using all cleaned content directly ({len(cleaned_scraped_content)} items)")
                else:
                    rag_context_str = ""
                    _logger.debug(f"Job {job_id}: No scraped content for LLM context")

            await _track_progress(
                "processing",
                {"message": "Generating story script with LLM", "progress": 40}
            )
            _logger.debug(f"Job {job_id}: Processing content with LLM. Topic: '{topic_str}', Temperature: {temperature}, Model: {llm_model_script}. Context preview: '{rag_context_str[:100]}...'")

            llm_result = process_with_llm(
                topic=topic_str,
                temperature=temperature,
                model=llm_model_script,
                system_prompt=get_system_instructions(),
                user_prompt=get_user_prompt(topic_str, rag_context_str)
            )
            if not llm_result:
                 _logger.error(f"Job {job_id}: LLM processing returned empty result.")
                 await _track_progress("failed", {"error": "LLM processing failed or returned empty."})
                 return False

            await _track_progress(
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
                    await _track_progress("failed", {"error": f"Failed to store story script: {story_script_info.get('error')}"})
                    return False
                else:
                    _logger.debug(f"Job {job_id}: Story script stored at {story_script_info.get('file_path')}")
            except Exception as store_err:
                 _logger.exception(f"Job {job_id}: Exception storing story script: {store_err}")
                 await _track_progress("failed", {"error": f"Exception storing story script: {store_err}"})
                 return False

            if not app_config.ENABLE_IMAGE_GENERATION:
                _logger.info("Image generation is disabled. Skipping image-related processing.")
                image_generation_enabled = False

            if not app_config.ENABLE_AUDIO_GENERATION:
                _logger.info("Audio generation is disabled. Skipping audio-related processing.")

            if not app_config.storyboard.ENABLE_STORYBOARD_GENERATION:
                _logger.info("Storyboard generation is disabled. Skipping storyboard-related processing.")

            storyboard_data = None
            storyboard_info = None
            sb_err = None

            raw_gen_sb_flag = job_data.get("generate_storyboard")
            should_generate_storyboard: bool

            if isinstance(raw_gen_sb_flag, bool):
                should_generate_storyboard = raw_gen_sb_flag
                _logger.info(f"Job {job_id}: 'generate_storyboard' flag from job_data (boolean): {should_generate_storyboard}")
            elif isinstance(raw_gen_sb_flag, str):
                should_generate_storyboard = raw_gen_sb_flag.lower() == 'true'
                _logger.info(f"Job {job_id}: 'generate_storyboard' flag from job_data (string): '{raw_gen_sb_flag}', parsed as: {should_generate_storyboard}")
            else:
                should_generate_storyboard = app_config.storyboard.ENABLE_STORYBOARD_GENERATION
                _logger.info(f"Job {job_id}: 'generate_storyboard' flag not found or invalid in job_data (value: {raw_gen_sb_flag}). Defaulting to global app_config.storyboard.ENABLE_STORYBOARD_GENERATION: {should_generate_storyboard}")

            if should_generate_storyboard:
                if not llm_result:
                    _logger.warning(f"Job {job_id}: Skipping storyboard generation because story script (llm_result) is empty.")
                    await _track_progress("processing", {"message": "Storyboard skipped: story script is empty.", "progress": 75})
                else:
                    _logger.info(f"Job {job_id}: Storyboard generation is ENABLED for this job. Proceeding.")

                    llm_endpoint_storyboard = app_config.llm.ENDPOINT
                    if llm_endpoint_storyboard is None:
                        _logger.error(f"Job {job_id}: LLM_ENDPOINT for storyboard is not configured in app_config. Skipping storyboard.")
                        await _track_progress("processing", {"message": "Storyboard generation skipped: LLM_ENDPOINT not configured.", "progress": 75})
                        sb_err = ValueError("LLM_ENDPOINT for storyboard not configured.")
                    else:
                        await _track_progress(
                            "processing",
                            {"message": "Generating storyboard (images/audio if enabled by their global flags)", "progress": 60}
                        )
                        try:
                            storyboard_data = generate_storyboard(
                                story=llm_result,
                                topic=topic_str,
                                task_id=job_id,
                                llm_endpoint=llm_endpoint_storyboard,
                                temperature=temperature,
                                voice_id=voice_id
                            )
                            if storyboard_data:
                                _logger.info(f"Job {job_id}: Storyboard generation process completed.")
                                storyboard_info = {
                                    "file_path": storyboard_data.get("storyboard_file"),
                                    "url": storyboard_data.get("storyboard_url"),
                                    "provider": storage_manager.provider
                                }
                            else:
                                _logger.debug(f"Job {job_id}: Storyboard generation returned no data")
                                await _track_progress("processing", {"message": "Storyboard generation returned no data.", "progress": 90})
                        except Exception as e:
                            sb_err = e
                            _logger.exception(f"Job {job_id}: Error during storyboard generation call: {sb_err}")
                            await _track_progress("processing", {"message": f"Storyboard generation failed: {str(sb_err)}", "progress": 90})
            else:
                _logger.debug(f"Job {job_id}: Storyboard generation disabled via config")
                await _track_progress("processing", {"message": "Storyboard generation skipped as per job/app configuration", "progress": 90})

            await _track_progress(
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
                "topic": topic_str,
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
                "story_script_llm_model": llm_model_script,
                "storyboard_llm_model": app_config.llm.MODEL_MULTI if app_config.llm.MODEL_MULTI else llm_model_script,
                "llm_temperature": temperature,
                "voice_id": voice_id,
            }

            if sb_err is not None:
                 final_metadata["message"] = f"Job completed, but storyboard generation failed: {str(sb_err)}"
                 final_metadata["error_details"] = final_metadata.get("error_details", {})
                 final_metadata["error_details"]["storyboard_generation"] = str(sb_err)
            elif should_generate_storyboard and not storyboard_info and llm_result:
                 final_metadata["message"] = "Job completed. Storyboard generation was attempted but produced no output."
            elif not should_generate_storyboard:
                 final_metadata["message"] = "Job completed. Storyboard generation was skipped as per configuration."
                 final_metadata["storyboard_file"] = None
                 final_metadata["storyboard_url"] = None
                 final_metadata["audio_file"] = None
                 final_metadata["audio_url"] = None
            elif not llm_result and should_generate_storyboard:
                 final_metadata["message"] = "Job completed. Story script generation failed; storyboard generation was skipped."


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
                     await _track_progress("failed", {"error": f"Failed to store final metadata: {final_metadata_info.get('error')}"})
                     return False
                else:
                     _logger.debug(f"Job {job_id}: Final metadata stored at {final_metadata_info.get('file_path')}")
            except Exception as meta_err:
                 _logger.exception(f"Job {job_id}: Exception storing final metadata: {meta_err}")
                 await _track_progress("failed", {"error": f"Exception storing final metadata: {meta_err}"})
                 return False

            story_details_for_redis = []
            if story_script_info and story_script_info.get("url"):
                story_details_for_redis.append({"url": story_script_info.get("url"), "content": None})

            storyboard_details_for_redis = []
            if storyboard_info and storyboard_info.get("url"):
                storyboard_details_for_redis.append({"url": storyboard_info.get("url"), "content": None})

            redis_final_payload_for_completed_job = {
                "message": final_metadata["message"],
                "story": story_details_for_redis if story_details_for_redis else None,
                "storyboard": storyboard_details_for_redis if storyboard_details_for_redis else None,
                "audio_url": final_metadata.get("audio_url"),
                "sources": final_metadata.get("sources"),
                "processing_time_seconds": final_metadata.get("processing_time_seconds"),
                "created_at": final_metadata.get("created_at"),
                "updated_at": final_metadata.get("updated_at"),
            }
            _logger.debug(f"Job {job_id}: Final Redis payload for completed status: {json.dumps(redis_final_payload_for_completed_job, indent=2)}")

            await message_broker.track_job_progress(job_id, "completed", redis_final_payload_for_completed_job)

            _logger.debug(f"Job {job_id} completed in {time.time() - start_time:.2f}s")
            return True
        else:
            await _track_progress(
                "failed",
                {"error": f"Unknown job type: {job_type}"}
            )
            _logger.warning(f"Unknown job type for {job_id}: {job_type}")
            return False

    except Exception as e:
        _logger.exception(f"Error processing job {job_id}: {e}")
        try:
            await _track_progress(
                "failed",
                {"error": f"Processing error: {str(e)}"}
            )
        except Exception as final_track_err:
             _logger.error(f"Job {job_id}: CRITICAL - Failed to update final status to failed in Redis: {final_track_err}")
        return False
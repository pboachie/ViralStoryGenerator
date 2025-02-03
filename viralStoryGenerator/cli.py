# viralStoryGenerator/cli.py

import argparse
import time
import logging
import os
import datetime

from viralStoryGenerator.llm import generate_story_script
from viralStoryGenerator.source_cleanser import chunkify_and_summarize  # We'll reuse this for multi-chunk summarization

def _save_story_output(result, topic):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    week_num = now.isocalendar().week

    folder_path = os.path.join("GeneratedStories", f"Week{week_num}")
    os.makedirs(folder_path, exist_ok=True)

    safe_topic = topic.replace("/", "_").replace("\\", "_")
    file_name = f"{safe_topic} - {date_str}.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("### Story Script:\n")
        f.write(result.get("story", ""))
        f.write("\n\n### Video Description:\n")
        f.write(result.get("video_description", ""))

    logging.info(f"Story saved to {file_path}")

def _read_sources_from_folder(folder_path):
    """
    Reads all files from `folder_path` (e.g., .txt, .md, or any extension),
    concatenates their contents, and returns one combined string.
    """
    combined_texts = []
    if not os.path.isdir(folder_path):
        logging.warning(f"Sources folder does not exist: {folder_path}")
        return ""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        combined_texts.append(text)
            except Exception as e:
                logging.error(f"Failed to read {file_path}: {e}")

    # Combine them into one big block of text separated by double newlines
    return "\n\n".join(combined_texts)

def cli_main():
    parser = argparse.ArgumentParser(
        description="Generate short, informal story scripts via a local LLM endpoint."
    )
    parser.add_argument("--topic", required=True, help="The topic for the story script.")
    parser.add_argument("--sources-folder", default="sources",
                        help="Folder containing files with notes/articles (defaults to 'sources').")
    parser.add_argument("--endpoint", default="http://192.168.1.190:1234/v1/chat/completions",
                        help="Local LLM API endpoint.")
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-14b@q4_k_m",
                        help="Model to use for generating the story.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher => more random).")
    parser.add_argument("--show-thinking", action="store_true",
                        help="If passed, print chain-of-thought (if available).")
    parser.add_argument("--chunk-size", type=int, default=1500,
                        help="Word chunk size for splitting sources.")
    args = parser.parse_args()

    start_exec = time.time()

    # 1) Read all the files from the sources folder into one combined text
    logging.info(f"Reading all files in folder '{args.sources_folder}' for sources...")
    raw_sources = _read_sources_from_folder(args.sources_folder)

    # 2) Chunkify & Summarize them into a single cohesive summary
    if raw_sources.strip():
        logging.info("Splitting & summarizing sources via LLM (multi-chunk)...")
        cleansed_sources = chunkify_and_summarize(
            raw_sources=raw_sources,
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            chunk_size=args.chunk_size
        )
        logging.info("Sources cleansed. Proceeding with story generation...")
    else:
        cleansed_sources = ""

    # 3) Generate the story script from these cleansed/merged sources
    result = generate_story_script(
        topic=args.topic,
        sources=cleansed_sources,
        endpoint=args.endpoint,
        model=args.model,
        temperature=args.temperature,
        show_thinking=args.show_thinking
    )

    # 4) Print chain-of-thought if requested
    if args.show_thinking and result.get("thinking"):
        print("\n=== CHAIN-OF-THOUGHT (DEBUG) ===")
        print(result["thinking"], "\n")

    # 5) Print final story
    if "story" in result and "video_description" in result and result["story"]:
        print("\n=== AGENT OUTPUT ===")
        print(f"### Story Script:\n{result['story']}\n")
        print(f"### Video Description:\n{result['video_description']}\n")
    else:
        print("\n=== RAW OUTPUT (Unformatted) ===")
        print(result)

    # 6) Save the final outputs
    _save_story_output(result, args.topic)

    total_exec_time = time.time() - start_exec
    logging.info(f"Total execution time (CLI start to finish): {total_exec_time:.2f} seconds")

if __name__ == "__main__":
    cli_main()

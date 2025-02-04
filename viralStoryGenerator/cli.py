# viralStoryGenerator/cli.py

import argparse
import time
import logging
import os
import datetime

from viralStoryGenerator.llm import generate_story_script
from viralStoryGenerator.source_cleanser import chunkify_and_summarize
from viralStoryGenerator.elevenlabs_tts import generate_elevenlabs_audio

def _save_story_output(result, topic):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    week_num = now.isocalendar().week

    folder_path = os.path.join("GeneratedStories", f"Week{week_num}")
    os.makedirs(folder_path, exist_ok=True)

    # Build text filename
    safe_topic = topic.replace("/", "_").replace("\\", "_")
    txt_file_name = f"{safe_topic} - {date_str}.txt"
    txt_file_path = os.path.join(folder_path, txt_file_name)

    # Write out the story & description
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write("### Story Script:\n")
        f.write(result.get("story", ""))
        f.write("\n\n### Video Description:\n")
        f.write(result.get("video_description", ""))

    logging.info(f"Story saved to {txt_file_path}")

    # Also generate audio from the story (if we have content)
    story_text = result.get("story", "")
    if story_text.strip():
        # Build MP3 filename (same base name, .mp3 extension)
        base_name = os.path.splitext(txt_file_name)[0]  # e.g. "MyTopic - 2025-02-02"
        mp3_file_path = os.path.join(folder_path, f"{base_name}.mp3")

        # We'll assume your ElevenLabs API key is stored somewhere
        # e.g. ENV var, config file, or CLI argument
        api_key = os.environ.get("ELEVENLABS_API_KEY", "sk_15cb1ec5322909d636dd3afb9223dd65578013807895d481")  # or pass from CLI
        if not api_key:
            logging.warning("No ElevenLabs API Key found. Skipping TTS generation.")
            return

        # Optionally let user pass a voice ID on the CLI or default
        default_voice_id = None  # will revert to the default inside generate_elevenlabs_audio
        success = generate_elevenlabs_audio(
            text=story_text,
            api_key=api_key,
            output_mp3_path=mp3_file_path,
            voice_id=default_voice_id,    # or from CLI arg
            model_id="eleven_monolingual_v2",  # or "eleven_multilingual_v2"
            stability=0.5,
            similarity_boost=0.75
        )
        if success:
            logging.info(f"Audio TTS saved to {mp3_file_path}")
        else:
            logging.warning("Audio generation failed.")

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
    parser.add_argument("--topic", required=True, help="Topic for the story script")
    parser.add_argument("--sources-folder", default="sources", help="Folder with source files")
    parser.add_argument("--endpoint", default="http://192.168.1.190:1234/v1/chat/completions",
                        help="Local LLM API endpoint.")
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-14b@q4_k_m",
                        help="Which model to use for generating the story.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher => more random).")
    parser.add_argument("--show-thinking", action="store_true",
                        help="If passed, print chain-of-thought (if available).")
    parser.add_argument("--chunk-size", type=int, default=1500,
                        help="Word chunk size for splitting sources.")
    # Optionally add --voice-id if you want user to specify a voice
    parser.add_argument("--voice-id", default=None, help="ElevenLabs voice ID override")
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

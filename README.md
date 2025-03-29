# Viral Story Generator 🚀✨

Welcome to **Viral Story Generator** – your one-stop tool for creating engaging, short story scripts powered by a local LLM endpoint. This project not only generates compelling story scripts but also produces audio using ElevenLabs and constructs detailed storyboards for video production. 🎉

---

## Features
- **Story Script Generation:** Automatically generates creative story scripts with video descriptions.
- **Audio TTS:** Converts the narrative into high-quality MP3 audio using ElevenLabs TTS. 🔊
- **Storyboard Creation:** Produces JSON-formatted storyboards for video planning and production.
- **Source Cleansing:** Merges and summarizes multiple source files into a coherent narrative.

---

## Installation
Install the package using Python's setuptools:

```bash
python setup.py install
```

Ensure you have the required dependencies installed:
- `requests`
- `python-dotenv`

You can install these via pip if needed:
```bash
pip install requests python-dotenv
```

---

## Configuration
Copy the sample environment file to a new file named `.env` and update the values as needed:
- `LOG_LEVEL` (e.g., DEBUG, INFO, etc.)
- `LLM_ENDPOINT` – Local LLM API endpoint.
- `LLM_MODEL` – Model name for story generation.
- `ELEVENLABS_API_KEY` – Your ElevenLabs API key.
- `ELEVENLABS_VOICE_ID` – ElevenLabs voice ID.
- And more options in the provided [.env.sample](.env.sample). 🔧

## Setting Up Your Environment
1. **Create a Virtual Environment:**
   Open your terminal and run:
   ```bash
   python -m venv venv
   ```
2. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. **Copy the .env Example:**
   Copy the sample file to create your environment configuration:
   ```bash
   cp .env.sample .env
   ```
4. **Edit the .env File:**
   Open `.env` in your favorite text editor and adjust the configuration values to match your setup.

---

## Usage
Run the CLI tool by specifying the topic and sources folder:
```bash
viralStoryGenerator --topic "Who really invented the wheel"
```

For additional options and help, execute:
```bash
viralStoryGenerator --help
```

---

## Contributing
Contributions and feedback are welcome! Feel free to open issues or submit pull requests to help improve the project. 🤝

For inquiries or feedback, please reach out at [princeboachie@gmail.com](mailto:princeboachie@gmail.com).

---

## License
This project is licensed as specified in the [LICENSE](#file:LICENSE) file.

---

Enjoy creating viral stories and engaging content! 💥🔥

---

## Changelog

**Updates:**
- **Environment & Configuration:**
  - Added a new configuration class in `utils/config.py` to load environment variables from `.env`.
  - Updated `.env.sample` with parameters such as `HTTP_TIMEOUT` and `LLM_MAX_TOKENS`.

- **CLI & Debugging:**
  - Updated CLI behavior in `src/cli.py` to support processing source files and chunking via the LLM.
  - Refined the launch configurations in `.vscode/launch.json` to point to the correct main script with optional arguments.

- **Dependency Management:**
  - Pinned dependency versions in `setup.py` (e.g., `requests==2.32.3` and `python-dotenv==1.1.0`).

- **Prompts & Story Generation:**
  - Enhanced prompt instructions in `prompts/prompts.py` to enforce proper formatting of story scripts and video descriptions.

- **Logging:**
  - Updated the logging configuration in `src/logger.py` to prevent duplicate logs and include file logging in production.

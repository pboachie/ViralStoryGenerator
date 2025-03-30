# Viral Story Generator 🚀✨

Welcome to **Viral Story Generator** – your one-stop tool for creating engaging, short story scripts powered by a local LLM endpoint. This project not only generates compelling story scripts but also produces audio using ElevenLabs and constructs detailed storyboards for video production. 🎉

---

## Features
- **Story Script Generation:** Automatically generates creative story scripts with video descriptions.
- **Audio TTS:** Converts the narrative into high-quality MP3 audio using ElevenLabs TTS. 🔊
- **Storyboard Creation:** Produces JSON-formatted storyboards for video planning and production.
- **Source Cleansing:** Merges and summarizes multiple source files into a coherent narrative.
- **Web Scraping:** Utilizes crawl4ai to extract content from web sources.

---

## Installation
Install the package using Python's setuptools:

```bash
python setup.py install
```

Ensure you have the required dependencies installed:
- `requests`
- `python-dotenv`
- `crawl4ai`

You can install these via pip if needed:
```bash
pip install requests python-dotenv crawl4ai
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

## Docker Setup 🐳

### Development Setup
To run the application in a development environment using Docker:

1. **Build and start the containers:**
   ```bash
   docker-compose up -d
   ```

2. **Check the status of the containers:**
   ```bash
   docker-compose ps
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop the containers:**
   ```bash
   docker-compose down
   ```

### Production Setup
For production deployment with scaling and monitoring:

1. **Copy and customize environment variables:**
   ```bash
   cp .env.sample .env
   ```

2. **Deploy with production configuration:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Scale services as needed:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d --scale scraper=5 --scale backend=3
   ```

4. **Access monitoring dashboards:**
   - Grafana: http://localhost:3000 (default credentials: admin/admin)
   - Prometheus: http://localhost:9090

### Container Architecture
This project uses a containerized architecture with separate services:

- **Redis**: Queue management and shared state
- **Backend**: Main application for story generation
- **Scraper**: Worker processes for crawl4ai web scraping
- **Monitoring**: Prometheus and Grafana for observability (production only)

Each service is independently scalable to handle increased load.

---

## Dependencies & Citations

### Crawl4AI
This project uses Crawl4AI for web scraping:

```bibtex
@software{crawl4ai2024,
  author = {UncleCode},
  title = {Crawl4AI: Open-source LLM Friendly Web Crawler & Scraper},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/unclecode/crawl4ai}}
}
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
  
- **Docker & Containerization:**
  - Added Docker configuration for development and production environments.
  - Included scalable services for backend, scraper, and Redis.
  - Implemented monitoring with Prometheus and Grafana.

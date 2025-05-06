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

You can install these via pip if needed:
```bash
pip install .
```

---

## Codebase Structure

The project is organized as follows:

- `viralStoryGenerator/` – Main package directory
  - `src/` – Core source code
    - `api.py` – FastAPI application and endpoints
    - `worker_runner.py` – Unified entry point for running API and worker processes
    - `api_worker.py`, `queue_worker.py`, `scrape_worker.py` – Worker process scripts
    - `logger.py` – Logging configuration
    - `llm.py`, `elevenlabs_tts.py`, `storyboard.py`, etc. – Main features and utilities
  - `utils/` – Utility modules (config, Redis, storage, etc.)
  - `models/` – Data models
  - `prompts/` – Prompt templates for LLM
- `tests/` – Test suite
- `docker-compose.yml`, `Dockerfile*` – Containerization and orchestration
- `README.md`, `requirements.txt`, `setup.py` – Documentation and dependencies

---

## Local Development & Testing

For local development and testing, follow these steps:

### Setup Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ViralStoryGenerator.git
   cd ViralStoryGenerator
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   ```
   This installs the package in "editable" mode, allowing you to modify the code and see changes immediately without reinstalling.

3. **Set up environment variables:**
   ```bash
   cp .env.sample .env
   ```
   Edit the `.env` file with your local configuration.

### Running Tests

1. **Execute the test suite:**
   ```bash
   python -m pytest tests/
   ```

2. **Run specific test files:**
   ```bash
   python -m pytest tests/test_main.py
   ```

3. **Run with verbose output:**
   ```bash
   python -m pytest -v tests/
   ```

### Manual Testing

1. **Start the API server locally:**
   ```bash
   python -m viralStoryGenerator.src.worker_runner api --port 8000 --reload
   ```
   The `--reload` flag enables auto-reloading when code changes are detected.

2. **Access the API documentation:**
   Open your browser and navigate to:
   ```
   http://localhost:8000/docs
   ```
   This provides an interactive Swagger UI for testing endpoints.

3. **Test individual components:**
   ```bash
   # Test LLM processing
   python -c "from viralStoryGenerator.src.llm import process_with_llm; print(process_with_llm('Sample topic', 'Sample content', 0.7))"

   # Test audio generation
   python -c "from viralStoryGenerator.src.elevenlabs_tts import generate_audio; print(generate_audio('This is a test audio generation'))"
   ```

### Debugging

- Set `LOG_LEVEL=DEBUG` in your `.env` file for detailed logging
- Use Python's debugger:
  ```bash
  python -m pdb -c continue -m viralStoryGenerator api
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
The Viral Story Generator is now available as an HTTP service. You can interact with it using the REST API endpoints.

### API Endpoints
Access the service through the following endpoints:

- **Generate a story:** Send a POST request to the story generation endpoint with your topic
- **Check status:** Monitor the progress of your story generation task
- **Retrieve results:** Get your completed story, audio, and storyboard

For detailed API documentation, refer to the API documentation when the server is running.

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

### Worker Commands

- **Scraper Worker:**
  ```bash
  python3 -m viralStoryGenerator.src.worker_runner worker --worker-type scrape
  ```
- **Queue Worker:**
  ```bash
  python3 -m viralStoryGenerator.src.worker_runner worker --worker-type queue
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

- **API & HTTP Service:**
  - Migrated from CLI-based to HTTP service architecture.
  - Implemented REST API endpoints in `src/api.py` for story generation and retrieval.
  - Added worker processes with Redis queue support in `src/api_worker.py`.

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

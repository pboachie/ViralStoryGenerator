FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy requirements file
COPY requirements-api.txt .

# Install build dependencies, then Python packages, then remove build dependencies
RUN apt-get update &&     apt-get install -y --no-install-recommends         build-essential         python3-dev         libxml2-dev         libxslt-dev         curl         wget         gnupg &&     pip install --no-cache-dir -r requirements-api.txt &&     apt-get purge -y --auto-remove         build-essential         python3-dev         libxml2-dev         libxslt-dev         curl         wget         gnupg &&     rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# Default command runs the API server
CMD ["python3", "-m", "viralStoryGenerator.src.worker_runner", "api", "--host", "0.0.0.0", "--port", "8000"]

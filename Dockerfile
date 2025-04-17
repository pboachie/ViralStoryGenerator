FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    wget \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_ENABLED=True

# RUN mkdir -p /app/vector_db && chown -R <user>:<group> /app/vector_db # Might be needed depending on base image user

# Expose the port the app runs on
EXPOSE 8000

# Default command runs the API server
CMD ["python", "-m", "viralStoryGenerator", "api", "--host", "0.0.0.0", "--port", "8000"]
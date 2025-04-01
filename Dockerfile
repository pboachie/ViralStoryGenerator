FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy the .env file to the container
COPY .env .env

# Ensure the .env file is loaded by the application
ENV DOTENV_PATH=.env

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_ENABLED=True

# Expose the port the app runs on
EXPOSE 8000

# Default command runs the API server
CMD ["python", "-m", "viralStoryGenerator", "api", "--host", "0.0.0.0", "--port", "8000"]
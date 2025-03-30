FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_ENABLED=True

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "viralStoryGenerator.main"]
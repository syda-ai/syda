FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Fix line endings and make entrypoint script executable
RUN dos2unix docker/entrypoint.sh && \
    chmod +x docker/entrypoint.sh

# Expose port
EXPOSE 8000

# Use entrypoint script (runs migrations before starting server)
ENTRYPOINT ["docker/entrypoint.sh"]

# Default command (can be overridden in docker-compose.yml)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


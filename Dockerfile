FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Logs directory (will be mounted as a volume in production)
RUN mkdir -p logs

# Default: run the scheduler
CMD ["python", "scripts/run_scheduler.py"]

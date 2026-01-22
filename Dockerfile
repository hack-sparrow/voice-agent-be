# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing and other requirements
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Download required files
RUN python main.py download-files

# Set environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1

# Expose port if needed (LiveKit agents typically connect to LiveKit server)
# Uncomment if your agent needs to expose a port
# EXPOSE 8080

# Run the application
CMD ["python", "main.py", "dev"]

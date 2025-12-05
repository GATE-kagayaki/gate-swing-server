# Base image for Python applications
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install OS dependencies required by MediaPipe, OpenCV, and FFmpeg
# Note: ffmpeg is needed for video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port Cloud Run will use
ENV PORT 8080
EXPOSE 8080

# Start Gunicorn server. 
# ★★★ -w 1 に修正: 起動時の負荷を下げ、タイムアウトを防ぐ ★★★
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]

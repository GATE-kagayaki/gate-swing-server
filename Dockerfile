# Base image for Python applications
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install OS dependencies required by MediaPipe, OpenCV, and FFmpeg
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

# Start Gunicorn server (★ ここが重要！)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]

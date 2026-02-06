FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV MEDIAPIPE_DISABLE_GPU=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libglib2.0-0 \
      libgl1 \
      libegl1-mesa \
      libgbm1 \
      libgl1-mesa-dri \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY templates ./templates

ENV PORT=8080
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "900", "server:app"]

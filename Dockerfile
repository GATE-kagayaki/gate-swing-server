FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ★ここ：server.py と templates を両方入れる
COPY server.py .
COPY templates ./templates

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 900 server:app

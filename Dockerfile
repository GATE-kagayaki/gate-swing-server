# ベースイメージ: Python 3.10
FROM python:3.10-slim-bookworm

# MediaPipe / OpenCV 用ライブラリ
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# Python依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ★ アプリ本体と templates をコピー（最重要）
COPY server.py .
COPY templates ./templates

# Cloud Run 設定
ENV PORT 8080
EXPOSE 8080

# 起動
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 900 server:app

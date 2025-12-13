# ベースイメージ: Python 3.10 (安定版 Bookworm)
FROM python:3.10-slim-bookworm

# 1. OSライブラリのインストール
# MediaPipe/OpenCVに必要な OpenGL (libgl1) と FFmpeg を入れます
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. 作業ディレクトリ
WORKDIR /app

# 3. Pythonライブラリ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリコード
COPY server.py .

# 5. 環境変数
ENV PORT 8080

# 6. 起動コマンド
# --timeout 900 (15分) に設定して、長い動画解析でも切断されないようにします
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 900 server:app

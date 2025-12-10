# FFmpegとPythonの依存関係を持つベースイメージを使用
# Debian 系のイメージは、FFmpeg やその他のライブラリをインストールしやすい
FROM python:3.10-slim-buster

# 1. 依存関係のインストール (FFmpeg)
# FFmpegをインストールするためにapt-getを使用
RUN apt-get update && apt-get install -y \
    ffmpeg \
    # MediaPipeの依存関係
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. 作業ディレクトリの設定
WORKDIR /app

# 3. Pythonの依存関係のコピーとインストール
# requirements.txtは、Flask, google-cloud-tasks, mediapipe などを含んでいるはずです。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードのコピー
COPY server.py .

# 5. 環境変数の設定 (Gunicornがポート8080で実行されることを想定)
ENV PORT 8080

# 6. コマンドの実行 (Gunicorn経由でFlaskアプリを起動)
# --workers 1 または 2 に設定し、動画解析のメモリを確保
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 server:app

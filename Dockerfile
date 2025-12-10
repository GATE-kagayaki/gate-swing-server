# FFmpegとPythonの依存関係を持つ、より安定したベースイメージ (Bookworm) を使用
FROM python:3.10-slim-bookworm

# 1. OS依存関係のインストール (FFmpegとMediaPipeの実行時ライブラリ)
# apt-get の非対話型フラグをセット
ENV DEBIAN_FRONTEND=noninteractive

# ビルド安定化のため、apt-get updateとinstallを単一のRUNレイヤーに統合し、依存関係を確実にインストールする
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. 作業ディレクトリの設定
WORKDIR /app

# 3. Pythonの依存関係のコピーとインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードのコピー
COPY server.py .

# 5. 環境変数の設定 (Gunicornがポート8080で実行されることを想定)
ENV PORT 8080

# 6. コマンドの実行 (Gunicorn経由でFlaskアプリを起動)
# --workers 1 または 2 に設定し、動画解析のメモリを確保
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 server:app

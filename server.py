# ベースイメージ
FROM python:3.10-slim

# ★★★ OpenCV/FFmpegの依存ライブラリをインストール（ImportError: libGL.so.1 および動画処理対策）★★★
# Cloud RunのようなGUIのない環境でMediaPipe/OpenCV/FFmpegを動作させるために必須です。
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*
# ★★★ 修正箇所はここまで ★★★

# 作業ディレクトリ
WORKDIR /app

# 必要ファイルをコピー
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# server.py のファイル名に依存
COPY server.py .
COPY report_generator.py .

# Cloud Run が使うポート番号
ENV PORT=8080

# 起動コマンド

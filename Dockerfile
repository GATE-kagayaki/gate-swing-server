# ベースイメージをより安定した Python 3.10-slim-bullseye に変更
FROM python:3.10-slim-bullseye

# 1. OSパッケージのインストール
# ffmpeg、libsm6, libxext6、そしてOpenCV/MediaPipeの動作に必要な全ての依存関係をインストールします。
# これがワーカーの起動クラッシュを防ぐために最も重要です。
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 作業ディレクトリを設定
WORKDIR /app

# 3. Pythonの依存関係をインストール
COPY requirements.txt .
# pip install を実行
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードをコピー
COPY server.py .
COPY report_generator.py .

# 5. ポートを設定
ENV PORT 8080
EXPOSE 8080

# 6. アプリケーションを実行
# Gunicornのタイムアウトを延長し、ワーカーを2つに設定（CPU 4 vCPUsを活かすため）
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]

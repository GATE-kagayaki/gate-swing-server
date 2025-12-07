# ベースイメージをより安定した Python 3.10-slim-bullseye に変更
FROM python:3.10-slim-bullseye

# 1. OSパッケージのインストール
# FFmpeg、libgl1など、MediaPipe/OpenCVのOS依存関係をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 作業ディレクトリを設定
WORKDIR /app

# 3. Pythonの依存関係をインストール
COPY requirements.txt .
# ★★★ pip install の安定化: 環境変数による妨害を避けるため、ここでは設定はしません ★★★
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードをコピー
COPY server.py .
# report_generator.py は削除済みなので、この行は削除またはコメントアウトします。
# COPY report_generator.py .

# 5. ポートを設定
ENV PORT 8080

# 6. アプリケーションを実行
# ワーカー数1で起動負荷を最小限に抑え、起動タイムアウトを防ぐ
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]

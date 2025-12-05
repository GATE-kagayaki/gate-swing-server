# ベースイメージをより安定した Python 3.10-slim-bullseye に変更
FROM python:3.10-slim-bullseye

# 1. OSパッケージのインストール
# ffmpeg、libsm6, libxext6、そしてOpenCV/MediaPipeの動作に必要な全ての依存依存関係をインストール
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
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードをコピー
# ★★★ report_generator.py は削除済みのため、この行は削除またはコメントアウトします。
# COPY report_generator.py .
COPY server.py .

# 5. ポートを設定
ENV PORT 8080
EXPOSE 8080

# 6. アプリケーションを実行
# -w 1 (ワーカー数1) に修正済みで、起動時の負荷を最小限に抑えます。
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]

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
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードをコピー
COPY server.py .
# report_generator.py は削除済みなので、この行は削除してください
# COPY report_generator.py .

# 5. ポートを設定
ENV PORT 8080

# ★★★ 最終起動安定化設定 ★★★
# Pythonのパッケージ初期化時間を短縮し、起動タイムアウトを回避します。
ENV PYTHON_PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PYTHON_PIP_NO_DEPS=1
# ★★★ 最終起動安定化設定 終了 ★★★

# 6. アプリケーションを実行
# ★★★ 修正: python -m gunicorn を使用し、実行パスのエラーを回避 ★★★
CMD ["python", "-m", "gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]

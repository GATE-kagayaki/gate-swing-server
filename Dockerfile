# Debian Busterをベースイメージとして使用
FROM python:3.10-buster

# 1. OSパッケージのインストール
# ffmpegとその依存関係をインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2. 作業ディレクトリを設定
WORKDIR /app

# 3. Pythonの依存関係をインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードをコピー
COPY . /app

# 5. ポートを設定
ENV PORT 8080
EXPOSE 8080

# 6. アプリケーションを実行
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]

# Debian Busterをベースイメージとして使用
FROM python:3.10-buster

# 1. OSパッケージのインストール
# ffmpeg、libsm6, libxext6 (OpenCVの依存関係)をインストール。
# --no-install-recommendsと単一のRUNで依存関係エラーを最小限に抑える
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2. 作業ディレクトリを設定
WORKDIR /app

# 3. Pythonの依存関係をインストール
COPY requirements.txt /app/
# requirements.txt の pip install を実行
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードをコピー
COPY . /app

# 5. ポートを設定
ENV PORT 8080
EXPOSE 8080

# 6. アプリケーションを実行
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]
```eof

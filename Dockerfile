# ベースイメージ
FROM python:3.10-slim

# ★★★ OpenCVの依存ライブラリをインストール（ImportError: libGL.so.1 対策）★★★
# Cloud RunのようなGUIのない環境でMediaPipe/OpenCVを動作させるために必須です。
RUN apt-get update && apt-get install -y \
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

COPY server.py .
COPY report_generator.py .

# Cloud Run が使うポート番号
ENV PORT=8080

# 起動コマンド
# server:app は、server.pyファイル内の Flask インスタンス(app)を指定しています。
CMD ["gunicorn", "-b", "0.0.0.0:8080", "server:app"]

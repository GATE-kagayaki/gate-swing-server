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
# ★★★ 環境変数を削除したため、ここでのインストールは成功します ★★★
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードをコピー
COPY server.py .
# report_generator.py は削除済みなので、この行は削除してください
# COPY report_generator.py .

# 5. ポートを設定
ENV PORT 8080

# ★★★ 起動安定化設定（Python ENV）は削除 ★★★
# これらの環境変数はpipのインストールを妨害するため、削除しました。

# 6. アプリケーションを実行
# ★★★ 修正: gunicorn の実行を単純なパス呼び出しに戻します ★★★
# Pythonのパスが正常になったため、これでgunicornが認識されます。
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "server:app", "--timeout", "900"]

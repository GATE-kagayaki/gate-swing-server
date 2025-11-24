# ベースイメージ
FROM python:3.10-slim

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
CMD ["gunicorn", "-b", "0.0.0.0:8080", "server:app"]


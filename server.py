import os
import datetime
import traceback

from flask import Flask, request, jsonify
import requests
from google.cloud import storage

# PDF生成用
# 期待インターフェース:
#   generate_pdf_report(output_path: str, video_url: str) -> str
#       - output_path に PDF を生成し、そのパスを返す
#   upload_to_gcs(local_path: str, bucket_name: str, dest_blob_name: str) -> str
#       - local_path のファイルを GCS の bucket_name/dest_blob_name にアップロードし、
#         アクセス用の URL (公開または署名付き) を返す
from report_generator import generate_pdf_report, upload_to_gcs

app = Flask(__name__)

LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

if not LINE_ACCESS_TOKEN:
    raise RuntimeError("環境変数 LINE_CHANNEL_ACCESS_TOKEN が設定されていません。")
if not BUCKET_NAME:
    raise RuntimeError("環境変数 GCS_BUCKET_NAME が設定されていません。")

# GCS クライアント
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


# ---------------------------------------------------
# LINE 返信ヘルパ
# ---------------------------------------------------
def reply_text(reply_token: str, text: str) -> None:
    """テキストメッセージで返信する"""
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
    }
    body = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}],
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        # 返信に失敗してもサーバー自体は 200 を返したいのでここはログのみ
        print("ERROR in reply_text:", e)
        traceback.print_exc()


# ---------------------------------------------------
# GCS 署名付きURL生成ヘルパ (必要なら使用)
# ---------------------------------------------------
def generate_signed_url(blob, expiration_minutes: int = 60) -> str:
    """指定した blob の署名付きURLを生成"""
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiration_minutes),
        method="GET",
    )
    return url


# ---------------------------------------------------
# 動画ダウンロード (stream)
# ---------------------------------------------------
def download_video_to_gcs(message_id: str) -> str:
    """
    LINE の message_id から動画バイナリを取得し、
    GCS に mp4 として保存する。

    戻り値:
        video_gcs_url: GCS からアクセス可能な URL (署名付きURLや公開URL)
    """
    content_url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_ACCESS_TOKEN}"}

    file_name = f"videos/video_{message_id}.mp4"
    blob = bucket.blob(file_name)

    # ストリーミングで GCS に書き込み
    try:
        with requests.get(content_url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            # blob.open("wb") で直接書き込み
            with blob.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print("ERROR in download_video_to_gcs:", e)
        traceback.print_exc()
        raise

    # セキュリティ要件に応じて以下のどちらかを選択

    # 1. 署名付きURL (推奨)
    video_url = generate_signed_url(blob, expiration_minutes=60)

    # 2. バケットを公開運用している場合は make_public も可能
    # blob.make_public()
    # video_url = blob.public_url

    print("Video stored to GCS:", file_name, "URL:", video_url)
    return video_url


# ---------------------------------------------------
# Webhook エンドポイント
# ---------------------------------------------------
@app.route("/callback", methods=["POST"])
def callback():
    try:
        body = request.get_json(force=True, silent=True) or {}
        events = body.get("events", [])

        print("RAW EVENT:", body)

        for event in events:
            # イベントごとに例外をキャッチして他のイベントに影響させない
            try:
                handle_event(event)
            except Exception as e:
                print("ERROR in handle_event:", e)
                traceback.print_exc()
                # エラー時もユーザーには一応返信しておく
                reply_token = event.get("replyToken")
                if reply_token:
                    reply_text(
                        reply_token,
                        "処理中にエラーが発生しました。\n時間をおいてもう一度お試しください。",
                    )

        # LINE 側には 200 を返せば OK
        return "OK", 200

    except Exception as e:
        print("ERROR in callback root:", e)
        traceback.print_exc()
        # ここで 500 を返すと LINE にリトライされる
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# イベント単位の処理
# ---------------------------------------------------
def handle_event(event: dict) -> None:
    """1件の LINE イベントを処理"""
    if event.get("type") != "message":
        # ここでは message イベントのみ処理
        return

    msg = event.get("message", {})
    reply_token = event.get("replyToken")

    if not reply_token:
        print("No replyToken in event:", event)
        return

    msg_type = msg.get("type")

    # テキストメッセージ
    if msg_type == "text":
        user_text = msg.get("text", "")
        reply_text(reply_token, f"受信: {user_text}")
        return

    # 動画メッセージ
    if msg_type == "video":
        reply_text(reply_token, "動画を受信しました。解析レポートを作成中です…")

        message_id = msg.get("id")
        if not message_id:
            reply_text(reply_token, "動画 ID が取得できませんでした。")
            return

        # 1. 動画を GCS に保存
        video_url = download_video_to_gcs(message_id)

        # 2. PDF生成
        #    report_generator.generate_pdf_report は
        #    generate_pdf_report("/tmp/report.pdf", video_url)
        #    のような形で実装されている想定
        pdf_local_path = "/tmp/report.pdf"
        pdf_path = generate_pdf_report(pdf_local_path, video_url)

        # 3. PDFをGCSにアップロード
        #    upload_to_gcs(pdf_path, bucket_name, object_name)
        pdf_object_name = f"reports/{message_id}.pdf"
        pdf_url = upload_to_gcs(pdf_path, BUCKET_NAME, pdf_object_name)

        # 4. 完成通知
        reply_text(reply_token, f"レポートが完成しました👇\n{pdf_url}")
        return

    # その他のメッセージタイプ
    reply_text(
        reply_token,
        "現在サポートしているのはテキストと動画のみです。",
    )


# ---------------------------------------------------
# ヘルスチェック用
# ---------------------------------------------------
@app.route("/upload_test", methods=["POST"])
def upload_test():
    return {"message": "upload test OK"}, 200

# ---------------------------------------------------
# Cloud Run 起動
# ---------------------------------------------------
if __name__ == "__main__":
    # ローカル開発用
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

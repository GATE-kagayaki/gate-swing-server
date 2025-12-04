import os
import datetime
import traceback

from flask import Flask, request, jsonify
import requests
from google.cloud import storage

from report_generator import generate_report_for_line

app = Flask(__name__)

# -------------------------
# 環境変数
# -------------------------
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
# LINE返信（テキスト）
# ---------------------------------------------------
def reply_text(reply_token: str, text: str):
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
        print("LINE reply response:", resp.status_code, resp.text)
        resp.raise_for_status()
    except Exception as e:
        print("ERROR in reply_text:", e)
        traceback.print_exc()


# ---------------------------------------------------
# 署名付きURL生成（動画アクセス用）
# ---------------------------------------------------
def generate_signed_url(blob, expiration_minutes=60):
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiration_minutes),
        method="GET",
    )
    return url


# ---------------------------------------------------
# LINE動画 → GCS保存
# ---------------------------------------------------
def download_video_to_gcs(message_id: str) -> str:
    content_url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_ACCESS_TOKEN}"}

    file_name = f"videos/{message_id}.mp4"
    blob = bucket.blob(file_name)

    try:
        with requests.get(content_url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            with blob.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    except Exception as e:
        print("ERROR in download_video_to_gcs:", e)
        traceback.print_exc()
        raise

    video_url = generate_signed_url(blob)
    print("Video saved:", file_name, "URL:", video_url)
    return video_url


# ---------------------------------------------------
# Webhook
# ---------------------------------------------------
@app.route("/callback", methods=["POST"])
def callback():
    try:
        body = request.get_json(force=True, silent=True) or {}
        print("=== RAW EVENT START ===")
        print(body)
        print("=== RAW EVENT END ===")

        events = body.get("events", [])
        for event in events:
            try:
                handle_event(event)
            except Exception as e:
                print("ERROR in handle_event:", e)
                traceback.print_exc()
                rt = event.get("replyToken")
                if rt:
                    reply_text(rt, "処理中にエラーが発生しました。もう一度お試しください。")

        return "OK", 200

    except Exception as e:
        print("ERROR in callback root:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# イベント別処理
# ---------------------------------------------------
def handle_event(event: dict):
    if event.get("type") != "message":
        return

    msg = event.get("message", {})
    reply_token = event.get("replyToken")
    if not reply_token:
        print("No replyToken in event")
        return

    msg_type = msg.get("type")

    # -------------------------
    # テキストメッセージ
    # -------------------------
    if msg_type == "text":
        user_text = msg.get("text", "")
        reply_text(reply_token, f"受信: {user_text}")
        return

    # -------------------------
    # 動画メッセージ
    # -------------------------
    if msg_type == "video":
        reply_text(reply_token, "動画を受信しました。解析レポートを作成中です…")

        message_id = msg.get("id")
        if not message_id:
            reply_text(reply_token, "動画IDが取得できませんでした。")
            return

        # ①動画保存（将来Bレベル解析用）
        video_url = download_video_to_gcs(message_id)

        # ②ざっくり解析 → AIレポート生成
        #   ※クラブ種別＆レベルは後でupgrade
        club_type = "ドライバー"
        user_level = "初心者"

        report_text = generate_report_for_line(
            mode="free",      # 有料にしたい場合は "paid"
            club_type=club_type,
            user_level=user_level
        )

        # ③レポート送信
        # （長すぎたら自動分割したいが、まずは1発送信）
        reply_text(reply_token, report_text)
        return

    # -------------------------
    # その他タイプ
    # -------------------------
    reply_text(reply_token, "現在サポートしているのはテキストと動画のみです。")


# ---------------------------------------------------
# 動作確認用
# ---------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

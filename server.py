from flask import Flask, request, jsonify
import os
import requests
from openai import OpenAI
from google.cloud import storage
import tempfile

app = Flask(__name__)

# ----------------------------
# OpenAI API クライアント
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# LINE アクセストークン
# ----------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# ----------------------------
# Cloud Storage 設定
# ----------------------------
GCS_BUCKET_NAME = "gate-swing-data"  # ←あなたのバケット名
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)


# ----------------------------
# LINE 返信処理
# ----------------------------
def reply(reply_token, message):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    data = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": message}]
    }
    requests.post(url, headers=headers, json=data)


# ----------------------------
# Cloud Storage に動画を保存
# ----------------------------
def save_video_to_gcs(file_content, file_name):
    blob = bucket.blob(file_name)
    blob.upload_from_string(file_content)
    blob.make_public()  # 公開URLにする
    return blob.public_url


# ----------------------------
# ホーム
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return "GATE Swing Server is running."


# ----------------------------
# LINE Webhook
# ----------------------------
@app.route("/callback", methods=["POST"])
def callback():
    try:
        body = request.get_json()
        events = body.get("events", [])

        for event in events:
            if event.get("type") == "message":

                msg_type = event["message"]["type"]

                # -------------------
                # テキストメッセージ
                # -------------------
                if msg_type == "text":
                    reply_token = event["replyToken"]
                    user_message = event["message"].get("text", "")
                    reply(reply_token, f"受け取りました：{user_message}")

                # -------------------
                # 動画メッセージ
                # -------------------
                elif msg_type == "video":
                    reply_token = event["replyToken"]

                    # content URL
                    content_url = f"https://api.line.me/v2/bot/message/{event['message']['id']}/content"

                    # 動画データ取得
                    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
                    video_data = requests.get(content_url, headers=headers).content

                    # ファイル名
                    file_name = f"video_{event['message']['id']}.mp4"

                    # Cloud Storage に保存
                    video_url = save_video_to_gcs(video_data, file_name)

                    # LINEに返信
                    reply(reply_token, "動画を受け取りました！AI解析中です…")

                    print("Saved video:", video_url)

        return "OK", 200

    except Exception as e:
        print("Error:", e)
        return "Error", 500


# ----------------------------
# 独自API（AI解析）
# ----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message received"}), 400

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたはプロのゴルフスイングコーチです。"},
                {"role": "user", "content": user_message}
            ]
        )

        answer = response.choices[0].message["content"]
        return jsonify({"reply": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Cloud Run
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

from flask import Flask, request, jsonify
import os
import requests
from openai import OpenAI

app = Flask(__name__)

# OpenAI API クライアント
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LINE アクセストークン
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# -------- LINE 返信処理 --------
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

# -------- ホーム --------
@app.route("/", methods=["GET"])
def home():
    return "GATE Swing Server is running."

# -------- LINE Webhook --------
@app.route("/callback", methods=["POST"])
def callback():
    try:
        body = request.get_json()

        events = body.get("events", [])
        for event in events:
            if event.get("type") == "message" and "message" in event:
                reply_token = event["replyToken"]
                user_message = event["message"].get("text", "")

                reply(reply_token, f"受け取りました：{user_message}")

        return "OK", 200

    except Exception as e:
        print("Error:", e)
        return "Error", 500

# -------- 独自 API（解析など） --------
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


# -------- Cloud Run --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


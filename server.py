from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

# OpenAI API キー（環境変数から取得）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "GATE Swing Server is running."
@app.route("/callback", methods=["POST"])
def callback():
    return "OK", 200

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message received"}), 400

        # GPT で応答生成
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


if __name__ == "__main__":
    # Cloud Run 必須：PORT=8080
    app.run(host="0.0.0.0", port=8080)


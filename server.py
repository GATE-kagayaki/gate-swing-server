# server.py

import os
import tempfile
import traceback

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, VideoMessage, TextSendMessage

from google.cloud import storage

from report_generator import generate_report_for_line


app = Flask(__name__)

# ======== 環境変数 =========
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# LINE API クライアント
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# GCS クライアント
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)


# ============================
# 1. Webhook 入口
# ============================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")

    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


# ============================
# 2. メッセージイベント
# ============================

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):

    text = event.message.text.strip()

    # 有料/無料の切り替え
    if text.lower() in ["paid", "有料"]:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage("有料版モードに切り替えました！\nこの後、スイング動画を送ってください。")
        )
        return

    if text.lower() in ["free", "無料"]:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage("無料版モードに切り替えました！\nこの後、スイング動画を送ってください。")
        )
        return

    # その他メッセージ
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage("スイング動画を送ってください！\n無料版＝07だけ\n有料版＝01〜10 全レポート作成します。")
    )


@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):

    # まず返信（処理中メッセージ）
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage("動画を受信しました。解析レポートを作成中です…")
    )

    try:
        # ========================
        # 1. 動画を一時保存
        # ========================
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            message_content = line_bot_api.get_message_content(event.message.id)
            for chunk in message_content.iter_content():
                tmp.write(chunk)
            tmp_path = tmp.name

        # ========================
        # 2. GCS に保存
        # ========================
        gcs_path = f"videos/{os.path.basename(tmp_path)}.mp4"
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(tmp_path)
        blob.make_public()
        video_url = blob.public_url

        # ========================
        # 3. 解析（stub → 生成AIレポート）
        #     - 初期は無料版 "free"
        #     - あなたの仕様では今は固定 free でOK
        # ========================
        mode = "paid" if "paid" in (event.source.user_id or "").lower() else "free"

        report_text = generate_report_for_line(
            mode="free",    # ←あとで paid 切り替え可
            club_type="ドライバー",
            user_level="初心者"
        )

        # ========================
        # 4. ユーザーへ送信
        # ========================
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(report_text)
        )

    except Exception as e:
        print("Error:", e)
        print(traceback.format_exc())

        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(
                "レポート生成中にエラーが発生しました。\n\n"
                f"{e}"
            )
        )


@app.route("/", methods=["GET"])
def index():
    return "GATE Swing Server Running."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

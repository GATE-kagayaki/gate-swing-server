import os
import tempfile
from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    VideoMessage,
    TextSendMessage,
)

from google.cloud import storage

from report_generator import generate_report_for_line

# ----------------------------------------------------
# 1. 環境変数
# ----------------------------------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise Exception("LINE の環境変数が正しく設定されていません。")

if not GCS_BUCKET_NAME:
    raise Exception("GCS_BUCKET_NAME が設定されていません。")


# ----------------------------------------------------
# 2. LINE SDK 初期化
# ----------------------------------------------------
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# ----------------------------------------------------
# 3. Flask
# ----------------------------------------------------
app = Flask(__name__)


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")

    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


# ----------------------------------------------------
# 4. メッセージ受信
# ----------------------------------------------------

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """
    テキストメッセージを受信したとき
    """
    text = event.message.text

    # シンプルな応答
    reply = f"「{text}」を受け取りました。動画を送ってください。"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    """
    動画メッセージを受信したとき
    """
    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="動画を受信しました。解析レポートを作成中です…")
        )
    except:
        pass  # 念のため

    try:
        # --------------------------------------------------------
        # ① 動画データを一時保存
        # --------------------------------------------------------
        message_content = line_bot_api.get_message_content(event.message.id)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for chunk in message_content.iter_content():
                tmp.write(chunk)

            tmp_path = tmp.name

        # --------------------------------------------------------
        # ② GCS にアップロード（ACL操作しない）
        # --------------------------------------------------------
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)

        gcs_path = f"videos/{os.path.basename(tmp_path)}.mp4"
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(tmp_path)

        # Uniform bucket access のため ACL変更は禁止
        # signed URL も作らない
        video_url = f"gs://{GCS_BUCKET_NAME}/{gcs_path}"

        # --------------------------------------------------------
        # ③ レポート生成（無料固定）
        # --------------------------------------------------------
        report_text = generate_report_for_line(
            mode="free",
            club_type="ドライバー",
            user_level="初心者"
        )

        # --------------------------------------------------------
        # ④ LINE へ返信
        # --------------------------------------------------------
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=report_text)
        )

    except Exception as e:
        # エラーがあっても必ず返信
        err_msg = f"レポート生成中にエラーが発生しました。\n{str(e)}"
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=err_msg)
        )


# ----------------------------------------------------
# 5. Cloud Run 用エントリポイント
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

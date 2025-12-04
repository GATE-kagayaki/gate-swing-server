import os
import datetime
import traceback

from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextSendMessage, VideoMessage

from report_generator import generate_report_for_line

# ------------- LINE 設定 -------------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Flask app
app = Flask(__name__)

# -----------------------------------
# 1. Webhook エンドポイント（LINE）
# -----------------------------------
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get("X-Line-Signature")

    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        print("Callback Error:", e)
        traceback.print_exc()
        return "Error", 400

    return "OK"


# -----------------------------------
# 2. メッセージ受信（動画 or 文字）
# -----------------------------------
@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    """
    ユーザーが動画を送ったときに呼ばれる部分。
    今は映像解析はしないため、ダミーのAレベル分析を実行して
    完成されたレポートを返す。
    """
    user_id = event.source.user_id

    # 今後ここに「動画をGCSへ保存 → 本物の解析」を入れられる
    # ----------------------------------------------------------

    # 暫定的にドライバー・初心者で仮定
    club_type = "ドライバー"
    user_level = "初心者"

    try:
        # 有料版レポート（あなたが指定したテンプレ構成）
        report_text = generate_report_for_line(
            mode="paid",
            club_type=club_type,
            user_level=user_level
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="動画を受信しました。解析レポートを作成します…")
        )

        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=report_text)
        )

    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"レポート生成中にエラーが発生しました。\n{e}")
        )
        traceback.print_exc()


@handler.add(MessageEvent)
def handle_text(event):
    """
    テキストメッセージが送信された場合の処理。
    """
    text = event.message.text

    if text in ["無料", "無料レポート"]:
        report_text = generate_report_for_line(
            mode="free",
            club_type="ドライバー",
            user_level="初心者"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=report_text))

    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="動画を送信してください📹")
        )


# -----------------------------------
# Cloud Run（デプロイ用）
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

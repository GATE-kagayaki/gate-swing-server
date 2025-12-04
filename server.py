import os
import tempfile
import logging
from typing import Optional

from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    VideoMessage,
)

from google.cloud import storage

from report_generator import generate_report


# ===== ログ設定 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== 環境変数 =====
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN が環境変数に設定されていません。")

if not GCS_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME が環境変数に設定されていません。")


line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)


# ===== Flask アプリ =====
app = Flask(__name__)


@app.get("/")
def health():
    """Cloud Run 用のヘルスチェック"""
    return "OK", 200


@app.post("/callback")
def callback():
    """LINE Webhook エンドポイント"""
    signature = request.headers.get("X-Line-Signature")

    body = request.get_data(as_text=True)
    logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid signature. Check your channel access token/secret.")
        abort(400)

    return "OK"


# ===== ユーザーのプラン判定ロジック =====

def get_plan_type(user_id: str) -> str:
    """
    ユーザーが無料か有料かを判定する。

    いまは暫定で「全員 free 」。
    Stripe / 会員DB と連携できるようになったら、
    ここだけを書き換えれば OK。

    戻り値:
        "free"  または "paid"
    """

    # TODO: ここをあとで Stripe / 会員管理システムと連携させる
    # 例:
    #   if is_paid_member_in_db(user_id):
    #       return "paid"
    #   else:
    #       return "free"

    return "free"


# ===== 動画処理ロジック =====

def upload_video_to_gcs(content: bytes, user_id: str) -> str:
    """
    受け取った動画バイト列を GCS にアップロードし、その URL を返す。
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(content)
    tmp.flush()
    tmp.close()

    blob_name = f"swings/{user_id}/{os.path.basename(tmp.name)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(tmp.name, content_type="video/mp4")

    # 後片付け
    os.remove(tmp.name)

    # 公開 URL（署名付き URL を使いたい場合は generate_signed_url に変える）
    blob.make_public()
    logger.info(f"Uploaded video to GCS: {blob.public_url}")
    return blob.public_url


def build_waiting_message(plan_type: str) -> str:
    """
    動画受信直後の「受付メッセージ」をプラン別に出し分け。
    """
    if plan_type == "paid":
        return (
            "動画を受信しました。\n"
            "有料プラン用の詳細レポートを作成中です…\n"
            "しばらくお待ちください。"
        )
    else:
        return (
            "動画を受信しました。\n"
            "無料版の簡易診断レポートを作成中です…"
        )


def split_message(text: str, chunk_size: int = 1800):
    """
    LINE の文字数制限に合わせて長文を分割するユーティリティ。
    """
    lines = text.splitlines()
    chunks = []
    current = ""

    for line in lines:
        # 行を追加しても chunk_size を超えないか確認
        if len(current) + len(line) + 1 <= chunk_size:
            current += (line + "\n")
        else:
            chunks.append(current.rstrip("\n"))
            current = line + "\n"

    if current:
        chunks.append(current.rstrip("\n"))

    return chunks


# ===== LINE イベントハンドラ =====

@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event: MessageEvent):
    user_id = event.source.user_id

    # 1) プラン判定（今は全員 free）
    plan_type = get_plan_type(user_id)
    logger.info(f"User {user_id} plan_type = {plan_type}")

    # 2) 受付メッセージを先に返す
    waiting_msg = build_waiting_message(plan_type)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=waiting_msg)
    )

    try:
        # 3) 動画データを取得
        message_id = event.message.id
        message_content = line_bot_api.get_message_content(message_id)
        video_bytes = b"".join(chunk for chunk in message_content.iter_content())

        # 4) GCS にアップロード
        video_url = upload_video_to_gcs(video_bytes, user_id)

        # 5) レポート生成（無料:07だけ / 有料:01〜10）
        logger.info(f"Generating report for user {user_id}, plan={plan_type}")
        report_text = generate_report(video_url, plan_type)

        # 6) 長文を分割してプッシュメッセージで送信
        chunks = split_message(report_text)

        messages = [TextSendMessage(text=chunk) for chunk in chunks]
        line_bot_api.push_message(to=user_id, messages=messages)

        logger.info(f"Report sent to user {user_id}.")
    except Exception as e:
        logger.exception("Error while processing video message.")
        # エラーが起きた場合もユーザーに一言返しておく
        line_bot_api.push_message(
            to=user_id,
            messages=TextSendMessage(
                text="レポートの作成中にエラーが発生しました。時間をおいて、もう一度お試しください。"
            ),
        )


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    """
    テキストメッセージが来たときの簡易応答。
    将来的に「料金プランの確認」などを入れる余地。
    """
    text = event.message.text.strip()

    if text in ("メニュー", "menu", "ヘルプ"):
        reply = (
            "GATE スイング診断サービスです。\n\n"
            "・動画を送るとスイング診断レポートをお返しします。\n"
            "・現在は無料版では『要約診断（07）』のみ、\n"
            "・有料版では 01〜10 までの詳細レポートを提供予定です。"
        )
    else:
        reply = (
            "スイング動画を送っていただくと、AI が自動で解析してレポートをお返しします。\n"
            "ご不明な点があれば「メニュー」と送信してください。"
        )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ===== Cloud Run 用エントリポイント =====
if __name__ == "__main__":
    # ローカル動作用。Cloud Run では gunicorn 経由で起動する想定。
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

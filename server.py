import os
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, abort, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, VideoMessage, TextSendMessage

# ★★★ 環境変数の設定 ★★★
# Cloud Runの環境変数に設定することを前提とします
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

app = Flask(__name__)

# LINE Bot APIとハンドラーの初期化
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ----------------------------------------------------
# 1. LINEからのWebhookを受け取るエンドポイント
# ----------------------------------------------------
@app.route("/callback", methods=['POST'])
def callback():
    # リクエストヘッダーから署名検証用ヘッダーを取得
    signature = request.headers['X-Line-Signature']

    # リクエストボディを取得
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # 署名検証とイベントハンドリング
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        # 署名検証エラーの場合、400を返す
        print("Invalid signature. Please check your channel access token/secret.")
        abort(400)

    return 'OK'

# ----------------------------------------------------
# 2. 動画メッセージを受信した時の処理
# ----------------------------------------------------
@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    message_id = event.message.id
    user_id = event.source.user_id
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="動画を受け付けました。解析を開始します。しばらくお待ちください...")
    )
    
    # ★★★ 動画のダウンロードと解析の実行 ★★★
    
    # 1. LINEサーバーから動画をダウンロード
    try:
        message_content = line_bot_api.get_message_content(message_id)
        
        # 2. 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name
            for chunk in message_content.iter_content():
                tmp_file.write(chunk)
            
        # 3. MediaPipe解析を実行
        # （ここでは骨格データ抽出のみのダミー関数を呼び出し）
        # TODO: report_generator.pyの関数を呼び出す
        # raw_landmarks = analyze_media_pipe(video_path) 
        
        # 4. 解析結果からレポートを作成
        # final_report = generate_report(raw_landmarks) 

        # 5. 結果をユーザーに返信
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=f"解析が完了しました。ユーザーID:{user_id}のレポートをLINEに返します。")
        )
        
        # 6. 後処理
        os.unlink(video_path)
        
    except Exception as e:
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=f"エラーが発生しました: {e}")
        )
        print(f"Error: {e}")

# ----------------------------------------------------
# 3. テキストメッセージを受信した時の処理 (オマケ)
# ----------------------------------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"動画を送ってくださいね。'{event.message.text}'というメッセージを確認しました。")
    )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    # 実行するファイル名を server_line.py に変更した場合、
    # gunicorn の起動コマンドも CMD ["gunicorn", "-b", "0.0.0.0:8080", "server_line:app"] に変更が必要です。
    app.run(host='0.0.0.0', port=port)

import os
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, VideoMessage, TextSendMessage
)

# ---------------------------------------------------------
# 環境変数の設定 (Cloud Runの環境変数設定で入力します)
# ---------------------------------------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# Flaskアプリケーションの初期化
app = Flask(__name__)

# LINE Bot API設定
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# MediaPipe設定
mp_pose = mp.solutions.pose

# ---------------------------------------------------------
# 1. LINE Webhook エンドポイント
# ---------------------------------------------------------
@app.route("/callback", methods=['POST'])
def callback():
    # 署名の検証
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# ---------------------------------------------------------
# 2. 動画メッセージを受け取った時の処理
# ---------------------------------------------------------
@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    message_id = event.message.id
    reply_token = event.reply_token
    user_id = event.source.user_id

    # とりあえず「解析開始」を伝える
    try:
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text="動画を受け取りました⛳️\nスイング解析を開始します...")
        )
    except Exception as e:
        print(f"Reply Error: {e}")

    # 動画のダウンロードと解析実行
    try:
        # LINEサーバーから動画コンテンツを取得
        message_content = line_bot_api.get_message_content(message_id)
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name
            for chunk in message_content.iter_content():
                tmp_file.write(chunk)

        # -----------------------------------------------
        # ★ ここで骨格解析を実行
        # -----------------------------------------------
        landmarks_data = process_video_with_mediapipe(video_path)
        
        # 解析結果に基づいてレポートを作成（仮のメッセージ）
        # 将来的には report_generator.py を呼び出す場所です
        report_text = f"解析完了！\n合計フレーム数: {len(landmarks_data)}\n\n(現在は骨格抽出まで完了しています。ここにスイング診断結果が表示されます)"

        # 結果をPUSHメッセージで送信（reply_tokenは1回しか使えないため）
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=report_text)
        )

    except Exception as e:
        error_msg = f"システムエラーが発生しました: {e}"
        print(error_msg)
        line_bot_api.push_message(user_id, TextSendMessage(text=error_msg))
    
    finally:
        # 一時ファイルの削除
        if os.path.exists(video_path):
            os.remove(video_path)

# ---------------------------------------------------------
# 3. MediaPipe解析ロジック
# ---------------------------------------------------------
def process_video_with_mediapipe(video_path):
    """
    動画から骨格ランドマークを抽出する関数
    """
    landmarks_data = []
    
    cap = cv2.VideoCapture(video_path)
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose: # Cloud RunなどCPU環境ではmodel_complexity=1 (or 0) が軽量でおすすめ
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 高速化のため書き込み不可モード＆RGB変換
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image)
            
            if results.pose_landmarks:
                # 必要なデータだけ抽出して保存
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                landmarks_data.append(frame_landmarks)
                
    cap.release()
    return landmarks_data

# ---------------------------------------------------------
# サーバー起動
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

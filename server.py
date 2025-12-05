import os
import threading # 非同期処理のため、これだけはトップレベルに残す
import tempfile
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, VideoMessage
# ★★★ 全ての重いインポート（numpyを含む）を削除しました ★★★

# 環境変数の設定
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set")

# FlaskアプリとLINE Bot APIの設定
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ------------------------------------------------
# 解析ロジック (analyze_swing) - 起動時タイムアウトを避けるため、ここに統合
# ------------------------------------------------
def analyze_swing(video_path):
    """
    動画を解析し、スイングの評価レポート（テキスト）を返します。
    """
    # ★★★ 全ての重いライブラリをここでインポートする ★★★
    import cv2
    import mediapipe as mp
    import numpy as np

    # ここに calculate_angle 関数を定義
    def calculate_angle(p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        v1 = p1 - p2
        v2 = p3 - p2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    # ----------------------------------------------
    
    mp_pose = mp.solutions.pose
    max_shoulder_rotation = -180
    min_hip_rotation = 180
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "【エラー】動画ファイルを開けませんでした。"

    frame_count = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # 画像処理
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True

            frame_count += 1
            
            if results.pose_landmarks:
                # 解析ロジック (省略) - ランドマーク抽出と角度計算
                landmarks = results.pose_landmarks.landmark
                # ... (簡略化された解析ロジックをここに続行)
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                # numpy関数を直接使用
                max_shoulder_rotation = np.degrees(np.arctan2(r_ear[1] - r_shoulder[1], r_ear[0] - r_shoulder[0]))
                
    cap.release()
    
    # レポート作成ロジック (簡略化)
    report = f"""
✅ スイング診断レポート (起動安定版) ✅
（解析動画フレーム数: {frame_count}）
----------------------------------
🏌️ 最大回転 (簡略化): {max_shoulder_rotation:.1f} 度
"""
    return report

# ------------------------------------------------
# メインの解析ロジックを別スレッドで実行する関数
# ------------------------------------------------
def process_video_async(user_id, video_content):
    """
    動画のダウンロード、圧縮、解析、レポート送信をバックグラウンドで実行します。
    """
    # ★★★ ここでrequests, ffmpegをインポートする ★★★
    import requests
    import ffmpeg
    
    original_video_path = None
    compressed_video_path = None
    
    # 1. オリジナル動画を一時ファイルに保存
    try:
        with tempfile.NamedTemporaryFile(suffix="_original.mp4", delete=False) as tmp_file:
            original_video_path = tmp_file.name
            tmp_file.write(video_content)
        app.logger.info(f"オリジナル動画ファイル保存成功: {original_video_path}")
    except Exception as e:
        app.logger.error(f"動画ファイルの保存に失敗: {e}", exc_info=True)
        return

    # 1.5 動画の自動圧縮とリサイズ処理
    try:
        compressed_video_path = tempfile.NamedTemporaryFile(suffix="_compressed.mp4", delete=False).name
        app.logger.info(f"動画を幅 640px に圧縮・変換開始。")
        
        # FFmpegで圧縮とリサイズを実行
        (
            ffmpeg
            .input(original_video_path)
            .output(compressed_video_path, vf='scale=640:-1', crf=28, vcodec='libx264')
            .overwrite_output()
            .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True) 
        )
        video_to_analyze = compressed_video_path
        app.logger.info(f"動画圧縮・変換成功: {compressed_video_path}")
        
    except ffmpeg.Error as e:
        error_details = e.stderr.decode('utf8') if e.stderr else '詳細不明'
        app.logger.error(f"FFmpegによる動画圧縮に失敗: {error_details}", exc_info=True)
        report_text = f"【動画処理エラー】圧縮に失敗しました。詳細: {error_details[:100]}..."
        line_bot_api.push_message(user_id, TextSendMessage(text=report_text))
        return
        
    except Exception as e:
        app.logger.error(f"予期せぬ圧縮エラー: {e}", exc_info=True)
        report_text = f"【予期せぬエラー】動画処理で問題が発生しました: {str(e)[:100]}..."
        line_bot_api.push_message(user_id, TextSendMessage(text=report_text))
        return
        
    # 2. 動画の解析を実行
    try:
        # analyze_swing 関数をこのファイル内で直接呼び出す
        report_text = analyze_swing(video_to_analyze)
    except Exception as e:
        report_text = f"【解析エラー】動画処理中に予期せぬエラーが発生しました: {e}"
        app.logger.error(f"解析中の致命的なエラー: {e}", exc_info=True)

    # 3. 結果をユーザーにPUSH通知で返信
    try:
        completion_message = "✅ 解析が完了しました！\nレポートを送信します。"
        line_bot_api.push_message(user_id, TextSendMessage(text=completion_message))
        
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=report_text)
        )
        app.logger.info(f"レポート送信成功: ユーザーID={user_id}")

    except LineBotApiError as e:
        app.logger.error(f"LINE APIエラー: Status={e.status_code}, Message={e.message}, Details={e.error_response}", exc_info=True)
    except Exception as e:
        app.logger.error(f"レポート送信中に予期せぬエラーが発生しました: {e}", exc_info=True)

    # 4. 一時ファイルを削除
    if original_video_path and os.path.exists(original_video_path):
        os.remove(original_video_path)
        app.logger.info(f"一時ファイル削除: {original_video_path}")
        
    if compressed_video_path and os.path.exists(compressed_video_path):
        os.remove(compressed_video_path)
        app.logger.info(f"一時ファイル削除: {compressed_video_path}")


# ------------------------------------------------
# LINE Webhookのメイン処理
# ------------------------------------------------
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("Invalid signature. Check your channel secret.")
        abort(400)
    except Exception as e:
        app.logger.error(f"Webhook handling error: {e}", exc_info=True)
        abort(500)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text in ["レポート", "テスト"]:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="動画を送信してください。ゴルフスイングの解析を行います。")
        )
        
@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event):
    user_id = event.source.user_id
    message_id = event.message.id

    # 1. ユーザーへの即時応答（LINEの応答タイムアウト回避）
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="動画を受け付けました。解析を開始します。しばらくお待ちください...")
    )
    
    # 2. 動画コンテンツの取得
    try:
        # requestsライブラリは処理内でインポートされるため、ここでは省略
        message_content = line_bot_api.get_message_content(message_id)
        video_content = message_content.content
    except Exception as e:
        app.logger.error(f"動画コンテンツの取得に失敗: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text="【エラー】動画のダウンロードに失敗しました。"))
        return

    # 3. 解析処理を別スレッドで起動（フリーズ回避）
    app.logger.info(f"動画解析を別スレッドで開始します。ユーザーID: {user_id}")
    thread = threading.Thread(target=process_video_async, args=(user_id, video_content))
    thread.start()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    # Cloud Runの起動安定化
    os.environ['HOME'] = '/tmp'
    app.run(host='0.0.0.0', port=port)

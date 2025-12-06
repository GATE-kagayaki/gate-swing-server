import os
import threading # 非同期処理のため必須 (処理のタイムアウト回避)
import tempfile 
import ffmpeg # 動画圧縮ライブラリ (メモリ不足回避のため必須)
import requests
import numpy as np 
from google import genai
from google.genai import types

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, VideoMessage

# 環境変数の設定
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set")

# FlaskアプリとLINE Bot APIの設定
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ------------------------------------------------
# 解析ロジック (analyze_swing) - 必須計測項目を全て実装
# ------------------------------------------------
def analyze_swing(video_path):
    """
    動画を解析し、スイングの評価レポート（テキスト）を返します。
    """
    # ★★★ 重いライブラリをここでインポートする (関数内インポート) ★★★
    import cv2
    import mediapipe as mp

    # 角度計算ヘルパー関数
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
    
    # 計測変数初期化
    max_shoulder_rotation = -180
    min_hip_rotation = 180
    head_start_x = None 
    max_head_drift_x = 0 
    max_wrist_cock = 0  
    
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
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True

            frame_count += 1
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 必須ランドマークの定義
                RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
                RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                RIGHT_EAR = mp_pose.PoseLandmark.RIGHT_EAR.value
                LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
                NOSE = mp_pose.PoseLandmark.NOSE.value
                RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
                RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                RIGHT_INDEX = mp_pose.PoseLandmark.RIGHT_INDEX.value

                # 座標抽出
                r_shoulder = [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y]
                r_ear = [landmarks[RIGHT_EAR].x, landmarks[RIGHT_EAR].y]
                l_hip = [landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y]
                r_hip = [landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y]
                nose = [landmarks[NOSE].x, landmarks[NOSE].y]
                r_wrist = [landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y]
                r_elbow = [landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y]
                r_index = [landmarks[RIGHT_INDEX].x, landmarks[RIGHT_INDEX].y]

                # 計測：最大肩回転
                shoulder_line_angle = np.degrees(np.arctan2(r_ear[1] - r_shoulder[1], r_ear[0] - r_shoulder[0]))
                if shoulder_line_angle > max_shoulder_rotation:
                    max_shoulder_rotation = shoulder_line_angle

                # 計測：最小腰回転
                hip_axis_x = l_hip[0] - r_hip[0]
                hip_axis_y = l_hip[1] - r_hip[1]
                current_hip_rotation = np.degrees(np.arctan2(hip_axis_y, hip_axis_x))
                if current_hip_rotation < min_hip_rotation:
                    min_hip_rotation = current_hip_rotation
                
                # 計測：頭の安定性
                if head_start_x is None:
                    head_start_x = nose[0]
                current_drift_x = abs(nose[0] - head_start_x)
                if current_drift_x > max_head_drift_x:
                    max_head_drift_x = current_drift_x
                    
                # 計測：手首のコック角
                if all(l is not None for l in [r_elbow, r_wrist, r_index]):
                    cock_angle = calculate_angle(r_elbow, r_wrist, r_index)
                    if cock_angle > max_wrist_cock:
                         max_wrist_cock = cock_angle
                
    cap.release()
    
    # 全ての計測結果を辞書で返す
    return {
        "frame_count": frame_count,
        "max_shoulder_rotation": max_shoulder_rotation,
        "min_hip_rotation": min_hip_rotation,
        "max_head_drift_x": max_head_drift_x,
        "max_wrist_cock": max_wrist_cock
    }

# ------------------------------------------------
# メインの解析ロジックを別スレッドで実行する関数
# ------------------------------------------------
def process_video_async(user_id, video_content):
    """
    動画のダウンロード、圧縮、解析、レポート送信をバックグラウンドで実行します。
    """
    import requests
    import ffmpeg
    
    original_video_path = None
    compressed_video_path = None
    
    # 1. オリジナル動画を一時ファイルに保存 (中略)
    try:
        with tempfile.NamedTemporaryFile(suffix="_original.mp4", delete=False) as tmp_file:
            original_video_path = tmp_file.name
            tmp_file.write(video_content)
    except Exception as e:
        app.logger.error(f"動画ファイルの保存に失敗: {e}", exc_info=True)
        return

    # 1.5 動画の自動圧縮とリサイズ処理 (メモリ不足回避のため必須)
    try:
        compressed_video_path = tempfile.NamedTemporaryFile(suffix="_compressed.mp4", delete=False).name
        (
            ffmpeg
            .input(original_video_path)
            .output(compressed_video_path, vf='scale=640:-1', crf=28, vcodec='libx264')
            .overwrite_output()
            .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True) 
        )
        video_to_analyze = compressed_video_path
        
    except Exception as e:
        app.logger.error(f"予期せぬ圧縮エラー: {e}", exc_info=True)
        report_text = f"【動画処理エラー】動画圧縮で問題が発生しました: {str(e)[:100]}..."
        line_bot_api.push_message(user_id, TextSendMessage(text=report_text))
        return
        
    # 2. 動画の解析を実行
    try:
        analysis_data = analyze_swing(video_to_analyze)
        
        # ★★★ AI診断の実行 - サービスロジックの中心 ★★★
        if GEMINI_API_KEY:
            ai_report_text = generate_full_member_advice(analysis_data) 
        else:
            # AIキーがない場合は無料会員相当の簡易レポートを生成
            ai_report_text = f"【無料会員】最大肩回転: {analysis_data['max_shoulder_rotation']:.1f}度\n詳細なAI診断レポートの生成にはGemini APIキーが必要です。"

        # 最終レポートを整形（有料会員向け詳細レポートを想定）
        report_text = f"⛳ GATEスイング診断 ⛳\n"
        report_text += "\n--- [ MediaPipe データ ] ---\n"
        report_text += f"フレーム数: {analysis_data['frame_count']}\n"
        report_text += f"最大肩回転: {analysis_data['max_shoulder_rotation']:.1f}度\n"
        report_text += f"最小腰回転: {analysis_data['min_hip_rotation']:.1f}度\n"
        report_text += f"頭の最大水平ブレ (0.001が最小): {analysis_data['max_head_drift_x']:.4f}\n"
        report_text += f"最大コック角 (180°が伸びた状態): {analysis_data['max_wrist_cock']:.1f}度\n"
        report_text += "\n--- [ AI 総合診断 ] ---\n"
        report_text += ai_report_text
        
    except Exception as e:
        report_text = f"【解析エラー】動画解析中に致命的なエラーが発生しました: {e}"
        app.logger.error(f"解析中の致命的なエラー: {e}", exc_info=True)

    # 3. 結果をユーザーにPUSH通知で返信 (中略)
    try:
        completion_message = "✅ 解析が完了しました！\n詳細レポートを送信します。"
        line_bot_api.push_message(user_id, TextSendMessage(text=completion_message))
        
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=report_text)
        )

    except Exception as e:
        app.logger.error(f"レポート送信中に予期せぬエラーが発生しました: {e}", exc_info=True)

    # 4. 一時ファイルを削除 (中略)
    if original_video_path and os.path.exists(original_video_path):
        os.remove(original_video_path)
    if compressed_video_path and os.path.exists(compressed_video_path):
        os.remove(compressed_video_path)

# ------------------------------------------------
# ★★★ Gemini API 呼び出し関数 (全項目網羅版) ★★★
# ------------------------------------------------
def generate_full_member_advice(analysis_data):
    """MediaPipeの数値結果をGemini APIに渡し、理想の10項目を網羅した詳細レポートを生成させる"""
    
    from google import genai
    from google.genai import types
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        return f"Geminiクライアント初期化失敗: {e}"
    
    shoulder_rot = analysis_data['max_shoulder_rotation']
    hip_rot = analysis_data['min_hip_rotation']
    head_drift = analysis_data['max_head_drift_x']
    wrist_cock = analysis_data['max_wrist_cock']

    system_prompt = (
        "あなたは世界トップクラスのゴルフコーチです。提供されたMediaPipeの計測結果に基づき、以下の10項目（02から10まで）の構成を網羅した、プロフェッショナルな診断レポートを生成してください。"
        "出力は必ずMarkdown形式で行い、各セクションの日本語タイトルは以下の指示に従ってください。\n"
        "【重要】項目09のフィッティング提案では、具体的な商品名やブランド名を**絶対に出さないで**ください。代わりに、シャフトの特性（調子、トルク、重量）といった専門的なフィッティング要素を提案してください。"
    )

    user_prompt = (
        f"ゴルフスイングの解析結果です。対象は初心者〜中級者です。全ての診断は以下の数値データに基づいて行ってください。\n"
        f"・最大肩回転 (Top of Backswing): {shoulder_rot:.1f}度\n"
        f"・最小腰回転 (Impact/Follow): {hip_rot:.1f}度\n"
        f"・頭の最大水平ブレ (Max Head Drift X, 0.001が最小ブレ): {head_drift:.4f}\n"
        f"・最大コック角 (Max Wrist Cock Angle, 180度が伸びた状態): {wrist_cock:.1f}度\n\n"
        f"レポート構成の指示:\n"
        f"02. 頭の安定性 (Head Stability)\n"
        f"03. 肩の回旋 (Shoulder Rotation)\n"
        f"04. 腰の回旋 (Hip Rotation)\n"
        f"05. 手首のメカニクス (Wrist Mechanics) - コック角に基づき、アーリーリリースなどを評価してください。\n"
        f"06. 手の軌道 (Hand Path) - データが限られているため、回転とコック角の傾向からアウトサイドイン/インサイドアウトを推測してください。\n"
        f"07. 総合診断 (Key Diagnosis)\n"
        f"08. 改善戦略とドリル (Improvement Strategy)\n"
        f"09. フィッティング提案 (Fitting Recommendation) - **商品名なし**で、シャフト特性を提案してください。\n"
        f"10. エグゼクティブサマリー (Executive Summary)\n"
        f"この構成で、各項目を詳細に分析してください。"
    )

    # Gemini API呼び出し
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )
        return response.text
        
    except Exception as e:
        return f"Gemini API呼び出し中にエラーが発生しました: {e}"

# ------------------------------------------------
# LINE Webhookのメイン処理 (省略)
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
            TextSendMessage(text="動画を送信してください。有料会員向けの**プロレベル詳細レポート**を生成します。")
        )
        
@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event):
    user_id = event.source.user_id
    message_id = event.message.id

    # 1. ユーザーへの即時応答（LINEの応答タイムアウト回避）
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="動画を受け付けました。AIによるプロレベル詳細解析を開始します。しばらくお待ちください...")
    )
    
    # 2. 動画コンテンツの取得 (中略)
    try:
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

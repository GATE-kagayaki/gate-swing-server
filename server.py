import os
import tempfile 
import ffmpeg 
import requests
import numpy as np 
import json
import datetime
# Cloud Tasksに必要なインポート
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
from google.cloud import firestore
# Firebase/Firestoreのインポート (Webレポート保存に必須)
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from google import genai
from google.genai import types

from flask import Flask, request, abort, jsonify, json, send_file 
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, VideoMessage

# ------------------------------------------------
# 環境変数の設定と定数定義
# ------------------------------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 
# GCP_PROJECT_ID, TASK_SA_EMAIL, SERVICE_HOST_URL は必須のため、厳しくチェック
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID') 
TASK_SA_EMAIL = os.environ.get('TASK_SA_EMAIL') 
SERVICE_HOST_URL = os.environ.get('SERVICE_HOST_URL')

TASK_QUEUE_LOCATION = os.environ.get('TASK_QUEUE_LOCATION', 'asia-northeast2') 
TASK_QUEUE_NAME = 'video-analysis-queue'
TASK_HANDLER_PATH = '/worker/process_video'

# 環境変数の必須チェックを強化 (起動時チェック)
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set")
if not SERVICE_HOST_URL:
    raise ValueError("SERVICE_HOST_URL must be set (e.g., https://<service-name>-<hash>.<region>.run.app)")
if not GCP_PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID must be set.")
# TASK_SA_EMAILは認証エラーの原因となるため、タスク投入関数で厳しくチェックする

# FlaskアプリとLINE Bot APIの設定
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app.config['JSON_AS_ASCII'] = False 

# Firestoreクライアントの初期化
db = None
try:
    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        initialize_app(cred, {'projectId': GCP_PROJECT_ID})
    db = firestore.client()
except Exception as e:
    app.logger.error(f"Error initializing Firestore: {e}")
    # dbがNoneのままになるため、Firestore関連関数内でdbのNoneチェックが必要

# Cloud Tasks クライアントの初期化
task_client = None
try:
    if GCP_PROJECT_ID: # GCP_PROJECT_IDがNoneでない場合のみ初期化を試行
        task_client = tasks_v2.CloudTasksClient()
        task_queue_path = task_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)
except Exception as e:
    app.logger.error(f"Cloud Tasks Client initialization failed: {e}")
    task_client = None

# ------------------------------------------------
# ★★★ Firestore連携関数 ★★★
# ------------------------------------------------

def save_report_to_firestore(user_id, report_id, report_data):
    """診断レポートをFirestoreに保存する"""
    if db is None:
        app.logger.error("Firestore client is not initialized. Cannot save report.")
        return False
    try:
        doc_ref = db.collection('reports').document(report_id)
        report_data['user_id'] = user_id
        report_data['timestamp'] = firestore.SERVER_TIMESTAMP
        doc_ref.set(report_data)
        return True
    except Exception as e:
        app.logger.error(f"Error saving report to Firestore: {e}")
        return False

def get_report_from_firestore(report_id):
    """Firestoreからレポートを取得する"""
    if db is None:
        app.logger.error("Firestore client is not initialized. Cannot fetch report.")
        return None
    try:
        doc_ref = db.collection('reports').document(report_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return None
    except Exception as e:
        app.logger.error(f"Error getting report from Firestore: {e}")
        return None

# ------------------------------------------------
# 解析ロジック (analyze_swing) - 必須計測項目を全て実装
# ------------------------------------------------
def analyze_swing(video_path):
    # 動画を解析し、スイングの評価レポート（テキスト）を返す。
    # この関数は、process_task内から呼び出されます。
    import cv2
    import mediapipe as mp
    import numpy as np

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
    
    mp_pose = mp.solutions.pose
    
    # 計測変数初期化
    max_shoulder_rotation = -180
    min_hip_rotation = 180
    head_start_x = None 
    max_head_drift_x = 0 
    max_wrist_cock = 0  
    knee_start_x = None
    max_knee_sway_x = 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "動画ファイルを開けませんでした。"}

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
                LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
                RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value

                # 座標抽出
                r_shoulder = [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y]
                r_ear = [landmarks[RIGHT_EAR].x, landmarks[RIGHT_EAR].y]
                l_hip = [landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y]
                r_hip = [landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y]
                nose = [landmarks[NOSE].x, landmarks[NOSE].y]
                r_wrist = [landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y]
                r_elbow = [landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y]
                r_index = [landmarks[RIGHT_INDEX].x, landmarks[RIGHT_INDEX].y]
                r_knee = [landmarks[RIGHT_KNEE].x, landmarks[RIGHT_KNEE].y]
                l_knee = [landmarks[LEFT_KNEE].x, landmarks[LEFT_KNEE].y]


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

                # 計測：最大膝ブレ（スウェイ）
                mid_knee_x = (r_knee[0] + l_knee[0]) / 2
                if knee_start_x is None:
                    knee_start_x = mid_knee_x
                current_knee_sway = abs(mid_knee_x - knee_start_x)
                if current_knee_sway > max_knee_sway_x:
                    max_knee_sway_x = current_knee_sway
                
    cap.release()
    
    # 全ての計測結果を辞書で返す
    return {
        "frame_count": frame_count,
        "max_shoulder_rotation": max_shoulder_rotation,
        "min_hip_rotation": min_hip_rotation,
        "max_head_drift_x": max_head_drift_x,
        "max_wrist_cock": max_wrist_cock,
        "max_knee_sway_x": max_knee_sway_x 
    }

# ------------------------------------------------
# Gemini API 呼び出し関数 (有料会員向け詳細レポート)
# ------------------------------------------------
def run_ai_analysis(raw_data): 
    """MediaPipeの数値結果をGemini APIに渡し、理想の10項目を網羅した詳細レポートを生成させる"""
    
    if not GEMINI_API_KEY:
        app.logger.error("GEMINI_API_KEY is not set.")
        return "## 03. AI総合評価\nAI診断レポートの生成に必要なAPIキーが設定されていません。", "AI診断が実行できませんでした。"
        
    try:
        # Geminiクライアントの初期化
        client = genai.Client(api_key=GEMINI_API_KEY)

        # プロンプトの構築
        prompt = (
            "あなたは世界トップクラスのゴルフスイングコーチであり、AIドクターです。\n"
            "提供されたスイングの骨格データ（MediaPipeによる数値）に基づき、以下の構造で詳細な日本語の診断レポートを作成してください。\n"
            "数値データは、プロの基準値（例: 最大肩回転90°〜110°、最小腰回転30°〜45°など）と対比させて論じてください。\n\n"
            "**レポートの構造:**\n"
            "1. 総合評価の要約（簡潔に、褒める言葉から始めること）\n"
            "2. **## 03. AI総合評価**\n"
            "3. **## 04. バックスイングの課題と改善点**\n"
            "4. **## 05. トップオブスイングの課題と改善点**\n"
            "5. **## 06. ダウンスイングの課題と改善点**\n"
            "6. **## 07. インパクトとフォロースルーの課題と改善点**\n"
            "7. **## 08. 練習ドリルとアドバイス**\n\n"
            "各セクションの内容は、Markdownの箇条書き（* を使用）を豊富に使い、具体的な数値を引用して説明してください。\n\n"
            "**骨格計測データ:**\n"
            f"{json.dumps(raw_data, indent=2, ensure_ascii=False)}\n"
        )

        # Gemini APIの呼び出し
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        full_report = response.text
        
        # 総合評価の要約を抽出（最初の数行）
        summary_match = full_report.split('## 03.')[0].strip()
        summary = summary_match if summary_match else "AIによる総合評価の抽出に失敗しましたが、詳細はレポート本文をご確認ください。"

        return full_report, summary

    except Exception as e:
        app.logger.error(f"Gemini API call failed: {e}")
        return "## 03. AI総合評価\nAI診断レポートの生成中にエラーが発生しました。", "AI診断が実行できませんでした。"

# ------------------------------------------------
# Cloud Tasksへジョブを投入する関数
# ------------------------------------------------

def create_cloud_task(report_id, video_url, user_id):
    """
    Cloud Tasksに動画解析タスクを作成し、Cloud Run Workerをトリガーする
    """
    # 必須認証情報が設定されているかチェック
    if not task_client:
        app.logger.error("Cloud Tasks client is not initialized.")
        return None
    if not TASK_SA_EMAIL:
        app.logger.error("TASK_SA_EMAIL is missing. Cannot authenticate Cloud Task.")
        return None
    if not SERVICE_HOST_URL:
        app.logger.error("SERVICE_HOST_URL is missing. Cannot create Cloud Task.")
        return None
        
    full_url = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

    # タスクに含めるペイロード (JSON形式)
    payload_dict = {
        'report_id': report_id,
        'video_url': video_url,
        'user_id': user_id,
    }
    # Cloud Tasksのペイロードはバイト文字列でなければならない
    task_payload = json.dumps(payload_dict).encode()

    task = {
        'http_request': {  # Cloud Run Workerを呼び出す設定
            'http_method': tasks_v2.HttpMethod.POST,
            'url': full_url,
            'body': task_payload,
            'headers': {'Content-Type': 'application/json'},
            # OIDC認証トークンを使用して認証を行う
            'oidc_token': {
                'service_account_email': TASK_SA_EMAIL, 
            },
        }
    }

    try:
        # タスクをキューに送信
        response = task_client.create_task(parent=task_queue_path, task=task)
        app.logger.info(f"Task created: {response.name}")
        return response.name
    except Exception as e:
        app.logger.error(f"Error creating Cloud Task: {e}")
        return None

# ------------------------------------------------
# LINE Bot Webhookハンドラー
# ------------------------------------------------

@app.route("/webhook", methods=['POST'])
def webhook():
    """LINEプラットフォームからのWebhookリクエストを受け付ける"""
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("Invalid signature. Check your channel access token/secret.")
        abort(400)
    except LineBotApiError as e:
        app.logger.error(f"LINE Bot API error: {e.status_code}, {e.error.message}")
        abort(500)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """テキストメッセージを受信したときの処理"""
    if event.message.text in ["レポート確認", "report"]:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="お送りいただいた動画の直近のレポートURLを後ほどお送りします。\n(実装簡略化のため、現在は動画を送るとすぐURLを返します)")
        )
    else:
        # ★★★ 修正: 新しい料金プランを反映 ★★★
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="動画をアップロードしてください。AIスイングドクターが解析を開始します。\n\n【料金プラン】\n・都度契約: 500円/1回\n・回数券: 1,980円/5回券 (実質1回あたり396円)\n・月額契約: 4,980円/無制限")
        )

@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    """動画メッセージを受信したときの処理"""
    user_id = event.source.user_id
    message_id = event.message.id
    report_id = f"{user_id}_{message_id}"
    
    app.logger.info(f"Received video message. User ID: {user_id}, Message ID: {message_id}")

    # 必須環境変数の再々々チェック
    if not SERVICE_HOST_URL or not TASK_SA_EMAIL or not task_client:
        error_msg = ("システムエラー: 環境設定が不完全です。"
                     "管理者にお問い合わせください。 (原因: SERVICE_HOST_URL, TASK_SA_EMAIL, または Cloud Tasks Client の未設定)")
        app.logger.error(error_msg)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
        return 'OK'

    try:
        # 1. FirestoreにPROCESSINGステータスで初期エントリを保存
        initial_data = {
            'status': 'PROCESSING',
            'user_id': user_id,
            'message_id': message_id,
            'video_url': f"line_message_id://{message_id}",
            'summary': '動画解析を開始しました。',
            'ai_report': '',
            'raw_data': {},
        }
        if not save_report_to_firestore(user_id, report_id, initial_data):
            raise Exception("Failed to save initial report to Firestore.")

        # 2. Cloud Tasksにジョブを登録
        task_name = create_cloud_task(report_id, initial_data['video_url'], user_id)
        
        if not task_name:
            # タスク登録に失敗した場合、ユーザーに失敗を通知し、Firestoreのステータスを更新
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="システムエラー: 動画解析ジョブの登録に失敗しました。時間をおいて再度お試しください。")
            )
            db.collection('reports').document(report_id).update({'status': 'TASK_FAILED', 'summary': 'タスク登録失敗'})
            return

        # 3. ユーザーに即時応答
        report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
        
        reply_message = (
            "✅ 動画を受信しました。解析を開始します！\n"
            "AIによるスイング診断には数分かかります。\n"
            "結果は準備でき次第、改めてメッセージでお知らせします。\n\n"
            f"**[処理状況確認URL]**\n{report_url}\n"
            "（LINEのタイムアウトを防ぐため、このURLで進捗を確認できます）\n\n"
            "【料金プラン】\n・都度契約: 500円/1回\n・回数券: 1,980円/5回券 (実質1回あたり396円)\n・月額契約: 4,980円/無制限"
        )
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_message)
        )

    except Exception as e:
        app.logger.error(f"Error in video message handler: {e}")
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"動画処理中に予期せぬエラーが発生しました: {e}. 管理者にお問い合わせください。")
            )
        except:
            pass 
            
    return 'OK' # Webhookは常にOKを返して終了する

# ------------------------------------------------
# Cloud Run Worker (タスク実行ハンドラー)
# ------------------------------------------------

@app.route("/worker/process_video", methods=['POST'])
def process_video_worker():
    """
    Cloud Tasksから呼び出される動画解析のWorkerエンドポイント
    """
    try:
        # Cloud Tasksから送られてくるJSONペイロードを解析
        task_data = request.get_json(silent=True)
        if not task_data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing task payload'}), 400

        report_id = task_data.get('report_id')
        # video_url = task_data.get('video_url') # Cloud Tasksでは動画URLではなくMessage IDを渡す
        user_id = task_data.get('user_id')
        message_id = report_id.split('_')[-1] # Report IDからMessage IDを抽出

        if not report_id or not user_id or not message_id:
            return jsonify({'status': 'error', 'message': 'Missing required parameters in payload'}), 400

        app.logger.info(f"Worker received job. Report ID: {report_id}")
        
        # 0. Firestoreのステータスを「IN_PROGRESS」に更新
        if db:
            db.collection('reports').document(report_id).update({'status': 'IN_PROGRESS', 'summary': '動画解析を実行中です...'})

        # 1. LINEから動画コンテンツを再取得 (Workerの処理本体)
        video_content = None
        try:
            # LINEからコンテンツを直接取得
            message_content = line_bot_api.get_message_content(message_id)
            video_content = message_content.content
        except Exception as e:
            app.logger.error(f"LINE Content API error for message ID {message_id}: {e}", exc_info=True)
            db.collection('reports').document(report_id).update({'status': 'LINE_FETCH_FAILED', 'summary': 'LINEからの動画取得に失敗しました。時間をおいて再実行されます。'})
            # Cloud Tasksにリトライを依頼するため、HTTP 500を返す
            return jsonify({'status': 'error', 'message': 'Failed to fetch video content from LINE'}), 500

        # 2. 動画の解析とAI診断の実行
        original_video_path = None
        compressed_video_path = None
        analysis_data = {}
        
        try:
            # 2.1 オリジナル動画を一時ファイルに保存
            with tempfile.NamedTemporaryFile(suffix="_original.mp4", delete=False) as tmp_file:
                original_video_path = tmp_file.name
                tmp_file.write(video_content)

            # 2.2 動画の自動圧縮とリサイズ処理
            compressed_video_path = tempfile.NamedTemporaryFile(suffix="_compressed.mp4", delete=False).name
            FFMPEG_PATH = '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else 'ffmpeg'
            
            ffmpeg.input(original_video_path).output(
                compressed_video_path, vf='scale=640:-1', crf=28, vcodec='libx264'
            ).overwrite_output().run(cmd=FFMPEG_PATH, capture_stdout=True, capture_stderr=True) 

            # 2.3 MediaPipe解析を実行
            analysis_data = analyze_swing(compressed_video_path)
            
            # 2.4 AIによる診断レポートの生成
            ai_report_markdown, summary_text = run_ai_analysis(analysis_data)
            
        except Exception as e:
            app.logger.error(f"MediaPipe/FFmpeg/AI processing failed: {e}", exc_info=True)
            # 解析失敗時も、タスクがリトライしないように200を返し、Firestoreでエラーを通知
            if db:
                 db.collection('reports').document(report_id).update({'status': 'ANALYSIS_FAILED', 'summary': f'動画解析処理中に予期せぬエラーが発生しました: {str(e)[:100]}...'})
            line_bot_api.push_message(user_id, TextSendMessage(text=f"【解析エラー】動画解析が失敗しました。全身が写っているかご確認ください。"))
            return jsonify({'status': 'error', 'message': 'Analysis failed'}), 200 # 200を返すことでタスクのリトライを停止
        
        finally:
            # 一時ファイルのクリーンアップ
            if original_video_path and os.path.exists(original_video_path): os.remove(original_video_path)
            if compressed_video_path and os.path.exists(compressed_video_path): os.remove(compressed_video_path)

        
        # 3. 結果をFirestoreに保存（ステータス: COMPLETED）
        final_data = {
            'status': 'COMPLETED',
            'summary': summary_text,
            'ai_report': ai_report_markdown,
            'raw_data': analysis_data,
        }
        if save_report_to_firestore(user_id, report_id, final_data):
            app.logger.info(f"Report {report_id} saved as COMPLETED.")

            # 4. ユーザーに最終通知をLINEで送信
            report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
            final_line_message = (
                "🎉 AIスイング診断が完了しました！\n\n"
                f"**[診断レポートURL]**\n{report_url}\n\n"
                f"**[総合評価]**\n{summary_text}\n"
                "詳細なレポートはURLからご確認ください。次の練習にお役立てください！"
            )
            line_bot_api.push_message(
                to=user_id,
                messages=TextSendMessage(text=final_line_message)
            )

            return jsonify({'status': 'success', 'report_id': report_id}), 200
        else:
            # Firestore保存失敗時
            return jsonify({'status': 'error', 'message': 'Failed to save final report to Firestore'}), 500

    except Exception as e:
        app.logger.error(f"Worker processing failed for task: {report_id}. Error: {e}")
        # Firestoreのステータスを更新 (処理失敗)
        if db:
             db.collection('reports').document(report_id).update({'status': 'FATAL_ERROR', 'summary': f'致命的なエラーが発生しました: {str(e)[:100]}...'})
        # Cloud Tasksにリトライを依頼するため、HTTP 500を返す (LINE通知は既に処理済みのため、ここでは不要)
        return jsonify({'status': 'error', 'message': f'Internal Server Error: {e}'}), 500

# ------------------------------------------------
# Webレポート表示エンドポイント
# ------------------------------------------------

@app.route("/report/<report_id>", methods=['GET'])
def get_report_web(report_id):
    """
    レポートIDに対応するWebレポートを表示する
    """
    report_data = get_report_from_firestore(report_id)

    if not report_data:
        # レポートが存在しない場合
        error_html = HTML_REPORT_TEMPLATE.replace('<!-- REPORT_STATUS_SCRIPT -->', f"""
            <script>
                window.onload = function() {{
                    displayFatalError("レポートが見つかりませんでした。", "指定されたID ({report_id}) のレポートは存在しないか、削除されています。");
                }};
            </script>
        """)
        return error_html, 404

    status = report_data.get('status')
    
    if status in ['PROCESSING', 'IN_PROGRESS']:
        # 処理中の場合
        processing_html = HTML_REPORT_TEMPLATE.replace('<!-- REPORT_STATUS_SCRIPT -->', f"""
            <script>
                window.onload = function() {{
                    displayProcessingMessage();
                }};
            </script>
        """)
        return processing_html, 202

    if status == 'COMPLETED':
        # 完了している場合、データをHTMLに埋め込んで返す
        ai_report_markdown = report_data.get('ai_report', '## 03. AI総合評価\nレポート本文がありません。')
        raw_data = report_data.get('raw_data', {})
        
        # JavaScriptで利用できるようにデータをJSON文字列として埋め込む
        report_data_json = json.dumps({
            'ai_report': ai_report_markdown,
            'raw_data': raw_data,
            'summary': report_data.get('summary', ''),
            'timestamp': report_data.get('timestamp').isoformat() if report_data.get('timestamp') else datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        
        # HTMLテンプレートのscript部分にデータをロードする処理を追加
        final_html = HTML_REPORT_TEMPLATE.replace('<!-- REPORT_STATUS_SCRIPT -->', f"""
            <script id="report-data-script" type="application/json">
            {report_data_json}
            </script>
            <script>
            window.onload = function() {{
                const reportData = JSON.parse(document.getElementById('report-data-script').textContent);
                
                const timestamp = new Date(reportData.timestamp).toLocaleString('ja-JP', {{
                    year: 'numeric', month: '2-digit', day: '2-digit', 
                    hour: '2-digit', minute: '2-digit', second: '2-digit'
                }});

                document.getElementById('timestamp').textContent = timestamp;
                document.getElementById('summary-text').textContent = reportData.summary;
                document.getElementById('report-id').textContent = "{report_id}";
                
                renderPages(reportData.ai_report, reportData.raw_data);
            }};
            </script>
        """)
        
        return final_html

    # その他の不明なステータス
    error_html = HTML_REPORT_TEMPLATE.replace('<!-- REPORT_STATUS_SCRIPT -->', f"""
        <script>
            window.onload = function() {{
                displayFatalError("レポート処理中にエラーが発生しています。", `ステータス: {status} / 詳細: {report_data.get('summary', '不明')}`);
            }};
        </script>
    """)
    return error_html, 500

# ------------------------------------------------
# Flask実行
# ------------------------------------------------
if __name__ == "__main__":
    # ローカル実行時には、環境変数でポートを指定する
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

import os
import tempfile 
import shutil
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
        report_data['status'] = report_data.get('status', 'COMPLETED') # デフォルトステータス
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
    
    # ファイルの存在確認を強化
    if not os.path.exists(video_path):
        app.logger.error(f"動画ファイルパスエラー: {video_path} が見つかりません。")
        return {"error": f"動画ファイルが見つかりません: {os.path.basename(video_path)}"}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        app.logger.error(f"cv2.VideoCaptureが開けませんでした。ファイル破損または不正な形式: {video_path}")
        return {"error": "動画ファイルを開けませんでした。不正な形式の可能性があります。"}

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
                         max_wrist_cock = cock_angle # cock_angle を代入

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
        summary_match = full_report.split('## 03.')[0].trim() # ★修正: trim()を削除して改行で分割
        
        summary_match = full_report.split('\n')[0].strip() # 最初の行をサマリーとして抽出
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
    global task_client, task_queue_path
    
    if task_client is None:
        # 初期化が失敗しているため、ここで再試行する
        try:
            task_client = tasks_v2.CloudTasksClient()
            task_queue_path = task_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)
        except Exception as e:
            app.logger.error(f"Cloud Tasks client initialization failed in runtime: {e}")
            return None # クライアント初期化失敗

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
                'service_account_email': TASK_SA_ACCOUNT, 
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
        app.logger.error("Invalid signature. Check your channel secret.")
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
    if not SERVICE_HOST_URL or not TASK_SA_EMAIL:
        error_msg = ("システムエラー: 環境設定が不完全です。"
                     "管理者にお問い合わせください。 (原因: SERVICE_HOST_URL, TASK_SA_EMAILが未設定)")
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
            # Firestoreの初期化に失敗している可能性
            error_msg = ("システムエラー: データベース接続に失敗しました。管理者にお問い合わせください。")
            app.logger.error(error_msg)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
            return 'OK'

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
    report_id = None
    original_video_path = None
    compressed_video_path = None
    user_id = None
    temp_dir = None

    try:
        # Cloud Tasksから送られてくるJSONペイロードを解析
        task_data = request.get_json(silent=True)
        if not task_data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing task payload'}), 400

        report_id = task_data.get('report_id')
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
        analysis_data = {}
        
        # ★修正: 一時ディレクトリを作成し、その中にファイルを配置することでパーミッション問題を回避
        temp_dir = tempfile.mkdtemp()
        original_video_path = os.path.join(temp_dir, "original.mp4")
        compressed_video_path = os.path.join(temp_dir, "compressed.mp4")

        try:
            # 2.1 オリジナル動画を一時ファイルに保存
            with open(original_video_path, 'wb') as f:
                f.write(video_content)

            # 2.2 動画の自動圧縮とリサイズ処理
            FFMPEG_PATH = '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else 'ffmpeg'
            
            # ffmpegコマンドをより安定化 (preset='veryfast'を維持)
            ffmpeg.input(original_video_path).output(
                compressed_video_path, vf='scale=640:-1', crf=28, vcodec='libx264', preset='veryfast',
            ).overwrite_output().run(cmd=FFMPEG_PATH, capture_stdout=True, capture_stderr=True) 

            # 2.3 MediaPipe解析を実行
            analysis_data = analyze_swing(compressed_video_path)
            
            if analysis_data.get("error"):
                # analyze_swing内部で動画ファイルが開けない等のエラーが検出された場合
                raise Exception(f"MediaPipe解析失敗: {analysis_data['error']}")
                
            # 2.4 AIによる診断レポートの生成
            ai_report_markdown, summary_text = run_ai_analysis(analysis_data)
            
        except Exception as e:
            # Worker内部での解析エラー
            error_details = str(e)
            app.logger.error(f"MediaPipe/FFmpeg/AI processing failed: {error_details}", exc_info=True)
            
            # Firestoreのステータスを更新 (ANALYSIS_FAILED)
            if db:
                 db.collection('reports').document(report_id).update({'status': 'ANALYSIS_FAILED', 'summary': f'動画解析処理中にエラーが発生しました。詳細: {error_details[:100]}...'})
            
            # ★修正: LINEへの通知はエラー詳細を削除し、シンプルに
            line_bot_api.push_message(user_id, TextSendMessage(text=f"【解析エラー】動画解析が失敗しました。全身が写っているかご確認ください。"))
            return jsonify({'status': 'error', 'message': 'Analysis failed'}), 200 
        
        finally:
            # ★修正: 一時ディレクトリ全体を確実にクリーンアップ
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        
        # 3. 結果をFirestoreに保存（ステータsス: COMPLETED）
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
        # Worker全体で予期せぬ致命的なエラーが発生した場合
        app.logger.error(f"Worker processing failed for task: {report_id}. Error: {e}")
        # Firestoreのステータスを更新 (処理失敗)
        if db:
             db.collection('reports').document(report_id).update({'status': 'FATAL_ERROR', 'summary': f'致命的なエラーが発生しました: {str(e)[:100]}...'})
        # Cloud Tasksにリトライを依頼するため、HTTP 500を返す
        return jsonify({'status': 'error', 'message': f'Internal Server Error: {e}'}), 500

# ------------------------------------------------
# Webレポート表示エンドポイント
# ------------------------------------------------

# レポート表示用のAPIエンドポイント
@app.route("/api/report_data/<report_id>", methods=['GET'])
def get_report_data(report_id):
    """WebレポートのフロントエンドにJSONデータを返すAPIエンドポイント"""
    app.logger.info(f"Report API accessed for ID: {report_id}")
    
    if not db:
        app.logger.error("Firestore DB connection is not initialized.")
        return jsonify({"error": "データベースが初期化されていません。サーバーログを確認してください。"}), 500

    try:
        doc = db.collection('reports').document(report_id).get()
        if not doc.exists:
            app.logger.warning(f"Report document not found: {report_id}")
            return jsonify({"error": "指定されたレポートは見つかりませんでした。"}), 404
        
        data = doc.to_dict()
        app.logger.info(f"Successfully retrieved data for report: {report_id}")
        
        # Webレポートのフロントエンドが必要なデータを構造化して返す
        # Timestampオブジェクトをブラウザで扱える形式に変換する
        timestamp_data = data.get('timestamp')
        if hasattr(timestamp_data, 'isoformat'):
             timestamp_str = timestamp_data.isoformat()
        elif isinstance(timestamp_data, dict) and '_seconds' in timestamp_data:
             timestamp_str = datetime.datetime.fromtimestamp(timestamp_data['_seconds']).isoformat()
        else:
             timestamp_str = str(timestamp_data)

        response_data = {
            "timestamp": timestamp_str,
            "mediapipe_data": data.get('raw_data', {}),
            "ai_report_text": data.get('ai_report', 'AIレポートがありません。'),
            "summary": data.get('summary', '総合評価データなし。'),
            "status": data.get('status', 'UNKNOWN')
        }
        
        # JSONレスポンスを返す際は、Flaskのjsonifyを使用するか、json.dumpsのdefault=strでdatetimeオブジェクトを処理する
        json_output = json.dumps(response_data, ensure_ascii=False)
        response = app.response_class(
            response=json_output,
            status=200,
            mimetype='application/json'
        )
        return response

    except Exception as e:
        app.logger.error(f"レポート表示APIエラー: {e}", exc_info=True)
        return jsonify({"error": f"レポートデータの取得中に予期せぬエラーが発生しました: {e}"}), 500

# WebレポートのHTMLを返すエンドポイント
@app.route("/report/<report_id>", methods=['GET'])
def get_report_web(report_id):
    """
    レポートIDに対応するWebレポートのHTMLテンプレートを返す
    """
    # HTMLの動的テンプレートを返す
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GATE AIスイングドクター診断レポート</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            /* 印刷時のCSS設定 */
            @media print {{
                body {{ padding: 0 !important; margin: 0 !important; font-size: 10pt; }}
                .no-print {{ display: none !important; }}
                #sidebar, #header-container {{ display: none !important; }}
                #main-content {{ margin-left: 0 !important; width: 100% !important; padding: 0 !important; }}
                .content-page {{ display: block !important; margin-bottom: 20px; page-break-after: always; }}
            }}
            
            /* カスタムCSS */
            .content-page {{
                /* ページングをシミュレートするため、非表示がデフォルト */
                display: none;
                min-height: calc(100vh - 80px);
            }}
            .content-page.active {{
                display: block;
            }}
            .report-content h2 {{
                font-size: 1.5em; 
                font-weight: bold;
                color: #059669; /* Emerald Green */
                border-bottom: 2px solid #34d399;
                padding-bottom: 0.25em;
                margin-top: 1.5em;
            }}
            .report-content strong {{
                color: #10b981;
            }}
            .report-content ul {{
                list-style-type: disc;
                margin-left: 1.5rem;
                padding-left: 0.5rem;
            }}
            .nav-item {{
                cursor: pointer;
                transition: background-color 0.2s;
                border-left: 4px solid transparent; 
            }}
            .nav-item:hover {{
                background-color: #f0fdf4;
            }}
            .nav-item.active {{
                background-color: #d1fae5;
                color: #059669;
                font-weight: bold;
                border-left: 4px solid #10b981;
            }}
        </style>
    </head>
    <body class="bg-gray-100 font-sans">
        
        <!-- Loading Spinner -->
        <div id="loading" class="fixed inset-0 bg-white bg-opacity-75 flex flex-col justify-center items-center z-50">
            <div class="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-green-500"></div>
            <p class="mt-4 text-xl text-gray-700 font-semibold">AIレポートを読み込み中...</p>
        </div>

        <!-- メインレイアウト -->
        <div id="report-container" class="flex min-h-screen max-w-full mx-auto" style="display: none;">

            <!-- サイドバー (ナビゲーション) -->
            <aside id="sidebar" class="w-64 fixed left-0 top-0 h-full bg-white shadow-xl p-4 overflow-y-auto no-print">
                <h1 class="text-2xl font-bold text-gray-800 border-b pb-2 mb-4">
                    ⛳ AI診断メニュー
                </h1>
                <nav id="nav-menu" class="space-y-1 text-gray-600">
                    <!-- ナビゲーション項目はJSで動的に挿入されます -->
                </nav>
            </aside>

            <!-- メインコンテンツエリア -->
            <main id="main-content" class="flex-1 transition-all duration-300 ml-64 p-4 md:p-8">
                
                <!-- レポートヘッダー -->
                <div id="header-container" class="bg-white p-4 rounded-lg shadow-md mb-6">
                    <header class="pb-2 border-b border-green-200">
                        <h1 class="text-3xl font-bold text-gray-800">
                            GATE AIスイングドクター診断レポート
                        </h1>
                        <p class="text-gray-500 mt-1 text-sm">
                            最終診断日: <span id="timestamp"></span> | レポートID: <span id="report-id"></span>
                        </p>
                    </header>
                </div>
                
                <!-- ページングされたコンテンツ -->
                <div id="report-pages" class="bg-white p-6 rounded-lg shadow-md min-h-[70vh] report-content">
                    <!-- 各診断項目（ページ）がここに動的に挿入されます -->
                </div>

                <footer class="mt-8 pt-4 border-t border-gray-300 text-center text-sm text-gray-500 no-print">
                    <p>このレポートはAIによる骨格分析に基づき診断されています。</p>
                    <button onclick="window.print()" class="mt-4 px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition duration-150 shadow-lg">
                        📄 PDFとして保存 / 印刷
                    </button>
                </footer>

            </main>
        </div>

        <script>
            // レポートIDをJSに渡す
            const REPORT_ID = "{report_id}";

            // ナビゲーションメニューの定義 (固定項目)
            const NAV_ITEMS = [
                {{ id: 'summary', title: '00. レポート概要' }},
                {{ id: 'mediapipe', title: '01. 骨格計測データ' }},
                {{ id: 'criteria', title: '02. データ評価基準' }},
            ];

            let aiReportContent = {{}};
            let currentPageId = 'summary';

            function displayFatalError(message, details = null) {{
                const loadingElement = document.getElementById('loading');
                loadingElement.classList.remove('hidden');
                loadingElement.innerHTML = `<div class="p-6 bg-red-100 border-l-4 border-red-500 text-red-700 m-8">
                    <p class="font-bold">🚨 致命的なエラーが発生しました</p>
                    <p class="mt-2">${{message}}</p>`;
                if (details) {{
                    loadingElement.innerHTML += `<p class="mt-2 text-sm">詳細: ${{details}}</p>`;
                }}
                loadingElement.innerHTML += `</div>`;
                document.getElementById('report-container').style.display = 'none';
            }}
            
            function displayProcessingMessage(reportId) {{
                const loadingElement = document.getElementById('loading');
                loadingElement.classList.remove('hidden');
                loadingElement.innerHTML = `<div class="p-8 bg-white rounded-xl shadow-2xl text-center">
                    <p class="text-3xl font-bold text-yellow-600 mb-4">⏱️ 解析処理中です...</p>
                    <p class="text-gray-700">LINEにプッシュ通知が届くまで、しばらくお待ちください。</p>
                    <p class="text-sm text-gray-500 mt-4">レポートID: ${{reportId}}</p>
                </div>`;
                document.getElementById('report-container').style.display = 'none';
            }}

            // Markdownコンテンツを解析し、ページを構築する関数
            function renderPages(markdownContent, rawData) {{
                const pagesContainer = document.getElementById('report-pages');
                const navMenu = document.getElementById('nav-menu');
                pagesContainer.innerHTML = '';
                navMenu.innerHTML = '';

                // 1. Markdownコンテンツを分割
                const sections = markdownContent.split('## ').filter(s => s.trim() !== '');
                const dynamicNavItems = [];
                
                sections.forEach((section, index) => {{
                    const titleMatch = section.match(/^([^\\n]+)/);
                    if (titleMatch) {{
                        const fullTitle = titleMatch[1].trim();
                        const id = 'ai-sec-' + index;
                        dynamicNavItems.push({{ id: id, title: fullTitle }});
                        
                        // Markdown本文を取得
                        const content = section.substring(titleMatch[0].length).trim();
                        aiReportContent[id] = content;
                    }}
                }});

                // 2. ナビゲーションメニューを構築
                const fullNavItems = [...NAV_ITEMS, ...dynamicNavItems];
                fullNavItems.forEach(item => {{
                    const navItem = document.createElement('div');
                    navItem.className = `nav-item p-2 rounded-lg text-sm transition-all duration-150 ${{item.id === currentPageId ? 'active' : ''}}`;
                    navItem.textContent = item.title;
                    navItem.dataset.pageId = item.id;
                    navItem.onclick = () => showPage(item.id);
                    navMenu.appendChild(navItem);
                }});

                // 3. 固定ページコンテンツの定義と挿入 (rawDataを使用)
                pagesContainer.appendChild(createSummaryPage());
                pagesContainer.appendChild(createRawDataPage(rawData));
                pagesContainer.appendChild(createCriteriaPage());

                // 4. AI動的ページコンテンツの定義と挿入
                dynamicNavItems.forEach(item => {{
                    const page = document.createElement('div');
                    page.id = item.id;
                    page.className = 'content-page p-4';
                    
                    page.innerHTML += `<h2 class="text-2xl font-bold text-green-700 mb-4">${{item.title}}</h2>`;
                    
                    // Markdownの改行とリストをHTMLに変換
                    let processedText = aiReportContent[item.id]
                        .split('\\n')
                        .map(line => {{
                            // Markdownのリスト項目を<li>に変換
                            if (line.trim().startsWith('* ')) {{
                                return `<li>${{line.trim().substring(2)}}</li>`;
                            }}
                            // その他の行は<br>で改行
                            return line + '<br>';
                        }})
                        .join('');

                    // 連続する<li>を<ul>で囲む (簡易Markdown処理)
                    processedText = processedText.replace(/(<br>)*(<li>.*?<\/li>)+/gs, (match) => {{
                        let listItems = match.replace(/<br>/g, '');
                        // ulタグが連続しないように、既に<ul>で囲まれていないか確認
                        if (!listItems.trim().startsWith('<ul>')) {{
                            listItems = `<ul>${{listItems.replace(/<\/li>/g, '</li>')}}</ul>`;
                        }}
                        return listItems;
                    }});
                    
                    // 最後に残った<br>を整理
                    processedText = processedText.replace(/(<br>){{2,}}/g, '<br><br>');
                    processedText = processedText.replace(/<br>$/, '');
                    
                    page.innerHTML += processedText; 
                    pagesContainer.appendChild(page);
                }});

                showPage(currentPageId);
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('report-container').style.display = 'flex';
            }}
            
            function createRawDataPage(raw) {{
                const page = document.createElement('div');
                page.id = 'mediapipe';
                page.className = 'content-page p-4';
                page.innerHTML = `
                    <h2 class="text-2xl font-bold text-green-700 mb-6">01. 骨格計測データ (MediaPipe)</h2>
                    <section class="mb-8">
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                            <div class="p-3 bg-gray-100 rounded-lg">
                                <p class="text-2xl font-bold text-gray-800">${{raw.frame_count || 'N/A'}}</p>
                                <p class="text-xs text-gray-500">解析フレーム数</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg">
                                <p class="text-2xl font-bold text-gray-800">${{raw.max_shoulder_rotation ? raw.max_shoulder_rotation.toFixed(1) + '°' : 'N/A'}}</p>
                                <p class="text-xs text-gray-500">最大肩回転</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg">
                                <p class="text-2xl font-bold text-gray-800">${{raw.min_hip_rotation ? raw.min_hip_rotation.toFixed(1) + '°' : 'N/A'}}</p>
                                <p class="text-xs text-gray-500">最小腰回転</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg">
                                <p class="text-2xl font-bold text-gray-800">${{raw.max_wrist_cock ? raw.max_wrist_cock.toFixed(1) + '°' : 'N/A'}}</p>
                                <p class="text-xs text-gray-500">最大コック角</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg col-span-2">
                                <p class="text-2xl font-bold text-gray-800">${{raw.max_head_drift_x ? raw.max_head_drift_x.toFixed(4) : 'N/A'}}</p>
                                <p class="text-xs text-gray-500">最大頭ブレ(Sway)</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg col-span-2">
                                <p class="text-2xl font-bold text-gray-800">${{raw.max_knee_sway_x ? raw.max_knee_sway_x.toFixed(4) : 'N/A'}}</p>
                                <p class="text-xs text-gray-500">最大膝ブレ(Sway)</p>
                            </div>
                        </div>
                    </section>
                `;
                return page;
            }}

            function createCriteriaPage() {{
                const page = document.createElement('div');
                page.id = 'criteria';
                page.className = 'content-page p-4';
                page.innerHTML = `
                    <h2 class="text-2xl font-bold text-green-700 mb-6">02. データ評価基準</h2>
                    <section class="mb-8">
                        <div class="space-y-4 text-sm text-gray-600">
                            <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                                <h3 class="font-bold text-gray-800">最大肩回転 (03. 肩の回旋の基礎)</h3>
                                <p class="mt-1">
                                    <span class="font-semibold text-green-700">適正範囲の目安:</span> 70°〜90°程度 (ドライバー)。<br>
                                    <span class="text-red-600">マイナス値:</span> 目標線に対して肩がオープンになっている（捻転不足）可能性を示します。
                                </p>
                            </div>
                            <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                                <h3 class="font-bold text-gray-800">最小腰回転 (04. 腰の回旋の基礎)</h3>
                                <p class="mt-1">
                                    <span class="font-semibold text-green-700">適正範囲の目安:</span> 30°〜50°程度 (インパクト時)。<br>
                                    <span class="text-red-600">マイナス値:</span> 腰の開きがほとんどないか、目標の逆を向いていることを示唆。回転不足やスウェイ（軸ブレ）の可能性。
                                </p>
                            </div>
                            <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                                <h3 class="font-bold text-gray-800">最大コック角 (05. 手首のメカニクスの基礎)</h3>
                                <p class="mt-1">
                                    <span class="font-semibold text-green-700">適正範囲の目安:</span> 90°〜110°程度 (トップスイング)。<br>
                                    <span class="text-red-600">数値が大きい (160°超) :</span> 手首のタメが不足し、「アーリーリリース」の可能性が高いです。
                                </p>
                            </div>
                            <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                                <h3 class="font-bold text-gray-800">最大膝ブレ(Sway) (06. 下半身安定の基礎)</h3>
                                <p class="mt-1">
                                    <span class="font-semibold text-green-700">適正範囲の目安:</span> 最小限 (セットアップ時からのブレが少ない)。<br>
                                    <span class="text-red-600">数値が大きい:</span> スイング中に下半身が水平方向に大きく移動している（スウェイ/スライド）ことを示します。軸が不安定になり、ミート率の低下やパワーロスにつながります。
                                </p>
                            </div>
                        </div>
                    </section>
                `;
                return page;
            }}
            
            function createSummaryPage() {{
                 const page = document.createElement('div');
                page.id = 'summary';
                page.className = 'content-page p-4';
                page.innerHTML = `
                    <h2 class="text-2xl font-bold text-green-700 mb-6">00. レポート概要</h2>
                    <div class="text-gray-700 space-y-4">
                        <p class="font-semibold">レポートの目的:</p>
                        <p>このレポートは、お客様のスイング動画をAIが骨格レベルで分析し、その計測データに基づいて詳細な診断と改善戦略を提供するものです。左側のメニューから各診断項目を選択して、詳細をご確認ください。</p>
                        <p class="font-semibold mt-4">診断項目一覧:</p>
                        <ul class="list-disc ml-6 text-sm text-gray-600">
                            <li>01. 骨格計測データ</li>
                            <li>02. データ評価基準</li>
                            <li>03. 肩の回旋</li>
                            <li>04. 腰の回旋</li>
                            <li>05. 手首のメカニクス</li>
                            <li>06. 下半身の安定性</li>
                            <li>07. 総合診断 (Key Diagnosis)</li>
                            <li>08. 改善戦略とドリル (Improvement Strategy)</li>
                            <li>09. フィッティング提案 (Fitting Recommendation)</li>
                            <li>10. エグゼクティブサマリー (Executive Summary)</li>
                        </ul>
                    </div>
                `;
                return page;
            }}

            function showPage(pageId) {{
                currentPageId = pageId;
                document.querySelectorAll('.content-page').forEach(page => {{
                    page.classList.remove('active');
                }});
                document.getElementById(pageId).classList.add('active');

                document.querySelectorAll('.nav-item').forEach(item => {{
                    item.classList.remove('active');
                    if (item.dataset.pageId === pageId) {{
                        item.classList.add('active');
                    }}
                }});
                window.scrollTo(0, 0); // ページ切り替え時に最上部へスクロール
            }}


            // メインのデータ取得とレンダリング
            document.addEventListener('DOMContentLoaded', async () => {{
                // レポートIDをURLから取得
                const reportId = window.location.pathname.split('/').pop();
                if (!reportId) {{
                    displayFatalError('レポートIDが指定されていません。');
                    return;
                }}
                
                // 処理中メッセージを表示
                displayProcessingMessage(reportId);

                try {{
                    // APIエンドポイントの修正: /api/report_data/<report_id>
                    const api_url = `${{window.location.origin}}/api/report_data/${{reportId}}`;
                    const response = await fetch(api_url);
                    
                    if (!response.ok) {{
                        const errorData = await response.json().catch(() => ({{error: `サーバーエラー ${{response.status}}`}}));
                        // レポートが見つからない場合は404エラーを表示
                        if(response.status === 404) {{
                            displayFatalError("レポートが見つかりませんでした。", `ID: ${{reportId}}`);
                        }} else {{
                            throw new Error(`API呼び出しエラー。HTTPステータス: ${{response.status}} (${{response.statusText}})`);
                        }}
                        return;
                    }}
                    
                    const data = await response.json();
                    
                    if (data.status === 'PROCESSING' || data.status === 'IN_PROGRESS') {{
                        // 処理中の場合はポーリングまたは手動リロードを推奨 (ここではメッセージを出すのみ)
                         displayProcessingMessage(reportId);
                         return;
                    }}
                    
                    if (data.error) {{
                         displayFatalError("APIがエラーを返しました。", data.error);
                         return;
                    }}
                    
                    // 1. 基本データの挿入
                    document.getElementById('report-id').textContent = reportId;
                    let timestamp = 'N/A';
                    try {{
                        // Firestoreから返された文字列またはオブジェクトをパース
                        const ts = data.timestamp;
                        if (ts && ts._seconds) {{
                            timestamp = new Date(ts._seconds * 1000).toLocaleString('ja-JP');
                        }} else if (ts) {{
                            timestamp = new Date(ts).toLocaleString('ja-JP');
                        }}
                    }} catch (e) {{
                        console.error("Timestamp parsing failed:", e);
                        timestamp = 'データ処理エラー';
                    }}
                    document.getElementById('timestamp').textContent = timestamp;
                    
                    // 2. Markdownコンテンツの取得
                    const markdownText = data.ai_report_text || "";
                    
                    // 3. ページングレンダリング開始
                    if (markdownText) {{
                        try {{
                            // MarkdownとRawDataをレンダリング
                            renderPages(markdownText, data.mediapipe_data || {{}});

                        }} catch (e) {{
                            console.error("Markdown structure parsing failed:", e);
                             displayFatalError("AIレポートの構造解析中にエラーが発生しました。", e.message);
                             return;
                        }}
                    }} else {{
                        // AIレポートがない場合も、固定ページは表示する
                        renderPages("", data.mediapipe_data || {{}});
                    }}

                }} catch (error) {{
                    displayFatalError("レポートの初期化中に致命的なエラーが発生しました。", error.message);
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content, 200

# ------------------------------------------------
# Flask実行
# ------------------------------------------------
if __name__ == "__main__":
    # ローカル実行時には、環境変数でポートを指定する
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

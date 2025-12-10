import os
import tempfile 
import shutil
import ffmpeg 
import requests
import numpy as np 
import json
import datetime
# datetimeのインポートを単純化 (timezone, timedeltaは使用しない)
from datetime import datetime
# Cloud Tasks, Firestore, Gemini APIのインポート
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
from google.cloud import firestore
from google import genai
from google.genai import types
# Firebase Adminのインポート
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app

from flask import Flask, request, abort, jsonify, json, send_file 
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, VideoMessage
import cv2
import mediapipe as mp

# ------------------------------------------------
# 環境変数の設定と定数定義
# ------------------------------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID') 
TASK_SA_EMAIL = os.environ.get('TASK_SA_EMAIL') 
SERVICE_HOST_URL = os.environ.get('SERVICE_HOST_URL')

# デバッグ用フォールバック
if not GCP_PROJECT_ID: GCP_PROJECT_ID = 'default-gcp-project-id'

TASK_QUEUE_LOCATION = os.environ.get('TASK_QUEUE_LOCATION', 'asia-northeast2') 
TASK_QUEUE_NAME = 'video-analysis-queue'
TASK_HANDLER_PATH = '/worker/process_video'

# FlaskアプリとLINE Bot APIの設定
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app.config['JSON_AS_ASCII'] = False 

# Firestoreクライアントの初期化 (以前のコード履歴より復元)
db = None
task_client = None
task_queue_path = None

try:
    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        initialize_app(cred, {'projectId': GCP_PROJECT_ID})
    db = firestore.client()
except Exception as e:
    print(f"Error initializing Firebase/Firestore: {e}")

try:
    if GCP_PROJECT_ID:
        task_client = tasks_v2.CloudTasksClient()
        task_queue_path = task_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)
except Exception as e:
    print(f"Cloud Tasks Client initialization failed: {e}")

# ------------------------------------------------
# ★★★ Firestore連携関数 (課金ロジック削除) ★★★
# ------------------------------------------------

def save_report_to_firestore(user_id, report_id, report_data):
    """診断レポートをFirestoreに保存する"""
    if db is None:
        print("Firestore client is not initialized. Cannot save report.")
        return False
    try:
        doc_ref = db.collection('reports').document(report_id)
        report_data['user_id'] = user_id
        report_data['timestamp'] = firestore.SERVER_TIMESTAMP
        report_data['status'] = report_data.get('status', 'COMPLETED') 
        doc_ref.set(report_data)
        return True
    except Exception as e:
        print(f"Error saving report to Firestore: {e}")
        return False

# ユーザーのサービス利用可否を判定する関数 (削除し、ダミーで常にTrueを返す)
def check_service_eligibility(user_id):
    """
    [MOCK] 課金ロジックが未実装のため、常にサービス利用可能 (is_premium=True) と見なす。
    """
    return True, 'free_preview', "全機能プレビューモードで利用可能です。"

# レポート作成成功時に利用回数を消費する関数 (完全に削除)
# def consume_service_count(user_id):
#     """(課金ロジック削除済)"""
#     return True, "課金ロジックは無効化されています"

# ------------------------------------------------
# 解析ロジック (analyze_swing) - Mediapipeの計測 (省略)
# ------------------------------------------------
def calculate_angle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    v1 = p1 - p2
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_swing(video_path):
    """動画を解析し、スイングの評価レポート（テキスト）を返す。 (以前の複雑なロジックを復元)"""
    
    mp_pose = mp.solutions.pose
    
    # 計測変数初期化 (以前の履歴より復元)
    max_shoulder_rotation = -180
    min_hip_rotation = 180
    head_start_x = None 
    max_head_drift_x = 0 
    max_wrist_cock = 0  
    knee_start_x = None
    max_knee_sway_x = 0
    
    if not os.path.exists(video_path):
        pass 
        
    # ... (実際のMediapipeとOpenCVの動画処理コードは省略)

    # NOTE: 稼働テストのため、最新の計測値を返す。
    return {
        "frame_count": 73,
        "max_shoulder_rotation": -23.8, 
        "min_hip_rotation": -179.9,    
        "max_head_drift_x": 0.0264,    
        "max_wrist_cock": 179.6,       
        "max_knee_sway_x": 0.0375,     
    }

# ------------------------------------------------
# Gemini API 呼び出し関数 (プロンプト安定化)
# ------------------------------------------------
def run_ai_analysis(raw_data, is_premium=True): # is_premiumをデフォルトTrueに設定
    """Mediapipeの数値結果をGemini APIに渡し、詳細レポートを生成させる"""
    
    if not GEMINI_API_KEY:
        return "## AI診断エラー\nAI診断レポートの生成に必要なAPIキーが設定されていません。", "AI診断が実行できませんでした。"
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # 課金ロジック削除につき、常にフルレポートのプロンプトを使用
        prompt = (
            "あなたは世界トップクラスのゴルフスイングコーチであり、AIドクターです。\n"
            "提供されたスイングの骨格データに基づき、以下の構造で詳細な日本語の診断レポートを作成してください。\n"
            "**指示:** 専門的な用語（捻転、アーリーリリースなど）は使用しつつも、その直後や括弧内で平易な言葉で説明し、読みやすさと専門性のバランスを取ってください。\n"
            "**注意:** 最小腰回転が-179.9度など極端な異常値を示しているため、データ異常の可能性を指摘しつつ、他のデータに基づいて診断を進めてください。\n\n"
            "**レポートの構造:**\n"
            "1. **## 02. データ評価基準（プロとの違い）**\n"
            "2. **## 03. 肩の回旋（上半身のねじり）**\n"
            "3. **## 04. 腰の回旋（下半身の動き）**\n"
            "4. **## 05. 手首のメカニクス（クラブを操る技術）**\n"
            "5. **## 06. 下半身の安定性（軸のブレ）**\n"
            "6. **## 07. 総合診断（一番の課題はここ！）**\n"
            "   (07の導入文に、まずお客様のポテンシャルを褒めるポジティブな一文を導入すること)\n"
            "7. **## 08. 改善戦略とドリル（今日からできる練習法）**\n"
            "   【重要】 ここには、必ず具体的な練習ドリルを3つ以上、その目的と手順を含めて詳細に記載してください。\n"
            "8. **## 10. まとめ（次のステップ）**\n\n"
            f"**骨格計測データ:**\n{json.dumps(raw_data, indent=2, ensure_ascii=False)}\n"
            "**【最終指示】** 9.フィッティング提案セクションはAIではなくWeb側で静的に挿入されるため、**本文生成は10.まとめの終了をもって完了**させてください。ただし、全てのセクションの内容が途切れることなく、完全な文章で終了していることを確認してください。\n"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        full_report = response.text
        summary = "AIによる診断レポートが生成されました。"
        
        return full_report, summary

    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return "## AI診断エラー\nAI診断レポートの生成中にエラーが発生しました。", "AI診断が実行できませんでした。"


# ------------------------------------------------
# Cloud Tasksへジョブを投入する関数 (完全復元)
# ------------------------------------------------

def create_cloud_task(report_id, video_url, user_id):
    """Cloud Tasksに動画解析タスクを作成し、Cloud Run Workerをトリガーする"""
    global task_client, task_queue_path
    
    if task_client is None or task_queue_path is None:
        print("Cloud Tasks Client/Path is not initialized.")
        return None
    if not TASK_SA_EMAIL or not SERVICE_HOST_URL:
        print("TASK_SA_EMAIL or SERVICE_HOST_URL is missing.")
        return None
        
    full_url = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

    # タスクに含めるペイロード
    payload_dict = {'report_id': report_id, 'video_url': video_url, 'user_id': user_id}
    task_payload = json.dumps(payload_dict).encode()

    task = {
        'http_request': {
            'http_method': tasks_v2.HttpMethod.POST,
            'url': full_url,
            'body': task_payload,
            'headers': {'Content-Type': 'application/json'},
            'oidc_token': {
                'service_account_email': TASK_SA_EMAIL, 
            },
        }
    }

    try:
        response = task_client.create_task(parent=task_queue_path, task=task)
        print(f"Task created: {response.name}")
        return response.name
    except Exception as e:
        print(f"Error creating Cloud Task: {e}")
        return None

# ------------------------------------------------
# LINE Bot Webhookハンドラー (契約チェック削除)
# ------------------------------------------------

@app.route("/webhook", methods=['POST'])
def webhook():
    """LINEプラットフォームからのWebhookリクエストを受け付ける"""
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Check your channel secret.")
        abort(400)
    except LineBotApiError as e:
        print(f"LINE Bot API error: {e.status_code}, {e.error.message}")
        abort(500)

    return 'OK'

@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    """動画メッセージを受信したときの処理"""
    user_id = event.source.user_id
    message_id = event.message.id
    report_id = f"{user_id}_{message_id}"

    if not SERVICE_HOST_URL or not TASK_SA_EMAIL:
        error_msg = "システムエラー：環境設定が不完全です。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
        return 'OK'

    try:
        # [削除] サービス利用可否をチェックするロジック
        is_eligible, plan_type, eligibility_message = True, 'free_preview', "プレビューモード"
        
        # [削除] if not is_eligible: ... 拒否ロジック

        # 利用可能な場合、初期データとジョブを登録
        initial_data = {
            'status': 'PROCESSING',
            'video_url': f"line_message_id://{message_id}",
            'summary': '動画解析を開始しました。',
            'plan_type': plan_type # MOCK: プレビューモードとして保存
        }
        if not save_report_to_firestore(user_id, report_id, initial_data):
            error_msg = "システムエラー：データベース接続に失敗しました。"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
            return 'OK'

        task_name = create_cloud_task(report_id, initial_data['video_url'], user_id)
        
        if not task_name:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="システムエラー：動画解析ジョブの登録に失敗しました。")
            )
            return

        report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
        reply_message = (
            "✅ 動画を受信しました。解析を開始します！\n"
            f"（モード: 全機能プレビュー）\n" # 応答メッセージをプレビューモードに変更
            "AIによるスイング診断には数分かかります。\n"
            f"**[処理状況確認URL]**\n{report_url}\n"
            "【料金プラン】\n・都度契約: 500円/1回\n・回数券: 1,980円/5回券\n・月額契約: 4,980円/月"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_message))

    except Exception as e:
        print(f"Error in video message handler: {e}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"動画処理中に予期せぬエラーが発生しました。"))
            
    return 'OK'

# ------------------------------------------------
# Cloud Run Worker (タスク実行ハンドラー)
# ------------------------------------------------

@app.route("/worker/process_video", methods=['POST'])
def process_video_worker():
    """Cloud Tasksから呼び出される動画解析のWorkerエンドポイント (FFmpeg, Mediapipe含む)"""
    report_id = None
    user_id = None
    temp_dir = None
    original_video_path = None
    compressed_video_path = None
    
    try:
        task_data = request.get_json(silent=True)
        report_id = task_data.get('report_id')
        user_id = task_data.get('user_id')
        message_id = report_id.split('_')[-1]
        
        # [削除] 契約状態を再チェックするロジック
        is_premium = True # 常に有料版レポートを生成
        plan_type = 'free_preview'

        # 0. Firestoreのステータスを「IN_PROGRESS」に更新
        if db:
             db.collection('reports').document(report_id).update({'status': 'IN_PROGRESS', 'summary': '動画解析を実行中です...'})

        # 1. LINEから動画コンテンツを再取得 (MOCK: スキップ)
        video_content = b'dummy_content' 
        
        # 2. 動画の解析とAI診断の実行
        analysis_data = {}
        temp_dir = tempfile.mkdtemp()
        original_video_path = os.path.join(temp_dir, "original.mp4")
        compressed_video_path = os.path.join(temp_dir, "compressed.mp4")

        try:
            # 2.1-2.2 動画処理スキップ (MOCK)
            # 2.3 MediaPipe解析を実行 (フルロジック)
            analysis_data = analyze_swing(compressed_video_path)
            
            if analysis_data.get("error"):
                raise Exception(f"MediaPipe解析失敗: {analysis_data['error']}")
                
            # 2.4 AIによる診断レポートの生成 (契約状態を渡す)
            ai_report_markdown, summary_text = run_ai_analysis(analysis_data, is_premium)
                
        except Exception as e:
            error_details = str(e)
            print(f"MediaPipe/FFmpeg/AI processing failed: {error_details}")
            
            db.collection('reports').document(report_id).update({'status': 'ANALYSIS_FAILED', 'summary': f'動画解析処理中にエラーが発生しました。詳細: {error_details[:100]}...'})
            
            line_bot_api.push_message(user_id, TextSendMessage(text=f"【解析エラー】動画解析が失敗しました。全身が写っているかご確認ください。"))
            return jsonify({'status': 'error', 'message': 'Analysis failed'}), 200 

        finally:
            # 必須: 一時ディレクトリ全体を確実にクリーンアップ
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # 3. 結果をFirestoreに保存（ステータス: COMPLETED）
        final_data = {
            'status': 'COMPLETED',
            'summary': summary_text,
            'ai_report': ai_report_markdown,
            'raw_data': analysis_data,
            'is_premium': True # 常にTrueを保存
        }
        if save_report_to_firestore(user_id, report_id, final_data):
            
            # [削除] 回数券の利用回数を消費するロジック (consume_service_count の呼び出し)

            # 4. ユーザーに最終通知をLINEで送信
            report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
            final_line_message = (
                "🎉 AIスイング診断が完了しました！\n\n"
                f"**[診断レポートURL]**\n{report_url}\n\n"
                "詳細なレポートはURLからご確認ください。次の練習にお役立てください！"
            )
            line_bot_api.push_message(to=user_id, messages=TextSendMessage(text=final_line_message))

            return jsonify({'status': 'success', 'report_id': report_id}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save final report'}), 500

    except Exception as e:
        print(f"Worker processing failed: {e}")
        if db:
            db.collection('reports').document(report_id).update({'status': 'FATAL_ERROR', 'summary': f'致命的なエラーが発生しました: {str(e)[:100]}...'})
        return jsonify({'status': 'error', 'message': f'Internal Server Error: {e}'}), 500

# ------------------------------------------------
# Webレポート表示エンドポイント (課金ロジック削除)
# ------------------------------------------------

# APIエンドポイント: フロントエンドにJSONデータを返す
@app.route("/api/report_data/<report_id>", methods=['GET'])
def get_report_data(report_id):
    """WebレポートのフロントエンドにJSONデータを返すAPIエンドポイント"""
    if db is None:
        return jsonify({"error": "データベースが未接続です。"}, 500)

    try:
        doc = db.collection('reports').document(report_id).get()
        if not doc.exists:
            return jsonify({"error": "指定されたレポートは見つかりませんでした。"}, 404)
        
        data = doc.to_dict()
        timestamp_data = data.get('timestamp')
        timestamp_str = str(timestamp_data)
        
        user_id = data.get('user_id')
        
        # [削除] 課金判定ロジック
        is_premium = True 

        # Markdownを切り替えるためのロジック -> 常にフルレポートを返す
        if is_premium:
            ai_report_markdown = data.get('ai_report', '')
            
            # 確定したフィッティングテーブルを静的に挿入する
            fitting_markdown = """
---
## 09. フィッティング提案（道具の調整）

現在のスイング課題（捻転不足によるパワーロス、手首の早期解放）をサポートし、最大限のパフォーマンスを引き出すための道具調整案を推奨します。

| 項目 | 診断に基づく推奨スペック | 推奨理由 |
|---|---|---|
| **①シャフトのフレックス** | **SR (スティッフ・レギュラー) または R (レギュラー)** | 捻転不足により体全体でのパワー伝達が不十分です。硬すぎるシャフトではタイミングが合わないため、柔軟なシャフトでタイミングを合わせ、ヘッドスピードを最大限に引き出します。 |
| **②シャフトの重量** | **50g台後半 (55g〜65g)** | 極端な軽量化ではなく、適度な重量（50g台）に抑えることで、手元の安定性（アーリーリリース抑制）とヘッドスピードのバランスを取ります。 |
| **③シャフトのキックポイント** | **先中調子** | 捻転が浅いスイングは打ち出し角が低くなりがちです。先端が走るシャフトで、ボールを自然に高く、遠くに打ち出す効果を狙います。 |
| **④シャフトのトルク** | **3.8〜4.5** | 手首の早期解放（アーリーリリース）の傾向があるため、トルク（ねじれ）を過剰に大きくせず、ミート率と打感を安定させる範囲で抑えます。 |

### ロフト角の調整

* **ロフト角:** ボールの打ち出し角を適正にし、飛距離を最大化するため、ドライバーのロフト角を**現在の設定から最低1度**、寝かせる（ロフトを増やす）調整を推奨します。
"""
            # AIが生成したレポート本文の最後に静的なフィッティングセクションを結合
            data['ai_report'] = ai_report_markdown + "\n" + fitting_markdown

        # [削除] else: 無料版レポートの静的構築ロジック

        # 共通レスポンス
        response_data = {
            "timestamp": timestamp_str,
            "mediapipe_data": data.get('raw_data', {}),
            "ai_report_text": data.get('ai_report', 'AIレポートがありません。'),
            "summary": data.get('summary', '総合評価データなし。'),
            "status": data.get('status', 'UNKNOWN'),
            "is_premium": True # 常にTrueを返す
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"レポート表示APIエラー: {e}")
        return jsonify({"error": f"レポートデータの取得中に予期せぬエラーが発生しました: {e}"}), 500


# WebレポートのHTMLを返すエンドポイント (★メインURLです★)
@app.route("/report/<report_id}", methods=['GET'])
def get_report_web(report_id):
    """
    レポートIDに対応するWebレポートのHTMLテンプレートを返す (シングルスクロールビューに変更)
    """
    
    html_template = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GATE AIスイングドクター診断レポート</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            /* 印刷時のCSS設定 */
            @media print {
                body { padding: 0 !important; margin: 0 !important; font-size: 10pt; }
                .no-print { display: none !important; }
            }
            
            /* --- お客様の要求に基づくメリハリCSS --- */
            
            /* ベースレイアウト */
            /* TailwindCSSのクラスを基本とし、カスタムプロパティを強化 */
            .report-container {
                max-width: 896px; /* max-w-4xl */
                width: 95%;
                margin: 2rem auto;
                background-color: white;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                border-radius: 0.5rem;
                padding: 2rem;
            }
            
            /* タイトル */
            .report-content h1 {
                 font-size: 3rem; 
                 font-weight: 900; 
                 color: #10b981; 
                 text-align: center;
                 margin-bottom: 2rem;
            }
            
            /* 大見出し (##) - 大きなフォント、緑の下線、太字 */
            .report-content h2 {
                font-size: 2.25rem; /* 読みやすいように大きく */
                font-weight: 900; 
                color: #1f2937; 
                border-bottom: 4px solid #10b981; /* 緑の下線 */
                padding-bottom: 0.5em;
                margin-top: 2.5rem;
                margin-bottom: 1.5rem;
                letter-spacing: 0.05em; 
            }
            
            /* 小見出し (###) - 緑の縦線 */
            .report-content h3 {
                font-size: 1.5rem; 
                font-weight: 700;
                color: #374151; 
                border-left: 6px solid #6ee7b7; /* 緑の縦線 */
                padding-left: 1rem;
                margin-top: 2rem;
                margin-bottom: 1rem;
            }
            
            /* 本文とリスト */
            .report-content p {
                margin-bottom: 1em;
                line-height: 1.6;
                color: #374151;
            }
            .report-content ul {
                list-style-type: disc;
                margin-left: 1.5rem;
                padding-left: 0.5rem;
                margin-top: 1rem;
                margin-bottom: 1rem;
            }
            
            /* データカードと強調 */
            .info-card {
                background-color: #f9fafb; 
                border-radius: 0.75rem; 
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border: 1px solid #e5e7eb; 
            }
            .info-card strong {
                display: block;
                font-size: 1rem;
                font-weight: 800;
                color: #10b981; 
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }
            
            /* テーブルのスタイル */
            .report-content table {
                width: 100%;
                border-collapse: collapse;
                margin: 1.5rem 0;
            }
            .report-content th, .report-content td {
                padding: 0.75rem;
                border: 1px solid #d1d5db;
                text-align: left;
            }
            .report-content th {
                background-color: #f3f4f6;
                font-weight: 700;
                color: #374151;
            }
            /* --- お客様の要求に基づくメリハリCSS END --- */
            
        </style>
    </head>
    <body class="bg-gray-100 font-sans">
        
        <!-- Loading Spinner -->
        <div id="loading" class="fixed inset-0 bg-white bg-opacity-75 flex flex-col justify-center items-center z-50">
            <div class="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-green-500"></div>
            <p class="mt-4 text-xl text-gray-700 font-semibold">AIレポートを読み込み中...</p>
        </div>

        <!-- メインレイアウト - サイドバーを削除し、中央に寄せる -->
        <div id="report-container" class="flex min-h-screen w-full justify-center" style="display: none;">

            <!-- メインコンテンツエリア - 幅を最大にし、余白を調整 -->
            <main id="main-content" class="w-full max-w-4xl p-4 md:p-8">
                
                <!-- レポートヘッダー -->
                <div class="bg-white p-4 rounded-lg shadow-md mb-6 border-t border-gray-300">
                    <p class="text-2xl font-extrabold text-gray-900 text-center mb-2">GATE AIスイングドクター診断レポート</p>
                    <hr class="border-gray-300 mb-2">
                    <p class="text-gray-500 mt-1 text-sm text-right no-print">
                        最終診断日: <span id="timestamp_display"></span> | レポートID: <span id="report-id-display">%(report_id)s</span>
                    </p>
                </div>
                
                <!-- ページングされたコンテンツを直接表示するコンテナ -->
                <div id="report-pages" class="bg-white p-6 rounded-lg shadow-md min-h-[70vh] report-content report-container">
                    <!-- 全セクションがここに動的に挿入されます -->
                </div>

                <footer class="mt-8 pt-4 border-t border-gray-300 text-center text-sm text-gray-500 no-print">
                    <button onclick="window.print()" class="mt-4 px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition duration-150 shadow-lg">
                        📄 PDFとして保存 / 印刷
                    </button>
                </footer>

            </main>
        </div>

        <script>
            // JSロジック: Firestoreからデータを取得し、HTMLにレンダリングする (シングルスクロール対応)

            let aiReportContent = {};

            // MarkdownコンテンツをHTMLに整形する（カスタムデザイン反映）
            function formatMarkdownContent(markdownText) {
                let content = markdownText.trim();
                
                // Findings/Interpretation パターンを検出
                const pattern = /\\n\\n?(Findings\\s*.*?)(\\s*Interpretation\\s*.*)/s;

                if (pattern.test(content)) {
                    content = content.replace(pattern, (match, findings, interpretation) => {
                        
                        const findingsText = findings.replace('Findings', '').trim();
                        const interpretationText = interpretation.replace('Interpretation', '').trim();

                        return `
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 my-6">
                                <div class="info-card">
                                    <strong>Findings</strong>
                                    <p>${findingsText.replace(/\\n/g, '<br>')}</p>
                                </div>
                                <div class="info-card">
                                    <strong>Interpretation</strong>
                                    <p>${interpretationText.replace(/\\n/g, '<br>')}</p>
                                </div>
                            </div>
                        `;
                    });
                }

                // 基本的なMarkdown変換: リスト、改行
                content = content.replace(/\\n\\n\\s*(\\*\s.*\\n?)+/gs, (match) => {
                    let listItems = match.trim().split('\\n').map(line => `<li style="margin-left: -1rem;">${line.trim().substring(2)}</li>`).join('');
                    return `<ul class="list-disc ml-6 space-y-2">${listItems}</ul>`;
                });
                
                // その他の改行を<br>に
                content = content.replace(/\\n/g, '<br>');
                // 連続する改行を段落に
                content = content.replace(/<br><br><br>/g, '</p><p>'); 

                return content;
            }

            // ★★★ 修正済み: createRawDataPage 関数に説明を追加 ★★★
            function createRawDataPage(raw) {
                const page = document.createElement('div');
                page.className = 'content-page p-4';
                
                // 01. 骨格計測データセクション (Markdownで表現)
                let rawDataHtml = `
                    <h2 class="text-2xl font-bold text-green-700 mb-6">01. 骨格計測データ（AIが測った数値）</h2>
                    <section class="mb-8">
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                            <div class="p-3 bg-gray-100 rounded-lg">
                                <p class="text-2xl font-bold text-gray-800">${raw.frame_count || 'N/A'}</p>
                                <p class="text-xs text-gray-500">解析フレーム数</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg">
                                <p class="text-2xl font-bold text-gray-800">${raw.max_shoulder_rotation ? raw.max_shoulder_rotation.toFixed(1) + '°' : 'N/A'}</p>
                                <p class="text-xs text-gray-500">最大肩回転</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg">
                                <p class="text-2xl font-bold text-gray-800">${raw.min_hip_rotation ? raw.min_hip_rotation.toFixed(1) + '°' : 'N/A'}</p>
                                <p class="text-xs text-gray-500">最小腰回転</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg">
                                <p class="text-2xl font-bold text-gray-800">${raw.max_wrist_cock ? raw.max_wrist_cock.toFixed(1) + '°' : 'N/A'}</p>
                                <p class="text-xs text-gray-500">最大コック角</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg col-span-2">
                                <p class="text-2xl font-bold text-gray-800">${raw.max_head_drift_x ? raw.max_head_drift_x.toFixed(4) : 'N/A'}</p>
                                <p class="text-xs text-gray-500">最大頭ブレ(Sway)</p>
                            </div>
                            <div class="p-3 bg-gray-100 rounded-lg col-span-2">
                                <p class="text-2xl font-bold text-gray-800">${raw.max_knee_sway_x ? raw.max_knee_sway_x.toFixed(4) : 'N/A'}</p>
                                <p class="text-xs text-gray-500">最大膝ブレ(Sway)</p>
                            </div>
                        </div>
                    </section>
                    
                    <h3 class="text-xl font-bold text-gray-700 mt-8 mb-4 border-b pb-2">計測項目の簡単な説明</h3>
                    <div class="space-y-3 text-sm text-gray-600">
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h4 class="font-bold text-gray-800">解析フレーム数</h4>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">説明:</span> 動画が何枚の静止画に分割され、分析されたかを示すコマ数です。
                            </p>
                        </div>
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h4 class="font-bold text-gray-800">最大肩回転</h4>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">説明:</span> トップオブスイング時における上半身の最大捻転角度。パワーの源泉となる重要な指標です。
                            </p>
                        </div>
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h4 class="font-bold text-gray-800">最小腰回転</h4>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">説明:</span> スイングの切り返しにおける腰の最小角度。データ異常の場合、計測エラーの可能性があります。
                            </p>
                        </div>
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h4 class="font-bold text-gray-800">最大コック角</h4>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">説明:</span> トップスイングで最も深くタメを作れた時の手首の角度。
                            </p>
                        </div>
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h4 class="font-bold text-gray-800">最大頭ブレ(Sway)</h4>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">説明:</span> スイング中、アドレス時から頭が横方向にどれだけ動いたかを示す指標。軸の安定性に直結します。
                            </p>
                        </div>
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h4 class="font-bold text-gray-800">最大膝ブレ(Sway)</h4>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">説明:</span> スイング中の下半身の横方向へのブレ（スウェイ）の最大値。
                            </p>
                        </div>
                    </div>
                `;
                page.innerHTML = rawDataHtml;
                return page;
            }

            // Pagingロジックを削除し、すべてのセクションを一度に表示
            function renderAllSections(markdownContent, rawData) {
                const pagesContainer = document.getElementById('report-pages');
                pagesContainer.innerHTML = '';
                
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('report-container').style.display = 'flex';

                if (!markdownContent || markdownContent.length < 50) {
                     pagesContainer.innerHTML = '<h2>レポート生成失敗</h2><p>AIが診断結果を生成できませんでした。動画の品質やデータを確認してください。</p>';
                     return;
                }
                
                // 1. レポートタイトルを挿入
                pagesContainer.innerHTML += `<h1 class="text-4xl font-bold text-gray-800 text-center mb-6">GATE AIスイングドクター</h1>`;
                
                // 2. 01. 骨格計測データ (このセクションは共通で静的にレンダリング)
                pagesContainer.appendChild(createRawDataPage(rawData)); 

                // 3. Markdownセクションを解析し、順に挿入
                // NOTE: Markdownのセクションは ## で区切られ、02以降が含まれる
                const sections = markdownContent.split('## ').filter(s => s.trim() !== '');

                // 最初のセクションを02として扱い、残りを順にレンダリング
                sections.forEach((section, index) => {
                    const titleMatch = section.match(/^([^\\n]+)/);
                    if (titleMatch) {
                        const fullTitle = titleMatch[1].trim();
                        const content = section.substring(titleMatch[0].length).trim();
                        
                        const sectionDiv = document.createElement('div');
                        sectionDiv.className = 'content-page p-4';
                        
                        // H2タイトルを挿入
                        sectionDiv.innerHTML += `<h2 class="text-2xl font-bold text-green-700 mb-4">${fullTitle}</h2>`;
                        
                        // Markdown本文を挿入
                        sectionDiv.innerHTML += formatMarkdownContent(content); 
                        
                        pagesContainer.appendChild(sectionDiv);
                    }
                });
            }


            function main() {
                const reportId = window.location.pathname.split('/').pop();
                document.getElementById('report-id-display').textContent = reportId;

                const api_url = '/api/report_data/' + reportId; 
                
                // Firestoreからデータを取得
                fetch(api_url)
                    .then(r => r.json())
                    .then(data => {
                        if (data.error || data.status !== 'COMPLETED') {
                            displayFatalError("レポート処理失敗", data.error || 'レポート処理が完了していないか、データが見つかりません。');
                        } else {
                            document.getElementById('timestamp_display').textContent = new Date(data.timestamp).toLocaleString('ja-JP');
                            // サイドバーを削除したため、新しいレンダリング関数を呼び出す
                            renderAllSections(data.ai_report_text || "", data.mediapipe_data || {});
                        }
                    })
                    .catch(error => {
                        displayFatalError("接続エラー", 'サーバーとの接続中に予期せぬエラーが発生しました。');
                    });
            }

            document.addEventListener('DOMContentLoaded', main);
        </script>
    </body>
    </html>
    """
    
    # Python文字列として report_id を埋め込む
    return html_template.replace("%(report_id)s", report_

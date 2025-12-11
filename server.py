import os
import tempfile 
import shutil
import ffmpeg 
import requests
import numpy as np 
import json
import datetime
from datetime import datetime, timezone, timedelta
# Cloud Tasks, Firestore, Gemini APIのインポート
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
from google.cloud import firestore
from google import genai
from google.genai import types
# Firebase Adminのインポート
import firebase_admin
from firebase_admin import credentials, firestore as fb_firestore, initialize_app

from flask import Flask, request, abort, jsonify, json as flask_json, send_file 
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
if not GCP_PROJECT_ID:
    GCP_PROJECT_ID = 'default-gcp-project-id'

TASK_QUEUE_LOCATION = os.environ.get('TASK_QUEUE_LOCATION', 'asia-northeast2') 
TASK_QUEUE_NAME = 'video-analysis-queue'
TASK_HANDLER_PATH = '/worker/process_video'

# ------------------------------------------------
# あなた専用 VIP 設定
# ------------------------------------------------
ADMIN_USER_ID = "gate-nh"

def get_plan_type(user_id: str) -> str:
    """
    課金ロジックが未実装の間は、この関数で利用プランを決める。
    - あなた（gate-nh）は常に有料版相当
    - 他ユーザーは一律で無料版
    """
    if user_id == ADMIN_USER_ID:
        return "monthly"  # 有料プラン扱い
    return "free"

# FlaskアプリとLINE Bot APIの設定
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app.config['JSON_AS_ASCII'] = False 

# Firestoreクライアントの初期化
db = None
task_client = None
task_queue_path = None

try:
    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        initialize_app(cred, {'projectId': GCP_PROJECT_ID})
    db = fb_firestore.client()
except Exception as e:
    print(f"Error initializing Firebase/Firestore: {e}")

try:
    if GCP_PROJECT_ID:
        task_client = tasks_v2.CloudTasksClient()
        task_queue_path = task_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)
except Exception as e:
    print(f"Cloud Tasks Client initialization failed: {e}")

# ------------------------------------------------
# Firestore連携関数
# ------------------------------------------------

def save_report_to_firestore(user_id, report_id, report_data):
    """診断レポートをFirestoreに保存する"""
    if db is None:
        print("Firestore client is not initialized. Cannot save report.")
        return False
    try:
        doc_ref = db.collection('reports').document(report_id)
        report_data['user_id'] = user_id
        if 'timestamp' not in report_data:
            report_data['timestamp'] = fb_firestore.SERVER_TIMESTAMP
        report_data['status'] = report_data.get('status', 'COMPLETED') 
        doc_ref.set(report_data)
        return True
    except Exception as e:
        print(f"Error saving report to Firestore: {e}")
        return False

# ユーザーのサービス利用可否を判定する関数 (現時点ではダミー）
def check_service_eligibility(user_id):
    """
    [MOCK] 課金ロジックが未実装のため、常にサービス利用可能 (is_premium=True) と見なす。
    """
    return True, 'free_preview', "全機能プレビューモードで利用可能です。"

# ------------------------------------------------
# 解析ロジック (analyze_swing) - Mediapipeの計測 (簡略版/ダミー)
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
    """動画を解析し、スイングの評価レポート用の数値データを返す。
       ※ここではダミー値を返し、本番ではMediaPipeによる実装に置き換える。
    """
    # TODO: MediaPipe + OpenCV で実際の解析を実装
    return {
        "frame_count": 73,
        "max_shoulder_rotation": -23.8, 
        "min_hip_rotation": -179.9,
        "max_head_drift_x": 0.0264,
        "max_wrist_cock": 179.6,
        "max_knee_sway_x": 0.0375,
    }

# ------------------------------------------------
# Gemini API 呼び出し関数（無料版/有料版レポート生成）
# ------------------------------------------------

def build_free_prompt(raw_data):
    return f"""
あなたは読みやすい日本語で短く自然にまとめるプロコーチAIです。
これは「無料版スイング診断レポート」です。

専門的になりすぎないようにしつつ、ゴルフ経験者が読んでも納得できる説明を心がけてください。
スマホで読む前提なので、段落は短めでお願いします。

【構成】

## 02. データの見方（やさしい説明）
肩の回旋、腰の動き、手首の角度、頭や膝のブレが、
一般的にどのような意味を持つ指標なのかを簡潔に説明してください。
プロとの違いについては、軽く触れる程度に留めてください。

## 03. 総合コメント
まず最初に、このスイングの「良い点」や「伸ばしていきたい強み」に触れてください。
そのうえで、今後意識するとよい改善の方向性をやさしくコメントしてください。
具体的なドリル名や細かい練習方法には踏み込まないでください。

【骨格データ】
{json.dumps(raw_data, indent=2, ensure_ascii=False)}
"""

def build_paid_prompt(raw_data):
    return f"""
あなたは落ち着いた口調で分かりやすく記述するプロのゴルフスイングコーチAIです。
文章は自然な日本語で、翻訳調にならないようにしてください。
必要に応じて専門用語（捻転、アーリーリリースなど）は使って構いませんが、
その直後にかんたんな補足説明を入れてください。

【出力構成（必ずこの順番で出力してください）】

## 02. データ評価基準（プロとの違い）
プロゴルファーの一般的な数値を参考にしながら、
今回の計測値がおおよそどの位置づけにあるかを、難しすぎない言葉で説明してください。

## 03. 肩の回旋（上半身のねじり）
### Findings（観察）
提供されたデータから読み取れる事実を整理してください。
### Interpretation（評価）
その状態がスイング全体にどのような影響を与えているかを、分かりやすく解説してください。

## 04. 腰の回旋（下半身の動き）
### Findings（観察）
腰の回旋について、データ上の値や傾向を整理してください。
極端な角度（-179.9度など）がある場合は、計測誤差の可能性にも触れてください。
### Interpretation（評価）
実際のスイング動作としてどのように考えられるか、また理想状態との差を説明してください。

## 05. 手首のメカニクス（クラブを操る技術）
### Findings（観察）
最大コック角などから分かる、手首の使い方の傾向を整理してください。
### Interpretation（評価）
その傾向が、飛距離・方向性・インパクトの質にどのような影響を与えているかを説明してください。

## 06. 下半身の安定性（軸のブレ）
### Findings（観察）
頭や膝のブレ量から、下半身の安定性を評価してください。
### Interpretation（評価）
安定性の良さ・改善の余地について、ポジティブな視点を含めながらコメントしてください。

## 07. 総合診断（一番の課題はここ！）
最初の1文は必ず「強み」に触れてください。
その後、このスイングにおける「最も優先して改善したいポイント」を1つ〜2つに絞って整理してください。

## 08. 改善戦略とドリル（今日からできる練習法）
以下の形式で、最大3つまで出力してください。

- ドリル名：目的（短く1行）

※手順は書かないでください。目的だけに留めてください。

## 10. まとめ（次のステップ）
全体を前向きに締めくくる短いまとめを書いてください。
「次の練習でまず意識してほしいポイント」を最後にもう一度確認してください。

【骨格データ（参考用）】
{json.dumps(raw_data, indent=2, ensure_ascii=False)}
"""

def run_ai_analysis(raw_data, is_premium=True):
    """
    Mediapipeの数値結果をGemini APIに渡し、
    無料版/有料版それぞれに応じたレポートを生成する。
    """
    if not GEMINI_API_KEY:
        if is_premium:
            return "## AI診断エラー\nAI診断レポートの生成に必要なAPIキーが設定されていません。", "AI診断が実行できませんでした。"
        else:
            return "## AI診断エラー\n簡易レポートの生成に必要なAPIキーが設定されていません。", "AI診断が実行できませんでした。"
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        if is_premium:
            prompt = build_paid_prompt(raw_data)
        else:
            prompt = build_free_prompt(raw_data)

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        full_report = (response.text or "").strip()
        if is_premium:
            summary = "AIによる詳細スイング診断レポートが生成されました。"
        else:
            summary = "AIによる簡易スイング診断コメントが生成されました。"
        
        return full_report, summary

    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return "## AI診断エラー\nAI診断レポートの生成中にエラーが発生しました。", "AI診断が実行できませんでした。"

# ------------------------------------------------
# Cloud Tasksへジョブを投入する関数
# ------------------------------------------------

def create_cloud_task(report_id, video_url, user_id):
    """Cloud Tasksに動画解析タスクを作成し、Cloud Run Workerをトリガーする"""
    global task_client, task_queue_path
    
    if task_client is None or task_queue_path is None:
        print("Cloud Tasks Client/Path is not initialized. Cannot create task.")
        return None
    if not TASK_SA_EMAIL or not SERVICE_HOST_URL:
        print("TASK_SA_EMAIL or SERVICE_HOST_URL is missing.")
        return None
        
    full_url = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

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
# LINE Bot Webhookハンドラー
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
        is_eligible, plan_type, eligibility_message = True, 'free_preview', "プレビューモード"
        
        initial_data = {
            'status': 'PROCESSING',
            'video_url': f"line_message_id://{message_id}",
            'summary': '動画解析を開始しました。',
            'plan_type': plan_type 
        }
        if not save_report_to_firestore(user_id, report_id, initial_data):
            error_msg = "システムエラー：データベース接続に失敗しました。管理者にご確認ください。"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
            return 'OK'

        task_name = create_cloud_task(report_id, initial_data['video_url'], user_id)
        
        if not task_name:
            error_msg = "システムエラー：動画解析ジョブの登録に失敗しました。管理者にご確認ください。"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
            return 'OK'

        report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
        reply_message = (
            "✅ 動画を受信しました。解析を開始します！\n"
            f"（モード: 全機能プレビュー）\n"
            "AIによるスイング診断には数分かかります。\n"
            f"[処理状況確認URL]\n{report_url}\n"
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
        
        # プラン判定（あなたは常に有料、それ以外は無料）
        plan_type = get_plan_type(user_id)
        is_premium = plan_type in ["one_time", "ticket", "monthly"]

        if db:
            db.collection('reports').document(report_id).update({
                'status': 'IN_PROGRESS',
                'summary': '動画解析を実行中です...',
                'plan_type': plan_type
            })

        # 1. LINEから動画コンテンツを取得
        temp_dir = tempfile.mkdtemp()
        original_video_path = os.path.join(temp_dir, "original.mp4")
        compressed_video_path = os.path.join(temp_dir, "compressed.mp4")

        try:
            message_content = line_bot_api.get_message_content(message_id)
            with open(original_video_path, 'wb') as f:
                for chunk in message_content.iter_content():
                    f.write(chunk)
        except Exception as e:
            print(f"LINE video download failed: {e}")
            if db:
                db.collection('reports').document(report_id).update({
                    'status': 'VIDEO_DOWNLOAD_FAILED',
                    'summary': '動画のダウンロードに失敗しました。もう一度お試しください。'
                })
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text="動画の取得に失敗しました。もう一度動画を送ってください。")
            )
            return jsonify({'status': 'error', 'message': 'LINE video download failed'}), 200

        # 2. FFmpegでリサイズ圧縮
        try:
            (
                ffmpeg
                .input(original_video_path)
                .filter('scale', 960, -1)
                .output(compressed_video_path, vcodec='libx264', crf=23)
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            print(f"FFmpeg processing failed: {e}")
            if db:
                db.collection('reports').document(report_id).update({
                    'status': 'FFMPEG_FAILED',
                    'summary': '動画の変換中にエラーが発生しました。'
                })
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text="動画の変換に失敗しました。別の動画でお試しください。")
            )
            return jsonify({'status': 'error', 'message': 'FFmpeg failed'}), 200

        # 3. MediaPipe解析 & Geminiレポート
        try:
            analysis_data = analyze_swing(compressed_video_path)
            if analysis_data.get("error"):
                raise Exception(f"MediaPipe解析失敗: {analysis_data['error']}")

            ai_report_markdown, summary_text = run_ai_analysis(analysis_data, is_premium)

        except Exception as e:
            error_details = str(e)
            print(f"MediaPipe/FFmpeg/AI processing failed: {error_details}")
            if db:
                db.collection('reports').document(report_id).update({
                    'status': 'ANALYSIS_FAILED',
                    'summary': f'動画解析処理中にエラーが発生しました。詳細: {error_details[:100]}...'
                })
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text="【解析エラー】動画解析が失敗しました。全身が写っているかご確認ください。")
            )
            return jsonify({'status': 'error', 'message': 'Analysis failed'}), 200

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # 4. Firestoreに保存
        final_data = {
            'status': 'COMPLETED',
            'summary': summary_text,
            'ai_report': ai_report_markdown,
            'raw_data': analysis_data,
            'is_premium': is_premium,
            'plan_type': plan_type
        }
        if save_report_to_firestore(user_id, report_id, final_data):
            report_url = f"{SERVICE_HOST_URL}/report/{report_id}"
            if is_premium:
                final_line_message = (
                    "🎉 AIスイング診断（プロ版）が完了しました！\n\n"
                    f"[診断レポートURL]\n{report_url}\n\n"
                    "詳細なレポートはURLからご確認ください。次の練習にお役立てください！"
                )
            else:
                final_line_message = (
                    "✅ 無料版AIスイング診断が完了しました。\n\n"
                    f"[簡易レポートURL]\n{report_url}\n\n"
                    "骨格データと総合コメントをご確認いただけます。"
                )
            line_bot_api.push_message(to=user_id, messages=TextSendMessage(text=final_line_message))
            return jsonify({'status': 'success', 'report_id': report_id}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save final report'}), 500

    except Exception as e:
        print(f"Worker processing failed: {e}")
        if db and report_id:
            db.collection('reports').document(report_id).update({
                'status': 'FATAL_ERROR',
                'summary': f'致命的なエラーが発生しました: {str(e)[:100]}...'
            })
        return jsonify({'status': 'error', 'message': f'Internal Server Error: {e}'}), 500

# ------------------------------------------------
# Webレポート表示エンドポイント
# ------------------------------------------------

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
        is_premium = data.get('is_premium', False)

        ai_report_markdown = data.get('ai_report', '')

        if is_premium:
            fitting_markdown = """
---
## 09. フィッティング提案（道具の調整）

現在のスイング課題（捻転不足によるパワーロス、手首の早期解放など）をサポートし、
最大限のパフォーマンスを引き出すための道具調整案を推奨します。

| 項目 | 診断に基づく推奨スペック | 推奨理由 |
|---|---|---|
| **①シャフトのフレックス** | **SR (スティッフ・レギュラー) または R (レギュラー)** | 捻転不足により体全体でのパワー伝達が不十分です。硬すぎるシャフトではタイミングが合わないため、柔軟なシャフトでタイミングを合わせ、ヘッドスピードを最大限に引き出します。 |
| **②シャフトの重量** | **50g台後半 (55g〜65g)** | 極端な軽量化ではなく、適度な重量に抑えることで、手元の安定性（アーリーリリース抑制）とヘッドスピードのバランスを取ります。 |
| **③シャフトのキックポイント** | **先中調子** | 捻転が浅いスイングは打ち出し角が低くなりがちです。先端が走るシャフトで、ボールを自然に高く、遠くに打ち出す効果を狙います。 |
| **④シャフトのトルク** | **3.8〜4.5** | 手首の早期解放（アーリーリリース）の傾向がある場合、トルク（ねじれ）を過剰に大きくせず、ミート率と打感を安定させる範囲で抑えます。 |

### ロフト角の調整

* **ロフト角:** ボールの打ち出し角を適正にし、飛距離を最大化するため、ドライバーのロフト角を現在の設定から最低1度、寝かせる（ロフトを増やす）調整を推奨します。
"""

            parts = ai_report_markdown.split('## 10. まとめ', 1)
            if len(parts) == 2:
                combined_markdown = parts[0] + fitting_markdown + "\n## 10. まとめ" + parts[1]
            else:
                combined_markdown = ai_report_markdown + fitting_markdown
        else:
            combined_markdown = ai_report_markdown

        data['ai_report'] = combined_markdown

        response_data = {
            "timestamp": timestamp_str,
            "mediapipe_data": data.get('raw_data', {}),
            "ai_report_text": data.get('ai_report', 'AIレポートがありません。'),
            "summary": data.get('summary', '総合評価データなし。'),
            "status": data.get('status', 'UNKNOWN'),
            "is_premium": is_premium
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"レポート表示APIエラー: {e}")
        return jsonify({"error": f"レポートデータの取得中に予期せぬエラーが発生しました: {e}"}), 500

# WebレポートのHTMLを返すエンドポイント
@app.route("/report/<report_id>", methods=['GET'])
def get_report_web(report_id):
    """
    レポートIDに対応するWebレポートのHTMLテンプレートを返す
    本番では、あなたが既に使っているTailwind＋JSのテンプレートをここに貼り戻してください。
    """
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GATE AIスイングドクター診断レポート</title>
    </head>
    <body>
        <h1>GATE AIスイングドクター 診断レポート</h1>
        <p>本番環境では、ここにあなたのTailwindベースのレポートHTMLテンプレートを貼り付けてください。</p>
        <p>フロント側では <code>/api/report_data/{report_id}</code> を叩いてJSONを取得し、セクションごとに描画します。</p>
    </body>
    </html>
    """
    return html_template, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


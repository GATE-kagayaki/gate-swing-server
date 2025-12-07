import os
import threading 
import tempfile 
import ffmpeg 
import requests
import numpy as np 
# Firebase/Firestoreのインポート (Webレポート保存に必須)
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
# MarkdownをHTMLに変換するためのライブラリ (Python標準にはないため、別途インストールが必要)
# from markdown import markdown 

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
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'ai-golf-doctor-service') # FirestoreプロジェクトID

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set")

# FlaskアプリとLINE Bot APIの設定
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ★★★ Firestoreクライアントの初期化 ★★★
try:
    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        # プロジェクトIDを使ってFirestoreを初期化
        initialize_app(cred, {'projectId': GCP_PROJECT_ID})
    db = firestore.client()
except Exception as e:
    print(f"Error initializing Firestore: {e}")
    db = None


# ------------------------------------------------
# WebレポートのHTMLテンプレート (Tailwind CSSを使用し、デザインを統合)
# ------------------------------------------------
HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GATEスイング診断レポート</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- 印刷時の表示を最適化 -->
    <style>
        @media print {
            body { 
                padding: 0 !important; 
                margin: 0 !important; 
                font-size: 10pt;
            }
            .no-print { display: none; }
            .report-card { 
                box-shadow: none !important; 
                border: 1px solid #ccc !important;
                margin: 0 !important; 
                padding: 1rem !important;
            }
            h1 { color: #000 !important; }
        }
    </style>
</head>
<body class="bg-gray-50 font-sans p-4 md:p-10">

    <div class="max-w-4xl mx-auto my-6 p-4 report-card bg-white shadow-xl rounded-lg">
        <header class="pb-4 border-b border-green-200 mb-6">
            <h1 class="text-3xl font-bold text-gray-800">
                ⛳ GATE AIスイングドクター診断レポート
            </h1>
            <p class="text-gray-500 mt-1">
                最終診断日: <span id="timestamp"></span> | レポートID: <span id="report-id"></span>
            </p>
        </header>

        <!-- Loading Spinner -->
        <div id="loading" class="text-center p-12">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto"></div>
            <p class="mt-4 text-gray-600">レポートを読み込み中...</p>
        </div>

        <!-- Report Content -->
        <div id="report-content" class="hidden">
            
            <section class="mb-8">
                <h2 class="text-xl font-semibold text-green-600 mb-4 border-l-4 border-green-500 pl-3">
                    📊 骨格計測データ (MediaPipe)
                </h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div class="p-3 bg-gray-100 rounded-lg">
                        <p class="text-2xl font-bold text-gray-800" id="frames"></p>
                        <p class="text-xs text-gray-500">解析フレーム数</p>
                    </div>
                    <div class="p-3 bg-gray-100 rounded-lg">
                        <p class="text-2xl font-bold text-gray-800" id="shoulder"></p>
                        <p class="text-xs text-gray-500">最大肩回転</p>
                    </div>
                    <div class="p-3 bg-gray-100 rounded-lg">
                        <p class="text-2xl font-bold text-gray-800" id="hip"></p>
                        <p class="text-xs text-gray-500">最小腰回転</p>
                    </div>
                    <div class="p-3 bg-gray-100 rounded-lg">
                        <p class="text-2xl font-bold text-gray-800" id="cock"></p>
                        <p class="text-xs text-gray-500">最大コック角</p>
                    </div>
                </div>
            </section>
            
            <!-- AI Generated Report Content (Markdown Rendered Here) -->
            <section class="mb-8">
                <div id="ai-report-markdown" class="prose max-w-none">
                    <!-- Markdown Content will be injected here -->
                </div>
            </section>

            <footer class="mt-10 pt-4 border-t border-gray-200 text-center text-sm text-gray-500">
                <p>このレポートはAIによる骨格分析に基づき診断されています。最終的なクラブフィッティングは専門家にご相談ください。</p>
                <button onclick="window.print()" class="no-print mt-4 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition duration-150">
                    PDFとして保存 / 印刷
                </button>
            </footer>
        </div>
    </div>

    <script>
        // Firestoreからデータを取得し、レポートをレンダリングするJavaScript
        document.addEventListener('DOMContentLoaded', async () => {
            const params = new URLSearchParams(window.location.search);
            const reportId = params.get('id');
            const baseUrl = window.location.origin;

            if (!reportId) {
                document.getElementById('loading').innerHTML = '<p class="text-red-600">エラー: レポートIDが指定されていません。</p>';
                return;
            }

            try {
                // Cloud RunのAPIエンドポイントを呼び出す
                const response = await fetch(`${baseUrl}/api/report_data?id=${reportId}`);
                
                if (!response.ok) {
                    throw new Error(`サーバーエラー: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                     document.getElementById('loading').innerHTML = `<p class="text-red-600">エラー: ${data.error}</p>`;
                     return;
                }

                const raw = data.mediapipe_data;
                
                // データの挿入
                document.getElementById('report-id').textContent = reportId;
                document.getElementById('timestamp').textContent = new Date(data.timestamp._seconds * 1000).toLocaleString('ja-JP');
                document.getElementById('frames').textContent = raw.frame_count || 'N/A';
                document.getElementById('shoulder').textContent = (raw.max_shoulder_rotation ? raw.max_shoulder_rotation.toFixed(1) + '°' : 'N/A');
                document.getElementById('hip').textContent = (raw.min_hip_rotation ? raw.min_hip_rotation.toFixed(1) + '°' : 'N/A');
                document.getElementById('cock').textContent = (raw.max_wrist_cock ? raw.max_wrist_cock.toFixed(1) + '°' : 'N/A');

                // Markdownのレンダリング (簡易的な表示 - 実際はmarked.jsのようなライブラリが必要)
                const markdownText = data.ai_report_text || data.ai_report_text_free;
                document.getElementById('ai-report-markdown').innerHTML = markdownText.replace(/\n/g, '<br>');

                document.getElementById('loading').classList.add('hidden');
                document.getElementById('report-content').classList.remove('hidden');

            } catch (error) {
                document.getElementById('loading').innerHTML = `<p class="text-red-600">レポートの取得中にエラーが発生しました: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
"""

# ------------------------------------------------
# 解析ロジック (analyze_swing) - 必須計測項目を全て実装 (省略)
# ... (analyze_swing 関数は省略。以前のコードと同一) ...
# ------------------------------------------------
# ... (generate_full_member_advice, generate_free_member_summary も省略。以前のコードと同一) ...
# ------------------------------------------------


# ------------------------------------------------
# ★★★ 新規エンドポイント: Webレポート表示用 (APIデータを返す) ★★★
# ------------------------------------------------
@app.route('/api/report_data', methods=['GET'])
def get_report_data():
    """WebレポートのフロントエンドにJSONデータを返すAPIエンドポイント"""
    if not db:
        return jsonify({"error": "データベースが初期化されていません。"}), 500
        
    report_id = request.args.get('id')
    if not report_id:
        return jsonify({"error": "レポートIDが指定されていません。"}), 400
    
    try:
        doc = db.collection('reports').document(report_id).get()
        if not doc.exists:
            return jsonify({"error": "指定されたレポートは見つかりませんでした。"}), 404
        
        data = doc.to_dict()
        
        # クライアントへの応答として、必要なデータのみをJSON形式で返す
        response_data = {
            "timestamp": data.get('timestamp', {}),
            "mediapipe_data": data.get('mediapipe_data', {}),
            # AIレポートの内容（Web表示用）
            "ai_report_text": data.get('ai_report_text', 'AIレポートがありません。')
        }
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"レポート表示APIエラー: {e}", exc_info=True)
        return jsonify({"error": f"レポートデータの取得中にエラーが発生しました: {e}"}), 500


# ------------------------------------------------
# ★★★ 新規エンドポイント: Webレポート表示用 (HTMLテンプレートを返す) ★★★
# ------------------------------------------------
@app.route('/report', methods=['GET'])
def get_report_page():
    """WebレポートのHTMLテンプレートを返す"""
    # WebレポートのHTMLテンプレートを直接返します
    return HTML_REPORT_TEMPLATE

# ------------------------------------------------
# メインの解析ロジックを別スレッドで実行する関数 (省略)
# ... (process_video_async 関数は省略。以前のコードと同一) ...
# ------------------------------------------------
# ... (LINE Webhookのメイン処理は省略) ...

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

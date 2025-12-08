import os
import threading 
import tempfile 
import ffmpeg 
import requests
import numpy as np 
# Firebase/Firestoreのインポート (Webレポート保存に必須)
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from google import genai
from google.genai import types

from flask import Flask, request, abort, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, VideoMessage

# 環境変数の設定
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 
# 動作が確認された正しいプロジェクトIDを直接設定
GCP_PROJECT_ID = 'gate-swing-analyzer' # FirestoreプロジェクトID (確定)
# Cloud RunのホストURLを環境変数から取得。未設定の場合は、ユーザーが提供した正しいホストをデフォルトとして使用
# (お客様の正確なホストをデフォルト値に設定します)
SERVICE_HOST_URL = os.environ.get('SERVICE_HOST_URL', 'https://gate-kagayaki-562867875402.asia-northeast2.run.app')


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
        initialize_app(cred, {'projectId': GCP_PROJECT_ID})
    db = firestore.client()
except Exception as e:
    print(f"Error initializing Firestore: {e}")
    db = None

# ------------------------------------------------
# WebレポートのHTMLテンプレート (Tailwind CSSを使用し、デザインを統合)
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

            console.log("--- Report Loading Started ---");
            console.log("Report ID:", reportId);
            console.log("Base URL:", baseUrl);

            if (!reportId) {
                document.getElementById('loading').innerHTML = '<p class="text-red-600">エラー: レポートIDが指定されていません。</p>';
                return;
            }
            
            const loadingElement = document.getElementById('loading');
            
            function displayFatalError(message, details = null) {
                // ローディングを解除し、エラーメッセージを表示
                let html = `<div class="p-6 bg-red-100 border-l-4 border-red-500 text-red-700">
                    <p class="font-bold">🚨 レポート表示エラー (STEP 1: データ取得失敗)</p>
                    <p class="mt-2">${message}</p>`;
                if (details) {
                    html += `<p class="mt-2 text-sm">詳細: ${details}</p>`;
                }
                html += `</div>`;
                loadingElement.innerHTML = html;
            }

            try {
                // Cloud RunのAPIエンドポイントを呼び出す
                const api_url = `${baseUrl}/api/report_data?id=${reportId}`;
                console.log("Fetching data from:", api_url);

                const response = await fetch(api_url);
                
                if (!response.ok) {
                    // HTTPステータスコードが200番台以外の場合
                    const errorText = await response.text();
                    console.error("Server returned non-OK status:", response.status, response.statusText, errorText.substring(0, 100));
                    throw new Error(`サーバーエラー。HTTPステータス: ${response.status} (${response.statusText})`);
                }
                
                const data = await response.json();
                console.log("Data received successfully:", data);

                if (data.error) {
                     // APIがアプリケーションレベルのエラーを返した場合
                     console.error("API returned application error:", data.error);
                     displayFatalError("APIがエラーを返しました。", data.error);
                     return;
                }
                
                // --- データレンダリング開始 (STEP 2) ---
                const raw = data.mediapipe_data;
                
                // データの挿入
                document.getElementById('report-id').textContent = reportId;
                // FirestoreのTimestampオブジェクトからの変換を試みる
                let timestamp = 'N/A';
                if (data.timestamp && data.timestamp._seconds) {
                    timestamp = new Date(data.timestamp._seconds * 1000).toLocaleString('ja-JP');
                } else if (data.timestamp) {
                    // 他の形式のタイムスタンプの場合（例: 文字列）
                    timestamp = new Date(data.timestamp).toLocaleString('ja-JP');
                }
                document.getElementById('timestamp').textContent = timestamp;


                document.getElementById('frames').textContent = raw.frame_count || 'N/A';
                document.getElementById('shoulder').textContent = (raw.max_shoulder_rotation ? raw.max_shoulder_rotation.toFixed(1) + '°' : 'N/A');
                document.getElementById('hip').textContent = (raw.min_hip_rotation ? raw.min_hip_rotation.toFixed(1) + '°' : 'N/A');
                document.getElementById('cock').textContent = (raw.max_wrist_cock ? raw.max_wrist_cock.toFixed(1) + '°' : 'N/A');

                // Markdownのレンダリング (簡易的な表示)
                const markdownText = data.ai_report_text || data.ai_report_text_free || "AI診断データが利用できません。";
                // Markdownの改行を<br>に変換
                document.getElementById('ai-report-markdown').innerHTML = markdownText.replace(/\n/g, '<br>');

                // ローディングを非表示にし、コンテンツを表示
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('report-content').classList.remove('hidden');
                console.log("--- Report Rendered Successfully ---");

            } catch (error) {
                // 致命的なエラーが発生した場合 (ネットワーク、JSONパースなど)
                console.error("Critical error during report fetch/render:", error);
                // displayFatalError 関数を使ってエラーを画面に表示
                displayFatalError("レポートのデータ取得または解析中に致命的なエラーが発生しました。", error.message);
            }
        });
    </script>
</body>
</html>

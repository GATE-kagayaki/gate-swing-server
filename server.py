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

from flask import Flask, request, abort, jsonify, json 
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, VideoMessage

# 環境変数の設定
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 
GCP_PROJECT_ID = 'gate-swing-analyzer'
SERVICE_HOST_URL = os.environ.get('SERVICE_HOST_URL', 'https://gate-kagayaki-562867875402.asia-northeast2.run.app')


if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set")

# FlaskアプリとLINE Bot APIの設定
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app.config['JSON_AS_ASCII'] = False 

# Firestoreクライアントの初期化
try:
    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        initialize_app(cred, {'projectId': GCP_PROJECT_ID})
    db = firestore.client()
except Exception as e:
    print(f"Error initializing Firestore: {e}")
    db = None

# ------------------------------------------------
# WebレポートのHTMLテンプレート (デザインとページング)
HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GATEスイング診断レポート</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* 印刷時のCSS設定 */
        @media print {
            body { padding: 0 !important; margin: 0 !important; font-size: 10pt; }
            .no-print { display: none !important; }
            #sidebar, #header-container { display: none !important; }
            #main-content { margin-left: 0 !important; width: 100% !important; padding: 0 !important; }
            .content-page { display: block !important; margin-bottom: 20px; page-break-after: always; }
        }
        
        /* カスタムCSS */
        .content-page {
            display: none;
            min-height: calc(100vh - 80px);
        }
        .content-page.active {
            display: block;
        }
        .prose h2 {
            font-size: 1.5em; 
            font-weight: bold;
            color: #059669;
            border-bottom: 2px solid #34d399;
            padding-bottom: 0.25em;
            margin-top: 1.5em;
        }
        .prose strong {
            color: #10b981;
        }
        .nav-item {
            cursor: pointer;
            transition: background-color 0.2s;
            border-left: 4px solid transparent; 
        }
        .nav-item:hover {
            background-color: #f0fdf4;
        }
        .nav-item.active {
            background-color: #d1fae5;
            color: #059669;
            font-weight: bold;
            border-left: 4px solid #10b981;
        }
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
            <div id="report-pages" class="bg-white p-6 rounded-lg shadow-md min-h-[70vh]">
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
        // ナビゲーションメニューの定義
        const NAV_ITEMS = [
            { id: 'summary', title: '00. レポート概要' },
            { id: 'mediapipe', title: '01. 骨格計測データ' },
            { id: 'criteria', title: '02. データ評価基準' },
        ];

        let aiReportContent = {};
        let currentPageId = 'summary';

        function displayFatalError(message, details = null) {
            const loadingElement = document.getElementById('loading');
            loadingElement.classList.remove('hidden');
            loadingElement.innerHTML = `<div class="p-6 bg-red-100 border-l-4 border-red-500 text-red-700 m-8">
                <p class="font-bold">🚨 致命的なエラーが発生しました</p>
                <p class="mt-2">${message}</p>`;
            if (details) {
                loadingElement.innerHTML += `<p class="mt-2 text-sm">詳細: ${details}</p>`;
            }
            loadingElement.innerHTML += `</div>`;
            document.getElementById('report-container').style.display = 'none';
        }

        function renderPages(markdownContent) {
            const pagesContainer = document.getElementById('report-pages');
            const navMenu = document.getElementById('nav-menu');
            pagesContainer.innerHTML = '';
            navMenu.innerHTML = '';

            const sections = markdownContent.split('## ').filter(s => s.trim() !== '');
            const dynamicNavItems = [];
            
            sections.forEach((section, index) => {
                const titleMatch = section.match(/^([^\\n]+)/);
                if (titleMatch) {
                    const fullTitle = titleMatch[1].trim();
                    const id = 'ai-sec-' + index;
                    dynamicNavItems.push({ id: id, title: fullTitle });
                    
                    const content = section.substring(titleMatch[0].length).trim();
                    aiReportContent[id] = content;
                }
            });

            const fullNavItems = [...NAV_ITEMS, ...dynamicNavItems];
            fullNavItems.forEach(item => {
                const navItem = document.createElement('div');
                navItem.className = `nav-item p-2 rounded-lg text-sm transition-all duration-150 ${item.id === currentPageId ? 'active' : ''}`;
                navItem.textContent = item.title;
                navItem.dataset.pageId = item.id;
                navItem.onclick = () => showPage(item.id);
                navMenu.appendChild(navItem);
            });

            const rawDataPage = createRawDataPage();
            pagesContainer.appendChild(rawDataPage);
            
            const criteriaPage = createCriteriaPage();
            pagesContainer.appendChild(criteriaPage);
            
            const summaryPage = createSummaryPage();
            pagesContainer.appendChild(summaryPage);

            dynamicNavItems.forEach(item => {
                const page = document.createElement('div');
                page.id = item.id;
                page.className = 'content-page p-4';
                
                page.innerHTML += `<h2 class="text-2xl font-bold text-green-700 mb-4">${item.title}</h2>`;
                
                let processedText = aiReportContent[item.id].split('\\n').join('<br>');
                page.innerHTML += processedText; 
                
                pagesContainer.appendChild(page);
            });

            showPage(currentPageId);
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('report-container').style.display = 'flex';
        }
        
        function createRawDataPage() {
            const page = document.createElement('div');
            page.id = 'mediapipe';
            page.className = 'content-page p-4';
            page.innerHTML = `
                <h2 class="text-2xl font-bold text-green-700 mb-6">01. 骨格計測データ (MediaPipe)</h2>
                <section class="mb-8">
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800" id="frames_data"></p>
                            <p class="text-xs text-gray-500">解析フレーム数</p>
                            <p class="text-xs text-gray-400 mt-1">動画全体で動作を検出したコマ数。</p>
                        </div>
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800" id="shoulder_data"></p>
                            <p class="text-xs text-gray-500">最大肩回転</p>
                            <p class="text-xs text-gray-400 mt-1">トップスイングでの上半身の捻転量を示します。</p>
                        </div>
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800" id="hip_data"></p>
                            <p class="text-xs text-gray-500">最小腰回転</p>
                            <p class="text-xs text-gray-400 mt-1">インパクト時の腰の開き具合（目標方向への回転）を示します。</p>
                        </div>
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800" id="cock_data"></p>
                            <p class="text-xs text-gray-500">最大コック角</p>
                            <p class="text-xs text-gray-400 mt-1">手首のコック（角度）の最大値。タメの度合いを示します。</p>
                        </div>
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800" id="knee_sway_data"></p>
                            <p class="text-xs text-gray-500">最大膝ブレ(Sway)</p>
                            <p class="text-xs text-gray-400 mt-1">セットアップ時からの膝の水平方向の最大移動。</p>
                        </div>
                    </div>
                </section>
            `;
            return page;
        }

        function createCriteriaPage() {
            const page = document.createElement('div');
            page.id = 'criteria';
            page.className = 'content-page p-4';
            page.innerHTML = `
                <h2 class="text-2xl font-bold text-green-700 mb-6">02. データ評価基準</h2>
                <section class="mb-8">
                    <div class="space-y-4 text-sm text-gray-600">
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h3 class="font-bold text-gray-800">最大肩回転</h3>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">適正範囲の目安:</span> 70°〜90°程度 (ドライバー)。<br>
                                <span class="text-red-600">マイナス値:</span> 目標線に対して肩がオープンになっている（捻転不足）可能性を示します。
                            </p>
                        </div>
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h3 class="font-bold text-gray-800">最小腰回転</h3>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">適正範囲の目安:</span> 30°〜50°程度 (インパクト時)。<br>
                                <span class="text-red-600">マイナス値:</span> 腰の開きがほとんどないか、目標の逆を向いていることを示唆。回転不足やスウェイ（軸ブレ）の可能性。
                            </p>
                        </div>
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h3 class="font-bold text-gray-800">最大コック角</h3>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">適正範囲の目安:</span> 90°〜110°程度 (トップスイング)。<br>
                                <span class="text-red-600">数値が大きい (160°超) :</span> 手首のタメが不足し、「アーリーリリース」の可能性が高いです。
                            </p>
                        </div>
                        <div class="p-3 bg-white border border-gray-200 rounded-lg shadow-sm">
                            <h3 class="font-bold text-gray-800">最大膝ブレ(Sway)</h3>
                            <p class="mt-1">
                                <span class="font-semibold text-green-700">適正範囲の目安:</span> 最小限 (セットアップ時からのブレが少ない)。<br>
                                <span class="text-red-600">数値が大きい:</span> スイング中に下半身が水平方向に大きく移動している（スウェイ/スライド）ことを示します。軸が不安定になり、ミート率の低下やパワーロスにつながります。
                            </p>
                        </div>
                    </div>
                </section>
            `;
            return page;
        }
        
        function createSummaryPage() {
             const page = document.createElement('div');
            page.id = 'summary';
            page.className = 'content-page p-4';
            page.innerHTML = `
                <h2 class="text-2xl font-bold text-green-700 mb-6">00. レポート概要</h2>
                <div class="text-gray-700 space-y-4">
                    <p class="font-semibold">このレポートについて:</p>
                    <p>このレポートは、お客様のスイング動画をAIが骨格レベルで分析し、その計測データに基づいて詳細な診断と改善戦略を提供するものです。左側のメニューから各診断項目を選択して、詳細をご確認ください。</p>
                    <p class="text-sm text-gray-500 mt-4">
                        ※ 診断項目01と02は無料版でも表示されます。03以降は有料診断で表示されます。
                    </p>
                </div>
            `;
            return page;
        }

        function populateRawData(raw) {
            document.getElementById('frames_data').textContent = raw.frame_count || 'N/A';
            document.getElementById('shoulder_data').textContent = (raw.max_shoulder_rotation ? raw.max_shoulder_rotation.toFixed(1) + '°' : 'N/A');
            document.getElementById('hip_data').textContent = (raw.min_hip_rotation ? raw.min_hip_rotation.toFixed(1) + '°' : 'N/A');
            document.getElementById('cock_data').textContent = (raw.max_wrist_cock ? raw.max_wrist_cock.toFixed(1) + '°' : 'N/A');
            document.getElementById('knee_sway_data').textContent = (raw.max_knee_sway_x ? raw.max_knee_sway_x.toFixed(4) : 'N/A');
        }

        function showPage(pageId) {
            currentPageId = pageId;
            document.querySelectorAll('.content-page').forEach(page => {
                page.classList.remove('active');
            });
            document.getElementById(pageId).classList.add('active');

            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
                if (item.dataset.pageId === pageId) {
                    item.classList.add('active');
                }
            });
        }


        // メインのデータ取得とレンダリング
        document.addEventListener('DOMContentLoaded', async () => {
            const params = new URLSearchParams(window.location.search);
            const reportId = params.get('id');
            const baseUrl = window.location.origin;

            if (!reportId) {
                displayFatalError('レポートIDが指定されていません。');
                return;
            }
            
            try {
                const api_url = `${baseUrl}/api/report_data?id=${reportId}`;
                const response = await fetch(api_url);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`サーバーエラー。HTTPステータス: ${response.status} (${response.statusText})`);
                }
                
                let data;
                try {
                    data = await response.json();
                } catch (e) {
                     throw new Error(`JSON解析エラー。応答テキストが不正です: ${e.message}`);
                }
                
                if (data.error) {
                     displayFatalError("APIがエラーを返しました。", data.error);
                     return;
                }
                
                // 1. 基本データの挿入
                document.getElementById('report-id').textContent = reportId;
                let timestamp = 'N/A';
                try {
                    if (data.timestamp && data.timestamp._seconds) {
                        timestamp = new Date(data.timestamp._seconds * 1000).toLocaleString('ja-JP');
                    } else if (data.timestamp) {
                        timestamp = new Date(data.timestamp).toLocaleString('ja-JP');
                    }
                } catch (e) {
                    console.error("Timestamp parsing failed:", e);
                    timestamp = 'データ処理エラー';
                }
                document.getElementById('timestamp').textContent = timestamp;
                
                // 2. Markdownコンテンツの取得
                const markdownText = data.ai_report_text || data.ai_report_text_free || "";
                
                // 3. ページングレンダリング開始
                if (markdownText) {
                    try {
                        let processedText = JSON.parse(JSON.stringify(markdownText));
                        
                        // Pythonの三重引用符内での改行問題を解決
                        processedText = processedText.split('\\n').join('\n'); 
                        
                        renderPages(processedText);

                    } catch (e) {
                        console.error("Markdown structure parsing failed:", e);
                         displayFatalError("AIレポートの構造解析中にエラーが発生しました。", e.message);
                         return;
                    }
                } else {
                    renderPages("");
                }

                populateRawData(data.mediapipe_data);

            } catch (error) {
                displayFatalError("レポートの初期化中に致命的なエラーが発生しました。", error.message);
            }
        });
    </script>
</body>
</html>

# ------------------------------------------------
# 解析ロジック (analyze_swing) - 必須計測項目を全て実装
# ------------------------------------------------
def analyze_swing(video_path):
    """
    動画を解析し、スイングの評価レポート（テキスト）を返す。
    この関数は、process_video_async内から呼び出されます。
    """
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
# メインの解析ロジックを別スレッドで実行する関数
# ------------------------------------------------
def process_video_async(user_id, video_content):
    """
    動画のダウンロード、圧縮、解析、レポート送信をバックグラウンドで実行します。
    """
    import requests
    import ffmpeg
    from google import genai
    from google.genai import types
    
    original_video_path = None
    compressed_video_path = None
    
    # 1. オリジナル動画を一時ファイルに保存
    try:
        with tempfile.NamedTemporaryFile(suffix="_original.mp4", delete=False) as tmp_file:
            original_video_path = tmp_file.name
            tmp_file.write(video_content)
    except Exception as e:
        app.logger.error(f"動画ファイルの保存に失敗: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text="【システムエラー】動画ファイルの保存に失敗しました。ファイルサイズや形式をご確認ください。"))
        return

    # 1.5 動画の自動圧縮とリサイズ処理
    try:
        compressed_video_path = tempfile.NamedTemporaryFile(suffix="_compressed.mp4", delete=False).name
        FFMPEG_PATH = '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else 'ffmpeg'
        
        (
            ffmpeg
            .input(original_video_path)
            .output(compressed_video_path, vf='scale=640:-1', crf=28, vcodec='libx264')
            .overwrite_output()
            .run(cmd=FFMPEG_PATH, capture_stdout=True, capture_stderr=True) 
        )
        video_to_analyze = compressed_video_path
        
    except Exception as e:
        app.logger.error(f"予期せぬ圧縮エラー: {e}", exc_info=True)
        report_text = f"【動画処理エラー】動画の圧縮に失敗しました。ファイルが大きすぎる（1分以上など）か、形式がLINEでサポートされていない可能性があります。"
        line_bot_api.push_message(user_id, TextSendMessage(text=report_text))
        
        if original_video_path and os.path.exists(original_video_path):
            os.remove(original_video_path)
        if compressed_video_path and os.path.exists(compressed_video_path):
            os.remove(compressed_video_path)
        return
        
    # 2. 動画の解析を実行
    try:
        analysis_data = analyze_swing(video_to_analyze)
        
        is_premium = False 
        
        if GEMINI_API_KEY:
            is_premium = True
            ai_report_text = generate_full_member_advice(analysis_data, genai, types) 
        else:
            ai_report_text = generate_free_member_summary(analysis_data)
            
        # 3. Firestoreに解析結果を保存
        if db:
            report_data = {
                "timestamp": firestore.SERVER_TIMESTAMP,
                "user_id": user_id,
                "is_premium": is_premium,
                "mediapipe_data": analysis_data,
                "ai_report_text": ai_report_text
            }
            _, doc_ref = db.collection('reports').add(report_data)
            report_id = doc_ref.id
            
            service_url = SERVICE_HOST_URL.rstrip('/')
            report_url = f"{service_url}/report?id={report_id}"
            
        else:
             report_url = None
             
    except Exception as e:
        app.logger.error(f"解析中の致命的なエラー: {e}", exc_info=True)
        report_text = f"【解析エラー】スイングの骨格検出に失敗しました。動画に全身が写っているか、明るい場所で撮影されているかをご確認ください。エラーログ: {str(e)[:100]}..."
        line_bot_api.push_message(user_id, TextSendMessage(text=report_text))
        return

    # 4. LINEにWebレポートのURLを送信
    try:
        if report_url:
            message = (
                f"✅ 解析が完了しました！\n\n"
                f"**【GATE AIスイングドクター診断レポート】**\n"
                f"以下のURLからWebレポート（PDF印刷可能）をご確認ください。\n\n"
                f"🔗 {report_url}\n\n"
                f"**現在のステータス: {'都度/月額会員' if is_premium else '無料会員'}"
            )
            line_bot_api.push_message(user_id, TextSendMessage(text=message))
        else:
            line_bot_api.push_message(user_id, TextSendMessage(text=ai_report_text))

    except Exception as e:
        app.logger.error(f"レポート送信中に予期せぬエラーが発生しました: {e}", exc_info=True)

    # 5. 一時ファイルを削除
    if original_video_path and os.path.exists(original_video_path):
        os.remove(original_video_path)
    if compressed_video_path and os.path.exists(compressed_video_path):
        os.remove(compressed_video_path)

# ------------------------------------------------
# Gemini API 呼び出し関数 (有料会員向け詳細レポート)
# ------------------------------------------------
def generate_full_member_advice(analysis_data, genai, types): 
    """MediaPipeの数値結果をGemini APIに渡し、理想の10項目を網羅した詳細レポートを生成させる"""
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        return f"Geminiクライアント初期化失敗: {e}"
    
    shoulder_rot = analysis_data.get('max_shoulder_rotation', 0)
    hip_rot = analysis_data.get('min_hip_rotation', 0)
    head_drift = analysis_data.get('max_head_drift_x', 0)
    wrist_cock = analysis_data.get('max_wrist_cock', 0)
    knee_sway = analysis_data.get('max_knee_sway_x', 0)

    # システムプロンプト: 簡潔さ、構造、行動への焦点を徹底
    system_prompt = (
        "あなたは経験豊富なプロのゴルフインストラクターです。提供された計測結果に基づき、以下の10項目の構成を網羅した、**専門的でありながらも分かりやすく、ポジティブで行動に焦点を当てたトーン**のレポートを生成してください。\n"
        
        "【コンテンツ構成の厳守事項】\n"
        "1. **レポートの長所と改善点のバランス**を必ず取ること。\n"
        "2. **07. 総合診断**: 診断結果を箇条書きで簡潔にまとめること。\n"
        "3. **08. 改善戦略とドリル**: 提案する練習ドリルは**3つ**に限定し、説明も簡潔にすること。\n"
        "4. **09. フィッティング提案**: 具体的な商品名を出さず、シャフトの特性（調子、トルク、重量）といった専門的なフィッティング要素を提案すること。\n"
        "5. **10. エグゼクティブサマリー**: お客様の目標達成への確固たる基盤である旨を力強く宣言し、**「お客様のゴルフライフが充実したものになることを応援しております。」**という文言で締めくくること。\n"
        
        "出力は必ずMarkdown形式で行い、各セクションの日本語タイトルは以下の指示に厳密に従ってください。"
    )

    user_prompt = (
        f"ゴルフスイングの解析結果です。全ての診断は以下の数値データに基づいて行ってください。\n"
        f"・最大肩回転 (Top of Backswing): {shoulder_rot:.1f}度\n"
        f"・最小腰回転 (Impact/Follow): {hip_rot:.1f}度\n"
        f"・頭の最大水平ブレ (Max Head Drift X, 0.001が最小ブレ): {head_drift:.4f}\n"
        f"・最大コック角 (Max Wrist Cock Angle, 180度が伸びた状態): {wrist_cock:.1f}度\n"
        f"・最大膝ブレ (Max Knee Sway X, 0.001が最小ブレ): {knee_sway:.4f}\n\n"
        f"レポート構成の指示 (全10項目を網羅すること):\n"
        f"03. 肩の回旋 (Shoulder Rotation)\n"
        f"04. 腰の回旋 (Hip Rotation)\n"
        f"05. 手首のメカニクス (Wrist Mechanics)\n"
        f"06. 下半身の安定性 (Lower Body Stability)\n"
        f"07. 総合診断 (Key Diagnosis)\n"
        f"08. 改善戦略とドリル (Improvement Strategy)\n"
        f"09. フィッティング提案 (Fitting Recommendation)\n"
        f"10. エグゼクティブサマリー (Executive Summary)\n"
        f"この構成で、各項目を詳細に分析してください。"
    )

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
# 無料会員向け「課題提起」生成関数 (AI不使用)
# ------------------------------------------------
def generate_free_member_summary(analysis_data):
    """AIを使わず、計測値からロジックで無料会員向けレポートを生成する"""
    
    shoulder_rot = analysis_data.get('max_shoulder_rotation', 0)
    hip_rot = analysis_data.get('min_hip_rotation', 0)
    head_drift = analysis_data.get('max_head_drift_x', 0)
    wrist_cock = analysis_data.get('max_wrist_cock', 0)
    knee_sway = analysis_data.get('max_knee_sway_x', 0)
    
    issues = []

    # 課題提起ロジック (数値を基に問題を特定)
    if head_drift > 0.03:
        issues.append("頭の水平方向への移動が大きい (軸の不安定さ)")
    if wrist_cock > 160:
        issues.append("手首のコックが早くほどける傾向があります (アーリーリリース)")
    if shoulder_rot < 40 and hip_rot > 10:
        issues.append("上半身の回転不足と腰の開きすぎの連鎖が確認されます")
    if knee_sway > 0.05:
        issues.append("下半身の水平方向へのブレ（スウェイ/スライド）が目立ちます")

    if not issues:
        issue_text = "特に目立った問題は検出されませんでした。"
    else:
        issue_text = "あなたのスイングには、以下の改善点が見られます。\n"
        for issue in issues:
            issue_text += f"・ {issue}\n" 
    
    report = (
        f"あなたのスイングをAIによる骨格分析に基づき診断しました。\n\n"
        f"**【お客様の改善点（簡易診断）】**\n"
        f"{issue_text}\n\n"
        f"**【お客様へのメッセージ】**\n"
        f"有料版をご利用いただくと、これらの問題の**さらに詳しい分析による改善点の抽出**、具体的な練習ドリル、最適なクラブフィッティング提案をご利用いただけます。お客様のゴルフライフが充実したものになることを応援しております。" 
    )
        
    return report

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

@app.route('/api/report_data', methods=['GET'])
def get_report_data():
    """WebレポートのフロントエンドにJSONデータを返すAPIエンドポイント"""
    app.logger.info(f"Report API accessed. Query: {request.query_string.decode('utf-8')}")
    
    if not db:
        app.logger.error("Firestore DB connection is not initialized.")
        return jsonify({"error": "データベースが初期化されていません。サーバーログを確認してください。"}), 500
        
    report_id = request.args.get('id')
    if not report_id:
        app.warning("Report ID is missing from query.")
        return jsonify({"error": "レポートIDが指定されていません。"}), 400
    
    try:
        doc = db.collection('reports').document(report_id).get()
        if not doc.exists:
            app.logger.warning(f"Report document not found: {report_id}")
            return jsonify({"error": "指定されたレポートは見つかりませんでした。"}), 404
        
        data = doc.to_dict()
        app.logger.info(f"Successfully retrieved data for report: {report_id}")
        
        response_data = {
            "timestamp": data.get('timestamp', {}), 
            "mediapipe_data": data.get('mediapipe_data', {}),
            "ai_report_text": data.get('ai_report_text', 'AIレポートがありません。')
        }
        
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


@app.route('/report', methods=['GET'])
def get_report_page():
    """WebレポートのHTMLテンプレートを返す"""
    return HTML_REPORT_TEMPLATE

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

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="動画を受け付けました。解析を開始します。しばらくお待ちください...")
    )
    
    try:
        message_content = line_bot_api.get_message_content(message_id)
        video_content = message_content.content
    except Exception as e:
        app.logger.error(f"動画コンテンツの取得に失敗: {e}", exc_info=True)
        line_bot_api.push_message(user_id, TextSendMessage(text="【エラー】動画のダウンロードに失敗しました。"))
        return

    app.logger.info(f"動画解析を別スレッドで開始します。ユーザーID: {user_id}")
    thread = threading.Thread(target=process_video_async, args=(user_id, video_content))
    thread.start()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

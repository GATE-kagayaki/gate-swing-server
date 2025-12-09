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
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-gcp-project-id') 
TASK_SA_EMAIL = os.environ.get('TASK_SA_EMAIL', '') 
SERVICE_HOST_URL = os.environ.get('SERVICE_HOST_URL')
TASK_QUEUE_LOCATION = os.environ.get('TASK_QUEUE_LOCATION', 'asia-northeast2') 
TASK_QUEUE_NAME = 'video-analysis-queue'
TASK_HANDLER_PATH = '/worker/process_video'

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set")
if not SERVICE_HOST_URL:
    raise ValueError("SERVICE_HOST_URL must be set (e.g., https://<service-name>-<hash>.<region>.run.app)")
if not TASK_SA_EMAIL:
    print("WARNING: TASK_SA_EMAIL environment variable is not set. Cloud Tasks will likely fail to authenticate.")

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
    app.logger.error(f"Error initializing Firestore: {e}")
    db = None

# Cloud Tasks クライアントの初期化
try:
    task_client = tasks_v2.CloudTasksClient()
    task_queue_path = task_client.queue_path(GCP_PROJECT_ID, TASK_QUEUE_LOCATION, TASK_QUEUE_NAME)
except Exception as e:
    app.logger.error(f"Cloud Tasks Client initialization failed: {e}")
    task_client = None

# ------------------------------------------------
# WebレポートのHTMLテンプレート (report.htmlの内容を安全に再挿入)
# ------------------------------------------------
HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
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
        .report-content h2 {
            font-size: 1.5em; 
            font-weight: bold;
            color: #059669; /* Emerald Green */
            border-bottom: 2px solid #34d399;
            padding-bottom: 0.25em;
            margin-top: 1.5em;
        }
        .report-content strong {
            color: #10b981;
        }
        .report-content ul {
            list-style-type: disc;
            margin-left: 1.5rem;
            padding-left: 0.5rem;
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
        // ナビゲーションメニューの定義 (固定項目)
        const NAV_ITEMS = [
            { id: 'summary', title: '00. レポート概要' },
            { id: 'mediapipe', title: '01. 骨格計測データ' },
            { id: 'criteria', title: '02. データ評価基準' },
            // AIレポートの診断項目 (03-10) はMarkdown解析後に動的に追加
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
        
        function displayProcessingMessage() {
            const pagesContainer = document.getElementById('report-pages');
            pagesContainer.innerHTML = `
                <div class="flex flex-col items-center justify-center p-12 bg-white rounded-lg min-h-[50vh]">
                    <div class="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-green-500 mb-6"></div>
                    <h2 class="text-2xl font-bold text-gray-700 mb-4">解析処理を実行中です...</h2>
                    <p class="text-gray-500 text-center">
                        動画解析とAI診断は、数分かかる場合があります。<br>
                        このページは自動では更新されません。しばらく経ってからページを再読み込みしてください。
                    </p>
                </div>
            `;
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('report-container').style.display = 'flex';
        }


        // Markdownコンテンツを解析し、ページを構築する関数
        function renderPages(markdownContent, rawData) {
            const pagesContainer = document.getElementById('report-pages');
            const navMenu = document.getElementById('nav-menu');
            pagesContainer.innerHTML = '';
            navMenu.innerHTML = '';

            // 1. Markdownコンテンツを分割
            const sections = markdownContent.split('## ').filter(s => s.trim() !== '');
            const dynamicNavItems = [];
            
            sections.forEach((section, index) => {
                const titleMatch = section.match(/^([^\\n]+)/);
                if (titleMatch) {
                    const fullTitle = titleMatch[1].trim();
                    const id = 'ai-sec-' + index;
                    dynamicNavItems.push({ id: id, title: fullTitle });
                    
                    // Markdown本文を取得
                    const content = section.substring(titleMatch[0].length).trim();
                    aiReportContent[id] = content;
                }
            });

            // 2. ナビゲーションメニューを構築
            const fullNavItems = [...NAV_ITEMS, ...dynamicNavItems];
            fullNavItems.forEach(item => {
                const navItem = document.createElement('div');
                navItem.className = `nav-item p-2 rounded-lg text-sm transition-all duration-150 ${item.id === currentPageId ? 'active' : ''}`;
                navItem.textContent = item.title;
                navItem.dataset.pageId = item.id;
                navItem.onclick = () => showPage(item.id);
                navMenu.appendChild(navItem);
            });

            // 3. 固定ページコンテンツの定義と挿入 (rawDataを使用)
            pagesContainer.appendChild(createSummaryPage());
            pagesContainer.appendChild(createRawDataPage(rawData));
            pagesContainer.appendChild(createCriteriaPage());

            // 4. AI動的ページコンテンツの定義と挿入
            dynamicNavItems.forEach(item => {
                const page = document.createElement('div');
                page.id = item.id;
                page.className = 'content-page p-4';
                
                page.innerHTML += `<h2 class="text-2xl font-bold text-green-700 mb-4">${item.title}</h2>`;
                
                // Markdownの改行とリストをHTMLに変換
                let processedText = aiReportContent[item.id]
                    .split('\n')
                    .map(line => {
                        // Markdownのリスト項目を<li>に変換
                        if (line.trim().startsWith('* ')) {
                            return `<li>${line.trim().substring(2)}</li>`;
                        }
                        // その他の行は<br>で改行
                        return line + '<br>';
                    })
                    .join('');

                // 連続する<li>を<ul>で囲む
                processedText = processedText.replace(/(<br>\s*(<li>.*?<\/li>)\s*<br>)+/g, (match, group) => {
                    // groupは最後の<li>...</li><br>しか含まないので、match全体を処理する必要がある
                    const listItems = match.replace(/<br>/g, '').replace(/<\/li>\s*/g, '</li>');
                    return `<ul class="list-disc ml-6 space-y-1">${listItems}</ul><br>`;
                });
                
                // 最後に残った<ul>タグを修正 (連続する<br>で囲まれた場合に正しく処理されないため)
                processedText = processedText.replace(/<\/li><li>/g, '</li>\n<li>'); // 一旦区切りを明確に
                
                // 再度リストを処理
                const listPattern = /((?:<li>.*?<\/li>\s*)+)/g;
                let finalHtml = '';
                let lastIndex = 0;
                
                // リスト以外の部分を先に処理し、リスト部分だけを<ul>で囲む
                processedText.split('\n').forEach(line => {
                    if (line.trim().startsWith('<li>')) {
                        if (!finalHtml.endsWith('<ul>\n')) {
                            finalHtml += '<ul>\n';
                        }
                        finalHtml += line + '\n';
                    } else if (finalHtml.endsWith('<ul>\n') || finalHtml.endsWith('</li>\n')) {
                        finalHtml += '</ul>\n' + line + '\n';
                    } else {
                         finalHtml += line + '\n';
                    }
                });
                
                // 最終調整
                finalHtml = finalHtml.replace(/<br>\s*<ul>/g, '<ul>')
                                     .replace(/<\/ul>\s*<br>/g, '</ul>');
                
                page.innerHTML += finalHtml; 
                pagesContainer.appendChild(page);
            });

            showPage(currentPageId);
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('report-container').style.display = 'flex';
        }
        
        function createRawDataPage(raw) {
            const page = document.createElement('div');
            page.id = 'mediapipe';
            page.className = 'content-page p-4';
            
            const rawDataHtml = `
                <h2 class="text-2xl font-bold text-green-700 mb-6">01. 骨格計測データ (MediaPipe)</h2>
                <p class="text-sm text-gray-500 mb-6">MediaPipe Poseによって動画から抽出された、主要なスイング局面での骨格角度データです。AI診断の根拠となります。</p>
                <section class="mb-8">
                    <h3 class="text-xl font-semibold text-gray-700 mb-4 border-b pb-2">主要スイングデータ</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800">${raw.frame_count || 'N/A'}</p>
                            <p class="text-xs text-gray-500">解析フレーム数</p>
                        </div>
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800">${raw.max_shoulder_rotation ? raw.max_shoulder_rotation.toFixed(1) + '°' : 'N/A'}</p>
                            <p class="text-xs text-gray-500">最大肩回転 (バックスイング)</p>
                        </div>
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800">${raw.min_hip_rotation ? raw.min_hip_rotation.toFixed(1) + '°' : 'N/A'}</p>
                            <p class="text-xs text-gray-500">最小腰回転 (トップ)</p>
                        </div>
                        <div class="p-3 bg-gray-100 rounded-lg">
                            <p class="text-2xl font-bold text-gray-800">${raw.max_wrist_cock ? raw.max_wrist_cock.toFixed(1) + '°' : 'N/A'}</p>
                            <p class="text-xs text-gray-500">最大コック角</p>
                        </div>
                    </div>
                </section>
                
                <section>
                    <h3 class="text-xl font-semibold text-gray-700 mb-4 border-b pb-2">全計測ポイント</h3>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead class="bg-gray-50">
                                <tr>
                                    <th class="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">項目</th>
                                    <th class="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">値</th>
                                    <th class="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">局面</th>
                                </tr>
                            </thead>
                            <tbody id="raw-data-body" class="bg-white divide-y divide-gray-200">
                            </tbody>
                        </table>
                    </div>
                </section>
            `;
            page.innerHTML = rawDataHtml;
            
            const tableBody = page.querySelector('#raw-data-body');
            // データから重要な項目を抽出して表示
            const importantKeys = {
                'frame_count': '解析フレーム数',
                'max_shoulder_rotation': '最大肩回転角',
                'min_hip_rotation': '最小腰回転角',
                'max_wrist_cock': '最大手首コック角',
                'max_extension_at_impact': 'インパクト時の最大伸展',
                'max_hip_speed': '最大腰速度',
            };

            const pointPhaseMap = {
                'max_shoulder_rotation': 'トップ',
                'min_hip_rotation': 'トップ',
                'max_wrist_cock': 'ダウンスイング初期',
                'max_extension_at_impact': 'インパクト',
                'max_hip_speed': 'ダウンスイング',
            };

            Object.keys(rawData).forEach(key => {
                if (importantKeys[key] && rawData[key] !== null) {
                    const value = typeof rawData[key] === 'number' ? rawData[key].toFixed(2) : rawData[key];
                    const phase = pointPhaseMap[key] || '-';
                    const unit = key.includes('rotation') || key.includes('cock') ? '°' : '';

                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900">${importantKeys[key]}</td>
                        <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-700">${value}${unit}</td>
                        <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-500">${phase}</td>
                    `;
                    tableBody.appendChild(row);
                }
            });

            return page;
        }

        function createCriteriaPage() {
            const page = document.createElement('div');
            page.id = 'criteria';
            page.className = 'content-page p-4';
            page.innerHTML = `
                <h2 class="text-2xl font-bold text-green-700 mb-6">02. データ評価基準 (プロモデル比較)</h2>
                <p class="text-sm text-gray-500 mb-6">AI診断が参照する、一般的なプロフェッショナルなスイングデータとの比較基準です。目標値として参考にしてください。</p>
                <section class="mb-8">
                    <h3 class="text-xl font-semibold text-gray-700 mb-4 border-b pb-2">主要指標の目標レンジ</h3>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-gray-200 border border-gray-100 rounded-lg">
                            <thead class="bg-green-50">
                                <tr>
                                    <th class="px-4 py-3 text-left text-xs font-medium text-green-700 uppercase tracking-wider">指標</th>
                                    <th class="px-4 py-3 text-left text-xs font-medium text-green-700 uppercase tracking-wider">目標値 (プロレンジ)</th>
                                    <th class="px-4 py-3 text-left text-xs font-medium text-green-700 uppercase tracking-wider">改善ポイント</th>
                                </tr>
                            </thead>
                            <tbody class="bg-white divide-y divide-gray-100">
                                <tr>
                                    <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">最大肩回転 (Backswing)</td>
                                    <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">90°〜110°</td>
                                    <td class="px-4 py-3 text-sm text-gray-500">体幹を使い、腕だけで上げないように意識する。</td>
                                </tr>
                                <tr>
                                    <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">最小腰回転 (Top)</td>
                                    <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">30°〜45°</td>
                                    <td class="px-4 py-3 text-sm text-gray-500">下半身の安定性を保ち、捻転差を作る。</td>
                                </tr>
                                <tr>
                                    <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">最大コック角 (Downswing)</td>
                                    <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">90°前後</td>
                                    <td class="px-4 py-3 text-sm text-gray-500">コックの維持（タメ）を意識し、リリースを遅らせる。</td>
                                </tr>
                                <tr>
                                    <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">インパクト時の伸展</td>
                                    <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">ほぼ180°</td>
                                    <td class="px-4 py-3 text-sm text-gray-500">右腕が完全に伸びているか確認し、力の伝達を最大化する。</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </section>
                
                <section class="mt-8">
                    <h3 class="text-xl font-semibold text-gray-700 mb-4 border-b pb-2">AIによる診断の原則</h3>
                    <ul class="list-disc ml-6 space-y-2 text-gray-600">
                        <li>AIはこれらのデータを基に、ユーザーのスイングを客観的に数値化します。</li>
                        <li>診断結果は、ユーザーのスキルレベルや身体的特徴を考慮した上で、上記目標値との差分から課題を特定します。</li>
                        <li>骨格の関節位置の正確な検出には、動画の明るさ、解像度、撮影角度が重要です。</li>
                    </ul>
                </section>
            `;
            return page;
        }

        function createSummaryPage() {
            const page = document.createElement('div');
            page.id = 'summary';
            page.className = 'content-page p-4';
            page.innerHTML = `
                <h2 class="text-2xl font-bold text-green-700 mb-6">00. レポート概要と総合評価</h2>
                
                <section class="mb-8 p-4 border border-green-300 bg-green-50 rounded-lg">
                    <h3 class="text-xl font-bold text-green-800 mb-3">総合診断</h3>
                    <p id="summary-text" class="text-gray-700 leading-relaxed">
                        <!-- 総合評価テキストが挿入されます -->
                    </p>
                </section>

                <section class="mb-8">
                    <h3 class="text-xl font-semibold text-gray-700 mb-3 border-b pb-2">AI診断フロー</h3>
                    <ol class="space-y-3 text-gray-600">
                        <li class="flex items-center">
                            <span class="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-green-500 text-white rounded-full mr-3 font-bold">1</span>
                            <span>**動画受信とタスク登録:** LINEで動画を受信後、即座にCloud Tasksに解析ジョブを登録します。（即時応答）</span>
                        </li>
                        <li class="flex items-center">
                            <span class="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-green-500 text-white rounded-full mr-3 font-bold">2</span>
                            <span>**MediaPipe解析:** Cloud RunのWorkerがジョブを実行し、動画から全フレームの骨格データ（関節位置、角度など）を抽出します。</span>
                        </li>
                        <li class="flex items-center">
                            <span class="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-green-500 text-white rounded-full mr-3 font-bold">3</span>
                            <span>**Gemini AI診断:** 抽出された数値データをGemini APIに送り、プロの基準と比較した詳細な診断レポート（Markdown）を生成します。</span>
                        </li>
                        <li class="flex items-center">
                            <span class="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-green-500 text-white rounded-full mr-3 font-bold">4</span>
                            <span>**レポート発行:** 診断結果をFirestoreに保存し、LINEでWebレポートURLをユーザーに返信します。（最終応答）</span>
                        </li>
                    </ol>
                </section>
                
                <section>
                    <h3 class="text-xl font-semibold text-gray-700 mb-3 border-b pb-2">このレポートの使い方</h3>
                    <ul class="list-disc ml-6 space-y-2 text-gray-600">
                        <li>左側のナビゲーションメニューから、**「03. AI総合評価」**以下の診断項目を順に確認してください。</li>
                        <li>**「01. 骨格計測データ」**で、あなたのスイングが客観的にどう数値化されたかを確認できます。</li>
                        <li>AIの提案を参考に、次回のスイング改善にお役立てください。</li>
                    </ul>
                </section>
            `;
            return page;
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

        async function fetchReport() {
            const urlParams = new URLSearchParams(window.location.search);
            const reportId = urlParams.get('id');

            if (!reportId) {
                displayFatalError("レポートIDが指定されていません。", "URLに`?id=<レポートID>`が必要です。");
                return;
            }
            
            document.getElementById('report-id').textContent = reportId;

            try {
                // Cloud Function / Cloud Run API (GET /report/<id>) を想定
                const response = await fetch(`/report/${reportId}`);
                
                if (!response.ok) {
                    if (response.status === 404) {
                        displayFatalError("指定されたレポートが見つかりません。", `レポートID: ${reportId}`);
                    } else if (response.status === 202) {
                        // 処理中のレスポンスコードを想定
                        displayProcessingMessage();
                        return;
                    } else {
                        throw new Error(`HTTP Error: ${response.status}`);
                    }
                    return;
                }

                const data = await response.json();

                if (data.status === 'PROCESSING') {
                    displayProcessingMessage();
                    return;
                }
                
                if (data.status !== 'COMPLETED') {
                    displayFatalError("レポート処理が失敗しました。", `ステータス: ${data.status}`);
                    return;
                }
                
                // データ抽出
                const aiReport = data.ai_report || "## 03. AI総合評価\nAIレポートの生成に失敗しました。";
                const rawData = data.raw_data || {};
                const summary = data.summary || "AIによる総合評価はまだ生成されていません。";
                const timestamp = new Date(data.timestamp.seconds * 1000).toLocaleString('ja-JP', {
                    year: 'numeric', month: '2-digit', day: '2-digit', 
                    hour: '2-digit', minute: '2-digit', second: '2-digit'
                });

                // レンダリング
                document.getElementById('timestamp').textContent = timestamp;
                document.getElementById('summary-text').textContent = summary;
                renderPages(aiReport, rawData);

            } catch (error) {
                console.error("Fetch error:", error);
                displayFatalError("レポートの取得中にネットワークまたはサーバーエラーが発生しました。", error.message);
            }
        }

        // ページロード時にレポート取得を開始
        window.onload = fetchReport;
    </script>
</body>
</html>"""

# ------------------------------------------------
# Firebase/Firestoreとの連携
# ------------------------------------------------

def save_report_to_firestore(user_id, report_id, report_data):
    """診断レポートをFirestoreに保存する"""
    if db is None:
        app.logger.error("Firestore client is not initialized.")
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
        app.logger.error("Firestore client is not initialized.")
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
# Cloud Tasksへジョブを投入する関数
# ------------------------------------------------

def create_cloud_task(report_id, video_url, user_id):
    """
    Cloud Tasksに動画解析タスクを作成し、Cloud Run Workerをトリガーする
    """
    if task_client is None:
        app.logger.error("Cloud Tasks client is not initialized.")
        return None

    # Cloud Run WorkerのエンドポイントURLを構築
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
                'service_account_email': TASK_SA_EMAIL, # ★★★修正点: 環境変数から取得
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

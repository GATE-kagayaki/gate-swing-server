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
        # Cloud Run環境ではApplicationDefault認証情報を使用
        cred = credentials.ApplicationDefault()
        initialize_app(cred, {'projectId': GCP_PROJECT_ID})
    db = firestore.client()
except Exception as e:
    # 認証情報の設定エラーまたはFirestore初期化エラーをログに出力
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
                    <p class="font-bold">🚨 レポート表示エラー (データ取得失敗)</p>
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
                
                // ★★★ 修正箇所 1: タイムスタンプの安全な処理 (try/catchを追加) ★★★
                let timestamp = 'N/A';
                try {
                    if (data.timestamp && data.timestamp._seconds) {
                        // Firestore Timestamp Object
                        timestamp = new Date(data.timestamp._seconds * 1000).toLocaleString('ja-JP');
                    } else if (data.timestamp) {
                        // Attempt to parse as a standard string/number
                        timestamp = new Date(data.timestamp).toLocaleString('ja-JP');
                    }
                } catch (e) {
                    console.error("Timestamp parsing failed:", e);
                    timestamp = 'データ処理エラー';
                }
                document.getElementById('timestamp').textContent = timestamp;


                document.getElementById('frames').textContent = raw.frame_count || 'N/A';
                document.getElementById('shoulder').textContent = (raw.max_shoulder_rotation ? raw.max_shoulder_rotation.toFixed(1) + '°' : 'N/A');
                document.getElementById('hip').textContent = (raw.min_hip_rotation ? raw.min_hip_rotation.toFixed(1) + '°' : 'N/A');
                document.getElementById('cock').textContent = (raw.max_wrist_cock ? raw.max_wrist_cock.toFixed(1) + '°' : 'N/A');

                // Markdownのレンダリング (簡易的な表示)
                const markdownText = data.ai_report_text || data.ai_report_text_free || "AI診断データが利用できません。";
                
                // ★★★ 修正箇所 2: Markdown処理の安定化 (decodeURIComponent削除) ★★★
                try {
                    // 以前のdecodeURIComponentを削除し、純粋なsplit/joinで改行コードに対応
                    const processedText = markdownText.split('\\n').join('<br>').split('\n').join('<br>');

                    document.getElementById('ai-report-markdown').innerHTML = processedText;
                    console.log("Markdown processing successful.");

                } catch (e) {
                    // Markdownの処理に失敗した場合、生のテキストを表示し、エラーを出力
                    console.error("Markdown processing failed:", e);
                    document.getElementById('ai-report-markdown').innerHTML = 
                        `<p class="text-red-500 font-bold">【レポート表示失敗】テキスト処理エラー: ${e.message}</p>
                         <p class="text-sm mt-1">Raw Data: ${markdownText}</p>`;
                }
                
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
"""

# ------------------------------------------------
# 解析ロジック (analyze_swing) - 必須計測項目を全て実装
# ------------------------------------------------
def analyze_swing(video_path):
    """
    動画を解析し、スイングの評価レポート（テキスト）を返す。
    この関数は、process_video_async内から呼び出されます。
    """
    # ★★★ 重いライブラリをここでインポートする (関数内インポート) ★★★
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
        return

    # 1.5 動画の自動圧縮とリサイズ処理 (メモリ不足回避のため必須)
    try:
        compressed_video_path = tempfile.NamedTemporaryFile(suffix="_compressed.mp4", delete=False).name
        # 処理遅延の原因となるFFmpeg処理の安定化
        FFMPEG_PATH = '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else 'ffmpeg'
        
        # 圧縮とリサイズを実行
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
        report_text = f"【動画処理エラー】動画圧縮で問題が発生しました: {str(e)[:100]}..."
        line_bot_api.push_message(user_id, TextSendMessage(text=report_text))
        return
        
    # 2. 動画の解析を実行
    try:
        analysis_data = analyze_swing(video_to_analyze)
        
        # ★★★ AI診断の実行 - サービスロジックの中心 ★★★
        is_premium = False # ダミーロジック: 決済ロジックが未実装のため、常にFalse
        
        if GEMINI_API_KEY:
            ai_report_text = generate_full_member_advice(analysis_data, genai, types) 
        else:
            # 無料会員向け: AIを使わず、MediaPipeデータに基づいた「課題提起」を生成
            ai_report_text = generate_free_member_summary(analysis_data)
            
        # 3. Firestoreに解析結果を保存 (Webレポートの基盤)
        if db:
            report_data = {
                "timestamp": firestore.SERVER_TIMESTAMP,
                "user_id": user_id,
                "is_premium": is_premium,
                "mediapipe_data": analysis_data,
                "ai_report_text": ai_report_text
            }
            # コレクション 'reports' にデータを追加
            _, doc_ref = db.collection('reports').add(report_data)
            report_id = doc_ref.id
            
            # WebレポートのURLを生成 (正しいホストURLを使用)
            service_url = SERVICE_HOST_URL.rstrip('/')
            report_url = f"{service_url}/report?id={report_id}"
            
        else:
             # DB接続失敗時は、テキストレポートを直接送る
             report_url = None
             
    except Exception as e:
        report_text = f"【解析エラー】動画解析中に致命的なエラーが発生しました: {e}"
        line_bot_api.push_message(user_id, TextSendMessage(text=f"【システムエラー】動画の解析中に問題が発生しました。エラーログ: {str(e)}"))
        app.logger.error(f"解析中の致命的なエラー: {e}", exc_info=True)
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
            # DB接続失敗時は、テキストレポートを直接送る
            line_bot_api.push_message(user_id, TextSendMessage(text=ai_report_text))

    except Exception as e:
        app.logger.error(f"レポート送信中に予期せぬエラーが発生しました: {e}", exc_info=True)

    # 5. 一時ファイルを削除
    if original_video_path and os.path.exists(original_video_path):
        os.remove(original_video_path)
    if compressed_video_path and os.path.exists(compressed_video_path):
        os.remove(compressed_video_path)

# ------------------------------------------------
# ★★★ Gemini API 呼び出し関数 (全項目網羅版) ★★★
# ------------------------------------------------
def generate_full_member_advice(analysis_data, genai, types): # genai, typesを引数で受け取る
    """MediaPipeの数値結果をGemini APIに渡し、理想の10項目を網羅した詳細レポートを生成させる"""
    
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
# ★★★ 無料会員向け「課題提起」生成関数 (AI不使用) ★★★
# ------------------------------------------------
def generate_free_member_summary(analysis_data):
    """AIを使わず、計測値からロジックで無料会員向けレポートを生成する"""
    
    shoulder_rot = analysis_data['max_shoulder_rotation']
    hip_rot = analysis_data['min_hip_rotation']
    head_drift = analysis_data['max_head_drift_x']
    wrist_cock = analysis_data['max_wrist_cock']
    
    issues = []

    # 課題提起ロジック (数値を基に問題を特定)
    # 課題1: 頭の移動が大きい (0.03以上)
    if head_drift > 0.03:
        issues.append("頭の水平方向への移動が大きい (軸の不安定さ)")
    # 課題2: コックが早くほどける (160度以上)
    if wrist_cock > 160:
        issues.append("手首のコックが早くほどける傾向があります (アーリーリリース)")
    # 課題3: 上半身の回転不足と腰の開きすぎ (40度以下 and 10度以上)
    if shoulder_rot < 40 and hip_rot > 10:
        issues.append("上半身の回転不足と腰の開きすぎの連鎖が確認されます")

    # 課題リストの整形 (黒丸リストに修正)
    if not issues:
        issue_text = "特に目立った問題は検出されませんでした。"
    else:
        issue_text = "あなたのスイングには、以下の改善点が見られます。\n"
        for issue in issues:
            issue_text += f"・ {issue}\n" # 黒丸「・」で箇条書き
    
    # 最終レポート構成
    report = (
        f"あなたのスイングをAIによる骨格分析に基づき診断しました。\n\n"
        f"**【お客様の改善点（簡易診断）】**\n"
        f"{issue_text}\n\n"
        f"**【お客様へのメッセージ】**\n"
        f"有料版をご利用いただくと、これらの問題の**さらに詳しい分析による改善点の抽出**、具体的な練習ドリル、最適なクラブフィッティング提案をご利用いただけます。お客様のゴルフライフが充実したものになることを応援しております。" 
    )
        
    return report

# ------------------------------------------------
# LINE Webhookのメイン処理 (重複解消済みの最終版)
# ------------------------------------------------
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        # LINE Bot SDKのハンドラーに処理を委譲
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
    """WebレポートのフロントエンドにJSONデータを返すAPIエンドポイント (重複解消済み)"""
    # ログを強化し、リクエストがこの関数に到達しているかを確認
    app.logger.info(f"Report API accessed. Query: {request.query_string.decode('utf-8')}")
    
    if not db:
        # DB接続が初期化されていない場合のエラー応答
        app.logger.error("Firestore DB connection is not initialized.")
        return jsonify({"error": "データベースが初期化されていません。サーバーログを確認してください。"}), 500
        
    report_id = request.args.get('id')
    if not report_id:
        app.logger.warning("Report ID is missing from query.")
        return jsonify({"error": "レポートIDが指定されていません。"}), 400
    
    try:
        # Firestoreからドキュメントを取得
        doc = db.collection('reports').document(report_id).get()
        if not doc.exists:
            app.logger.warning(f"Report document not found: {report_id}")
            return jsonify({"error": "指定されたレポートは見つかりませんでした。"}), 404
        
        data = doc.to_dict()
        app.logger.info(f"Successfully retrieved data for report: {report_id}")
        
        # クライアントへの応答として、必要なデータのみをJSON形式で返す
        response_data = {
            # FirestoreのTimestampオブジェクトはJSONシリアライズできないため、そのまま返す
            "timestamp": data.get('timestamp', {}), 
            "mediapipe_data": data.get('mediapipe_data', {}),
            "ai_report_text": data.get('ai_report_text', 'AIレポートがありません。')
        }
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"レポート表示APIエラー: {e}", exc_info=True)
        return jsonify({"error": f"レポートデータの取得中に予期せぬエラーが発生しました: {e}"}), 500


@app.route('/report', methods=['GET'])
def get_report_page():
    """WebレポートのHTMLテンプレートを返す (重複解消済み)"""
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

    # 1. ユーザーへの即時応答（LINEの応答タイムアウト回避）
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="動画を受け付けました。解析を開始します。しばらくお待ちください...")
    )
    
    # 2. 動画コンテンツの取得
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
    app.run(host='0.0.0.0', port=port)

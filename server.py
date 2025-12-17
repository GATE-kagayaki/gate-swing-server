import os
import json
import math
import shutil
import traceback
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from flask import Flask, request, jsonify, abort, render_template

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, VideoMessage, TextSendMessage

from google.cloud import firestore
from google.cloud import tasks_v2
from google.api_core.exceptions import NotFound, PermissionDenied


# ==================================================
# CONFIG
# ==================================================
app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")

PROJECT_ID = (
    os.environ.get("PROJECT_ID")
    or os.environ.get("GCP_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GCP_PROJECT")
    or ""
)

QUEUE_NAME = os.environ.get("TASK_QUEUE_NAME", "video-analysis-queue")
QUEUE_LOCATION = os.environ.get("TASK_QUEUE_LOCATION", "asia-northeast2")

SERVICE_HOST_URL = os.environ.get("SERVICE_HOST_URL", "").rstrip("/")
TASK_SA_EMAIL = os.environ.get("TASK_SA_EMAIL", "")

TASK_HANDLER_PATH = "/task-handler"
TASK_HANDLER_URL = f"{SERVICE_HOST_URL}{TASK_HANDLER_PATH}"

db = firestore.Client()

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

tasks_client = tasks_v2.CloudTasksClient()


# ==================================================
# Helpers
# ==================================================
def firestore_safe_set(report_id: str, data: Dict[str, Any]) -> None:
    try:
        db.collection("reports").document(report_id).set(data, merge=True)
    except Exception:
        print(traceback.format_exc())


def firestore_safe_update(report_id: str, patch: Dict[str, Any]) -> None:
    try:
        db.collection("reports").document(report_id).update(patch)
    except Exception:
        print(traceback.format_exc())


def safe_line_reply(reply_token: str, text: str) -> None:
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError:
        print(traceback.format_exc())


def safe_line_push(user_id: str, text: str) -> None:
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text))
    except LineBotApiError:
        print(traceback.format_exc())


def make_initial_reply(report_id: str) -> str:
    url = f"{SERVICE_HOST_URL}/report/{report_id}"
    return (
        "✅ 動画を受信しました。\n"
        "AIによるスイング解析を開始します。\n\n"
        "解析完了まで、1〜3分ほどお待ちください。\n"
        "完了次第、結果をお知らせします。\n\n"
        "【進行状況の確認】\n"
        "以下のURLから、解析の進行状況や\n"
        "レポートの準備状況を確認できます。\n"
        f"{url}\n\n"
        "【料金プラン（プロ評価付きフルレポート）】\n"
        "① 都度会員　500円／1回\n"
        "② 回数券　1,980円／5回\n"
        "③ 月額会員　4,980円／月\n\n"
        "より詳しい分析をご希望の方は、ぜひフルレポートをご活用ください。"
    )


def make_done_push(report_id: str) -> str:
    url = f"{SERVICE_HOST_URL}/report/{report_id}"
    return (
        "🎉 スイング計測が完了しました！\n\n"
        "以下のリンクから診断レポートを確認できます。\n\n"
        f"{url}"
    )


# ==================================================
# Premium判定（本番は決済と連携でOK）
# ==================================================
def is_premium_user(user_id: str) -> bool:
    # ここはStripe連携後に置き換え
    # いまは「有料版テスト」を優先するなら True にしてください
    return True


# ==================================================
# Cloud Tasks
# ==================================================
def create_cloud_task(report_id: str, user_id: str, message_id: str) -> str:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is empty. Set PROJECT_ID or GCP_PROJECT_ID.")
    if not SERVICE_HOST_URL:
        raise RuntimeError("SERVICE_HOST_URL is empty.")
    if not TASK_SA_EMAIL:
        raise RuntimeError("TASK_SA_EMAIL is empty.")

    queue_path = tasks_client.queue_path(PROJECT_ID, QUEUE_LOCATION, QUEUE_NAME)

    payload = json.dumps(
        {"report_id": report_id, "user_id": user_id, "message_id": message_id},
        ensure_ascii=False,
    ).encode("utf-8")

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": TASK_HANDLER_URL,
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            "oidc_token": {
                "service_account_email": TASK_SA_EMAIL,
                "audience": SERVICE_HOST_URL,
            },
        }
    }

    resp = tasks_client.create_task(parent=queue_path, task=task)
    return resp.name


# ==================================================
# MediaPipe analysis
# ==================================================
def analyze_swing_with_mediapipe(video_path: str) -> Dict[str, Any]:
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCVがビデオを読み込めませんでした。")

    frame_count = 0
    max_shoulder = 0.0
    min_hip = 999.0
    max_wrist = 0.0
    max_head = 0.0
    max_knee = 0.0

    def angle(p1, p2, p3):
        ax, ay = p1[0] - p2[0], p1[1] - p2[1]
        bx, by = p3[0] - p2[0], p3[1] - p2[1]
        dot = ax * bx + ay * by
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na * nb == 0:
            return 0.0
        c = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(c))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            def xy(i): return (lm[i].x, lm[i].y)

            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            LI = mp_pose.PoseLandmark.LEFT_INDEX.value
            NO = mp_pose.PoseLandmark.NOSE.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value

            max_shoulder = max(max_shoulder, angle(xy(LS), xy(RS), xy(RH)))
            min_hip = min(min_hip, angle(xy(LH), xy(RH), xy(LK)))
            max_wrist = max(max_wrist, angle(xy(LE), xy(LW), xy(LI)))
            max_head = max(max_head, abs(xy(NO)[0] - 0.5))
            max_knee = max(max_knee, abs(xy(LK)[0] - 0.5))

    cap.release()

    if frame_count < 10:
        raise RuntimeError("解析に必要なフレーム数が不足しています。")

    return {
        "frame_count": int(frame_count),
        "max_shoulder_rotation": round(float(max_shoulder), 2),
        "min_hip_rotation": round(float(min_hip), 2),
        "max_wrist_cock": round(float(max_wrist), 2),
        "max_head_drift": round(float(max_head), 4),
        "max_knee_sway": round(float(max_knee), 4),
    }


# ==================================================
# 3×3×3（27パターン）分類ユーティリティ
# ==================================================
def cat3_by_range(value: float, lo: float, hi: float) -> str:
    """low / mid / high"""
    if value < lo:
        return "low"
    if value > hi:
        return "high"
    return "mid"


def cat3_sway(value: float, lo: float, hi: float) -> str:
    """swayは小さいほど良いので low=良, mid=普通, high=悪 の扱い"""
    if value < lo:
        return "good"
    if value > hi:
        return "bad"
    return "mid"


def pick_2to6_bullets(section: str, main: str, head: str, knee: str) -> Tuple[List[str], List[str]]:
    """
    02-06 用： (main × head × knee) の 3×3×3=27パターン
    main: low/mid/high
    head: good/mid/bad
    knee: good/mid/bad
    """
    good: List[str] = []
    bad: List[str] = []

    # --- main評価（セクション別） ---
    if section == "02":  # shoulder
        if main == "low":
            bad.append("肩回転量がやや少なく、上半身の捻転によるエネルギーが出にくい可能性があります。")
        elif main == "mid":
            good.append("肩回転量は目安レンジ内で、上半身の回旋は安定しています。")
        else:
            good.append("肩回転量は大きく、パワーを出せるポテンシャルがあります。")
            bad.append("回し過ぎになると、軸ブレやタイミングのズレにつながる可能性があります。")

    if section == "03":  # hip
        if main == "low":
            bad.append("腰回転が浅くなりやすく、下半身からの推進力が活かし切れない可能性があります。")
        elif main == "mid":
            good.append("腰回転は目安レンジ内で、土台の回旋は安定しています。")
        else:
            good.append("腰回転が大きく、下半身主導の動きが作れています。")
            bad.append("回転が強すぎると、上体がつられて開きやすくなる可能性があります。")

    if section == "04":  # wrist
        if main == "low":
            bad.append("コック量が少なく、タメが作りにくい可能性があります。")
        elif main == "mid":
            good.append("コック量は目安レンジ内で、再現性の高いリリースが期待できます。")
        else:
            good.append("コック量が大きく、ヘッドスピードを作りやすい形です。")
            bad.append("コックが大きすぎると手首主導になり、タイミングがズレやすい可能性があります。")

    if section == "05":  # head sway (main=good/mid/bad を流用)
        if main == "good":
            good.append("頭部の左右ブレが小さく、スイング軸の安定性が高い状態です。")
        elif main == "mid":
            good.append("頭部ブレは平均的で、大きく崩れる動きは見られません。")
            bad.append("局所的にブレが増える場面があると、ミート率が落ちやすくなります。")
        else:
            bad.append("頭部の左右ブレが大きく、再現性が落ちやすい傾向があります。")

    if section == "06":  # knee sway
        if main == "good":
            good.append("膝の左右ブレが小さく、下半身の安定性が高い状態です。")
        elif main == "mid":
            good.append("膝ブレは平均的で、土台は大きく崩れていません。")
            bad.append("踏み替えのタイミングで左右差が出ると、軸がズレやすくなります。")
        else:
            bad.append("膝の左右ブレが大きく、体重移動が横流れになりやすい可能性があります。")

    # --- head/knee補正（27パターン化の核） ---
    # headがbadなら、どのセクションでも「再現性」観点の悪い点を足す
    if head == "bad":
        bad.append("頭部の安定性が低い場面があるため、インパクトの再現性が落ちやすくなります。")
    elif head == "good":
        good.append("頭部が安定しているため、動作の再現性を作りやすい状態です。")

    # kneeがbadなら、どのセクションでも「土台の安定」観点の悪い点を足す
    if knee == "bad":
        bad.append("下半身が流れやすい場面があるため、上体の動きも乱れやすくなります。")
    elif knee == "good":
        good.append("下半身が安定しているため、回転動作の土台がしっかりしています。")

    # 箇条書き最大3に丸める
    good = good[:3]
    bad = bad[:3]

    # 片側が0のときの保険（読みにくさ回避）
    if not good:
        good = ["大きな崩れは見られず、改善を積み上げやすい状態です。"]
    if not bad:
        bad = ["現状の動きは安定しており、再現性を維持しやすい状態です。"]

    return good, bad


# ==================================================
# Analysis JSON（最終構造）
# ==================================================
def build_section_01(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "01. 骨格計測データ（AIが測定）",
        "items": [
            {
                "name": "解析フレーム数",
                "value": raw["frame_count"],
                "description": "動画から解析できたフレーム数です。数が多いほど、動作全体を安定して解析できています。",
                "guide": "150〜300 フレーム",
            },
            {
                "name": "最大肩回転角（°）",
                "value": raw["max_shoulder_rotation"],
                "description": "スイング中に肩がどれだけ回転したかを示す角度です。上半身の回旋量の指標になります。",
                "guide": "80〜110°",
            },
            {
                "name": "最小腰回転角（°）",
                "value": raw["min_hip_rotation"],
                "description": "スイング中に腰が最も回転した瞬間の角度です。下半身の回旋量を表します。",
                "guide": "35〜45°",
            },
            {
                "name": "最大手首コック角（°）",
                "value": raw["max_wrist_cock"],
                "description": "スイング中に手首が最も折れた角度です。クラブのコック量の指標になります。",
                "guide": "120〜150°",
            },
            {
                "name": "最大頭部ブレ（Sway）",
                "value": raw["max_head_drift"],
                "description": "スイング中に頭の位置が左右にどれだけ動いたかを示します。スイング軸の安定性を表します。",
                "guide": "0.05〜0.15",
            },
            {
                "name": "最大膝ブレ（Sway）",
                "value": raw["max_knee_sway"],
                "description": "スイング中に膝が左右にどれだけ動いたかを示します。下半身の安定性の指標です。",
                "guide": "0.05〜0.20",
            },
        ],
    }


def build_free_07(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 無料版：プロ評価なし（導線あり）
    return {
        "title": "07. 総合評価",
        "text": [
            "本レポートでは、スイング全体の傾向を骨格データに基づいて評価しています。",
            "回転量と安定性のバランスを整えることで、再現性の向上が期待できます。",
            "",
            "より詳しい分析をご希望の方へ",
            "ご自身のスイングを深く理解したい方は、ぜひフルレポートをご活用ください。",
        ],
    }


def build_paid_02_to_06(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 3×3×3 のためのカテゴリ
    shoulder_cat = cat3_by_range(raw["max_shoulder_rotation"], 80, 110)
    hip_cat = cat3_by_range(raw["min_hip_rotation"], 35, 45)
    wrist_cat = cat3_by_range(raw["max_wrist_cock"], 120, 150)

    head_cat = cat3_sway(raw["max_head_drift"], 0.05, 0.15)  # good/mid/bad
    knee_cat = cat3_sway(raw["max_knee_sway"], 0.05, 0.20)  # good/mid/bad

    # 02
    g2, b2 = pick_2to6_bullets("02", shoulder_cat, head_cat, knee_cat)
    # 03
    g3, b3 = pick_2to6_bullets("03", hip_cat, head_cat, knee_cat)
    # 04
    g4, b4 = pick_2to6_bullets("04", wrist_cat, head_cat, knee_cat)
    # 05（mainを head_cat として使う）
    g5, b5 = pick_2to6_bullets("05", head_cat, head_cat, knee_cat)
    # 06（mainを knee_cat として使う）
    g6, b6 = pick_2to6_bullets("06", knee_cat, head_cat, knee_cat)

    return {
        "02": {
            "title": "02. Shoulder Rotation（肩回転）",
            "value": raw["max_shoulder_rotation"],
            "good": g2,
            "bad": b2,
            "pro_comment": "回転量は“出す”より“揃える”ことで、結果が安定しやすくなります。",
        },
        "03": {
            "title": "03. Hip Rotation（腰回転）",
            "value": raw["min_hip_rotation"],
            "good": g3,
            "bad": b3,
            "pro_comment": "腰の安定は強みです。上体との同調が取れると一段良くなります。",
        },
        "04": {
            "title": "04. Wrist Cock（コック角）",
            "value": raw["max_wrist_cock"],
            "good": g4,
            "bad": b4,
            "pro_comment": "コックは“作る”より“自然に入る”形が安定しやすいです。",
        },
        "05": {
            "title": "05. Head Stability（頭部ブレ）",
            "value": raw["max_head_drift"],
            "good": g5,
            "bad": b5,
            "pro_comment": "頭の位置が整うと、ミート率と方向性は一気に安定します。",
        },
        "06": {
            "title": "06. Knee Stability（膝ブレ）",
            "value": raw["max_knee_sway"],
            "good": g6,
            "bad": b6,
            "pro_comment": "膝の安定は“軸”そのもの。ここが揃うとブレが減ります。",
        },
    }


def build_paid_07(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 02-06の傾向から“個別感”を出す（同じにならないように）
    issues = []
    if raw["max_head_drift"] > 0.15:
        issues.append("頭部ブレ")
    if raw["max_knee_sway"] > 0.20:
        issues.append("膝ブレ")
    if raw["max_wrist_cock"] > 150:
        issues.append("手首主導（コック過多）")
    if raw["max_shoulder_rotation"] > 110:
        issues.append("肩回転の回し過ぎ")
    if raw["min_hip_rotation"] < 35:
        issues.append("腰回転の浅さ")

    if not issues:
        issues_txt = "大きな崩れは見られず、安定した土台が整っています。"
    else:
        issues_txt = "主な改善テーマは「" + "／".join(issues[:3]) + "」です。"

    return {
        "title": "07. 総合評価（プロ評価）",
        "text": [
            "回転量の土台はできており、ポテンシャルは十分にあります。",
            issues_txt,
            "今回の結果では「安定性」と「タイミング（手首主導の抑制）」を優先すると、再現性が上がりやすいです。",
        ],
    }


def build_paid_08(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 数値連動：最大3つ
    drills = []

    shoulder_cat = cat3_by_range(raw["max_shoulder_rotation"], 80, 110)
    head_cat = cat3_sway(raw["max_head_drift"], 0.05, 0.15)
    knee_cat = cat3_sway(raw["max_knee_sway"], 0.05, 0.20)
    wrist_cat = cat3_by_range(raw["max_wrist_cock"], 120, 150)

    # 1) 同調（肩がhigh or shoulderと安定性課題）
    if shoulder_cat == "high" or head_cat == "bad" or knee_cat == "bad":
        drills.append({
            "name": "上半身と下半身の同調ドリル（クロスアームターン）",
            "purpose": "上半身だけが先行する動きを抑え、体全体で回す感覚を作る",
            "how": "①胸の前で腕を軽く組む\n②下半身を固定して胸と腰を同時に回す\n③左右交互に10回×2セット",
        })

    # 2) 手首主導抑制（コック high）
    if wrist_cat == "high":
        drills.append({
            "name": "L to L スイング",
            "purpose": "コック過多を抑え、体の回転でクラブを動かす",
            "how": "①腰〜腰の小さい振り幅\n②手先で合わせず回転で振る\n③一定リズムで20回",
        })

    # 3) 軸安定（head/knee bad）
    if head_cat == "bad" or knee_cat == "bad":
        drills.append({
            "name": "壁チェック（軸安定）",
            "purpose": "頭部・下半身の左右ブレを抑える",
            "how": "①壁の前でアドレス\n②頭と壁の距離を一定に保つ\n③膝の横流れも同時に確認",
        })

    drills = drills[:3]
    if not drills:
        drills = [{
            "name": "テンポドリル（メトロノーム）",
            "purpose": "再現性を上げるためにタイミングを一定にする",
            "how": "①ゆっくり素振り\n②同じテンポで10回\n③その後ボールを10球",
        }]

    return {"title": "08. Training Drills（練習ドリル）", "drills": drills}


def build_paid_09(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 入力（HS/ミス傾向）がない前提 → 断定しないがレンジを削って“答え”にする
    head_bad = raw["max_head_drift"] > 0.15
    knee_bad = raw["max_knee_sway"] > 0.20
    wrist_high = raw["max_wrist_cock"] > 150

    rows = []

    # 重量
    if head_bad or knee_bad or wrist_high:
        weight = "50g台後半〜60g台前半"
        reason = "軽すぎるとタイミングが合いにくく、再現性が落ちやすいため"
    else:
        weight = "50g台前半〜60g台前半"
        reason = "振り切りやすさと安定性のバランスが取りやすいため"
    rows.append({"item": "重量", "guide": weight, "reason": reason})

    # キックポイント
    if wrist_high:
        kp = "中調子〜中元調子"
        reason = "先調子寄りだと挙動が大きくなりやすいため"
    else:
        kp = "中調子"
        reason = "タイミングと再現性を取りやすいため"
    rows.append({"item": "キックポイント", "guide": kp, "reason": reason})

    # フレックス（HS未入力なので幅を残す）
    flex = "R〜SR〜S"
    rows.append({"item": "フレックス", "guide": flex, "reason": "柔らかすぎるとタイミングが合いにくくなるため"})

    # トルク
    if wrist_high or head_bad:
        tq = "3.5〜4.5"
        reason = "手元の暴れを抑えて方向性を安定させやすいため"
    else:
        tq = "4.0〜5.0"
        reason = "適度なしなり感で振りやすさを確保しやすいため"
    rows.append({"item": "トルク", "guide": tq, "reason": reason})

    return {
        "title": "09. Shaft Fitting Guide（推奨）",
        "table": rows,
        "note": "本結果はあくまでも指標です。ご購入の際は試打を行った上でご検討ください。",
    }


def build_paid_10(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "10. Summary（まとめ）",
        "text": [
            "今回の解析では、回転量を活かせるポテンシャルが確認できました。",
            "次のステップは「安定性」と「タイミング」を揃えることです。",
            "練習ドリルとフィッティング指針を参考に、段階的に改善を進めていきましょう。",
            "",
            "あなたのゴルフライフが、より充実したものになることを切に願っています。",
        ],
    }


def build_analysis(raw: Dict[str, Any], premium: bool) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {"01": build_section_01(raw)}

    if not premium:
        analysis["07"] = build_free_07(raw)
        return analysis

    analysis.update(build_paid_02_to_06(raw))
    analysis["07"] = build_paid_07(raw)
    analysis["08"] = build_paid_08(raw)
    analysis["09"] = build_paid_09(raw)
    analysis["10"] = build_paid_10(raw)
    return analysis


# ==================================================
# Routes
# ==================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "project_id": PROJECT_ID,
        "queue_location": QUEUE_LOCATION,
        "queue_name": QUEUE_NAME,
        "service_host_url": SERVICE_HOST_URL,
        "task_handler_url": TASK_HANDLER_URL,
        "task_sa_email_set": bool(TASK_SA_EMAIL),
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event: MessageEvent):
    user_id = event.source.user_id
    msg = event.message
    report_id = f"{user_id}_{msg.id}"

    premium = is_premium_user(user_id)

    firestore_safe_set(
        report_id,
        {
            "user_id": user_id,
            "status": "PROCESSING",
            "is_premium": premium,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        task_name = create_cloud_task(report_id, user_id, msg.id)
        firestore_safe_update(report_id, {"task_name": task_name})
        safe_line_reply(event.reply_token, make_initial_reply(report_id))
    except (NotFound, PermissionDenied) as e:
        firestore_safe_update(report_id, {"status": "TASK_FAILED", "error": str(e)})
        safe_line_reply(event.reply_token, "システムエラーが発生しました。時間を置いて再度お試しください。")
    except Exception as e:
        firestore_safe_update(report_id, {"status": "TASK_FAILED", "error": str(e)})
        print("Failed to create task:", traceback.format_exc())
        safe_line_reply(event.reply_token, "システムエラーが発生しました。時間を置いて再度お試しください。")


@app.route("/task-handler", methods=["POST"])
def task_handler():
    d = request.get_json(silent=True) or {}
    report_id = d.get("report_id")
    message_id = d.get("message_id")
    user_id = d.get("user_id")

    if not report_id or not message_id or not user_id:
        return "Invalid payload", 400

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, f"{message_id}.mp4")

    doc_ref = db.collection("reports").document(report_id)

    try:
        doc_ref.update({"status": "IN_PROGRESS"})

        # download
        content = line_bot_api.get_message_content(message_id)
        with open(video_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        # analyze
        raw_data = analyze_swing_with_mediapipe(video_path)

        # build report
        doc = doc_ref.get()
        premium = bool((doc.to_dict() or {}).get("is_premium", False))
        analysis = build_analysis(raw_data, premium)

        doc_ref.update({
            "status": "COMPLETED",
            "raw_data": raw_data,
            "analysis": analysis,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

        safe_line_push(user_id, make_done_push(report_id))
        return jsonify({"ok": True}), 200

    except Exception as e:
        print(traceback.format_exc())
        doc_ref.update({"status": "FAILED", "error": str(e)})
        safe_line_push(user_id, "システムエラーが発生し、解析を完了できませんでした。")
        return "Internal Error", 500

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/report/<report_id>")
def report_page(report_id):
    # ★Jinjaでanalysisを参照しない（UndefinedError対策）
    return render_template("report.html", report_id=report_id)


@app.route("/api/report_data/<report_id>")
def api_report_data(report_id):
    doc = db.collection("reports").document(report_id).get()
    if not doc.exists:
        return jsonify({"error": "not found"}), 404
    d = doc.to_dict() or {}
    return jsonify({
        "status": d.get("status"),
        "analysis": d.get("analysis", {}),
        "raw_data": d.get("raw_data", {}),
        "is_premium": d.get("is_premium", False),
        "error": d.get("error"),
        "created_at": d.get("created_at"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)


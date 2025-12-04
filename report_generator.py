# report_generator.py

import os
import textwrap
import openai

# OpenAI APIキー（必須）
openai.api_key = os.getenv("OPENAI_API_KEY")

# ================================
# 1. スイングの「ざっくり分析」(Aレベル)
# ================================
def analyze_swing_stub(club_type: str = "ドライバー", user_level: str = "初心者") -> dict:
    """
    今はまだ本物の映像解析は入れず、
    クラブ種別とレベルから「ありがちな傾向」を返すダミー分析。
    将来ここを、本物のOpenCV/MediaPipe解析に差し替える前提。
    """

    # 超ざっくりな傾向をパターン分け
    if club_type == "ドライバー":
        base = {
            "head": "テークバックで頭が右に流れやすく、切り返しで左に戻る動きが強め。",
            "shoulder": "トップまではしっかり回るが、ダウンで肩の戻りが早くなりやすい。",
            "hip": "腰の回転が肩と同時になりがちで、下半身リードが弱め。",
            "wrist": "切り返しでコックがほどけやすく、アーリーリリース傾向。",
            "path": "クラブはややアウトサイドから入りやすい。",
        }
    else:
        base = {
            "head": "頭の上下動は少なめだが、左右にはややブレやすい傾向。",
            "shoulder": "肩は十分に回るものの、ダウンスイングで開きが早め。",
            "hip": "下半身の回転量が少なく、体全体で振りにいきやすい。",
            "wrist": "インパクト前に手首の角度がほどけやすく、当てにいく動きになりがち。",
            "path": "ややカット軌道気味で、左への引っかけやスライスが出やすい。",
        }

    if user_level == "超初心者":
        level_note = "まだボールにしっかり当てることが最優先の段階です。"
    elif user_level == "中級":
        level_note = "スコア安定のために、再現性とインパクトゾーンの質を高めていきたい段階です。"
    else:
        level_note = "基礎はある程度できているので、『ミスのパターン』を減らせると一気に伸びます。"

    return {
        "club_type": club_type,
        "user_level": user_level,
        "level_note": level_note,
        **base,
    }

# ================================
# 2. 無料レポート（要約版：セクション07 相当）
# ================================
def generate_free_report_text(analysis: dict) -> str:
    """
    無料版：07.Key Diagnosis 相当の「総合診断コメントだけ」を返す。
    短く・分かりやすく・前向きに。
    """

    prompt = f"""
    あなたは日本語のゴルフコーチAIです。
    次のスイング分析情報をもとに、
    「総合診断コメント」と「改善の方向性」を
    初心者〜中級者向けに、やさしく・前向きに・具体的に説明してください。

    - クラブ: {analysis['club_type']}
    - レベル感: {analysis['user_level']}（{analysis['level_note']}）
    - 頭の動き: {analysis['head']}
    - 肩の動き: {analysis['shoulder']}
    - 腰の動き: {analysis['hip']}
    - 手首の使い方: {analysis['wrist']}
    - スイング軌道: {analysis['path']}

    出力フォーマット（日本語）:

    07. 総合診断

    ● 今のスイングの特徴（2〜3行）
    ● その結果出やすい球筋やミス（1〜2行）
    ● まず意識したいポイント（箇条書きで2つ）

    ・専門用語は使いすぎない
    ・「ダメ出し」ではなく、「こうするともっと良くなる」という言い方
    ・全体で300〜400文字くらい
    """

    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": textwrap.dedent(prompt)}],
        temperature=0.7,
    )

    return res["choices"][0]["message"]["content"].strip()

# ================================
# 3. 有料レポート（01〜10フル）
# ================================
def generate_paid_report_text(analysis: dict) -> str:
    """
    有料版：01〜10までフル構成のレポートを生成。
    すでにあなたと合意したテンプレ構成に沿って出力。
    """

    prompt = f"""
    あなたは日本語のゴルフスイング解析コーチAIです。
    次のスイング分析情報をもとに、
    「01〜10の構成」を持つ詳細レポートを作成してください。

    ▼スイング情報（ざっくり・Aレベル）
    - クラブ: {analysis['club_type']}
    - レベル感: {analysis['user_level']}（{analysis['level_note']}）
    - 頭の動き: {analysis['head']}
    - 肩の動き: {analysis['shoulder']}
    - 腰の動き: {analysis['hip']}
    - 手首の使い方: {analysis['wrist']}
    - スイング軌道: {analysis['path']}

    ▼レポート構成（タイトルは必ずこのままの日本語で）

    01. 概要
    02. 頭の安定性
    03. 肩の回転
    04. 腰の回転
    05. 手首の使い方
    06. スイング軌道
    07. 総合診断
    08. 改善ドリル
    09. フィッティングの方向性
    10. まとめ

    ▼セクションごとの役割

    01. 概要
      - このレポートで分かることを2〜3行で説明

    02〜06. 各要素の分析
      - Findings: 現状の特徴（箇条書き2〜4個）
      - Interpretation: それがボールにどう影響しているか（2〜3行）

    07. 総合診断
      - 一言でいうとどんなタイプか
      - 良いところ / もったいないところ

    08. 改善ドリル
      - 具体的な練習メニューを2〜3個
      - それぞれについて:
        ・狙い
        ・やり方（ステップ形式）
        ・注意点
      - 初心者でもイメージできる表現で

    09. フィッティングの方向性
      - 具体的なモデル名は出さない
      - 例: 「先調子で50g台のSフレックス」など、
        重さ・調子・硬さの方向性を提案
      - 今のスイングのまま使いやすい方向性
        ＋ 改善後に合いやすい方向性 の2軸で提案

    10. まとめ
      - 一番大事なポイントを2〜3個に絞って再提示
      - 前向きに背中を押す一言

    ▼共通ルール
    - すべて日本語
    - 初心者〜中級者向けに専門用語を噛み砕く
    - 「できていない」より「こうするともっと良くなる」
    - セクション番号とタイトルは必ず入れる（例: "02. 頭の安定性"）

    この条件を満たすレポート全文を出力してください。
    """

    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": textwrap.dedent(prompt)}],
        temperature=0.7,
    )

    return res["choices"][0]["message"]["content"].strip()

# ================================
# 4. LINE用のラッパー関数
# ================================
def generate_report_for_line(mode: str = "free",
                             club_type: str = "ドライバー",
                             user_level: str = "初心者") -> str:
    """
    LINEのサーバー側から呼ぶ入口。
    mode: "free" or "paid"
    """

    analysis = analyze_swing_stub(club_type=club_type, user_level=user_level)

    if mode == "free":
        return generate_free_report_text(analysis)
    else:
        return generate_paid_report_text(analysis)

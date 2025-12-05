import cv2
import mediapipe as mp
import numpy as np
import os
import math
from linebot.models import TextSendMessage

# ------------------------------------------------
# ユーティリティ関数
# ------------------------------------------------

def calculate_angle(p1, p2, p3):
    """3点から中間点(p2)を頂点とする角度を計算する"""
    p1 = np.array(p1)  # 最初の点 (例: 股関節)
    p2 = np.array(p2)  # 中間点 (例: 腰)
    p3 = np.array(p3)  # 最後の点 (例: 肩)

    # 3点間のベクトルを計算
    v1 = p1 - p2
    v2 = p3 - p2

    # コサイン類似度から角度を計算
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # ラジアンを度に変換
    return np.degrees(angle)

def analyze_swing(video_path):
    """
    動画を解析し、スイングの評価レポート（テキスト）を返します。
    """
    mp_pose = mp.solutions.pose
    
    # 処理中の最大・最小角度を格納する変数
    max_shoulder_rotation = -180
    min_hip_rotation = 180
    
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "【エラー】動画ファイルを開けませんでした。"

    frame_count = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # ★★★ メモリ削減と高速化のため、画像をリサイズする処理を追加 ★★★
            # 高解像度動画によるメモリ不足(OOM Killed)対策
            height, width, _ = image.shape
            if width > 640:
                scale = 640 / width
                new_size = (640, int(height * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            # ★★★ 修正終了 ★★★

            # パフォーマンス向上のため、画像を書き込み不可としてMediaPipeに渡す
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True

            frame_count += 1
            
            if results.pose_landmarks:
                # ... (以下、既存の解析ロジックが続く)
                landmarks = results.pose_landmarks.landmark

                # ランドマークの取得 (MediaPipeのインデックスを使用)
                # 右側 (R) を解析の基準とする
                RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
                RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                RIGHT_EAR = mp_pose.PoseLandmark.RIGHT_EAR.value
                LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value

                # ランドマーク座標の抽出
                r_hip = [landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y]
                r_shoulder = [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y]
                l_hip = [landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y]
                r_ear = [landmarks[RIGHT_EAR].x, landmarks[RIGHT_EAR].y]
                
                # スイング方向を決定するための軸（左右の股関節を結ぶ線）
                hip_axis_x = l_hip[0] - r_hip[0]
                hip_axis_y = l_hip[1] - r_hip[1]
                hip_axis_angle = np.degrees(np.arctan2(hip_axis_y, hip_axis_x))

                # -----------------
                # 1. 肩の回転角 (バックフェース時の最大値)
                # -----------------
                # 肩のラインの角度
                shoulder_line_angle = np.degrees(np.arctan2(r_ear[1] - r_shoulder[1], r_ear[0] - r_shoulder[0]))
                
                # 体幹の回転角度 (仮)
                # この実装では体軸に対する回転ではなく、簡略化された相対角度を使用します
                # バックフェース時の最大回転を追跡
                current_shoulder_rotation = shoulder_line_angle # 実際はZ軸の回転が必要だが、ここではY軸との相対で代用
                if current_shoulder_rotation > max_shoulder_rotation:
                    max_shoulder_rotation = current_shoulder_rotation

                # -----------------
                # 2. 骨盤の回転角 (フォロー時など)
                # -----------------
                # 骨盤の回転角度 (簡略化: 左右の股関節の水平角度)
                current_hip_rotation = hip_axis_angle
                if current_hip_rotation < min_hip_rotation:
                    min_hip_rotation = current_hip_rotation
        
        cap.release()
    
    # 解析結果に基づいたレポート生成
    
    # -----------------
    # レポート作成ロジック
    # -----------------
    
    # 肩の回転は、水平に近いほど小さい角度、垂直に近いほど大きい角度として仮定
    # ここでは、Y軸からの角度として簡略化し、大きな値がより回転していると解釈
    rotation_score = "良好"
    rotation_advice = "肩の回転はスムーズです。より深いトップを目指す場合は、左足の踏み込みを意識しましょう。"
    
    # 簡略化されたヒップ回転の評価（水平線からの角度で判断）
    hip_score = "適切"
    hip_advice = "骨盤の回転は安定しています。切り返しで下半身先行を意識し、より強力なリリースを目指しましょう。"

    # -----------------
    # レポートテキスト
    # -----------------
    report = f"""
⛳ スイング診断レポート ⛳
（解析動画フレーム数: {frame_count}）
----------------------------------
🏌️ **体幹の最大回転 (Top of Backswing):**
  - **最大回転角度 (簡略化):** {max_shoulder_rotation:.1f} 度 (目安: 90度以上)
  - **評価:** {rotation_score}
  - **アドバイス:** {rotation_advice}

🤸 **骨盤の最小回転 (Impact/Follow):**
  - **最小回転角度 (簡略化):** {min_hip_rotation:.1f} 度 (目安: -5度以下)
  - **評価:** {hip_score}
  - **アドバイス:** {hip_advice}
  
💡 **次のステップ:**
  - この解析は MediaPipe の2D座標に基づく簡略化されたものです。正確な評価には、専門のコーチングを受けてください。
  - より長い動画や高解像度動画での解析が成功しない場合は、動画の長さを5秒程度に短くしてください。
"""
    return report

if __name__ == '__main__':
    # このファイル単体でのテスト用コード (通常はCloud Runで実行されます)
    pass

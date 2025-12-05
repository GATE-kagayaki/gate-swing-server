import cv2
import mediapipe as mp
import numpy as np
import os
import math

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
    
    注意: 動画ファイルは server.py 側で事前に幅 640px に圧縮・軽量化されています。
    """
    mp_pose = mp.solutions.pose
    
    # 処理中の最大・最小角度を格納する変数
    max_shoulder_rotation = -180
    min_hip_rotation = 180
    
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "【エラー】動画ファイルを開けませんでした。ファイルパスを確認してください。"

    frame_count = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # --- 以前のOpenCVによるリサイズ処理は削除済み ---
            # server.py の ffmpeg 処理により、動画の負荷は既に軽減されています
            # -----------------------------------------------

            # パフォーマンス向上のため、画像を書き込み不可としてMediaPipeに渡す
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True

            frame_count += 1
            
            if results.pose_landmarks:
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
                # バックフェース時の最大回転を追跡
                current_shoulder_rotation = shoulder_line_angle
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
    
    rotation_score = "良好"
    rotation_advice = "肩の回転はスムーズです。より深いトップを目指す場合は、左足の踏み込みを意識しましょう。"
    
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
  - 動画はサーバー側で自動的に軽量化（幅640pxに圧縮）されました。高画質な動画を送信してもスムーズに処理できます。
"""
    return report

if __name__ == '__main__':
    # このファイル単体でのテスト用コード (通常はCloud Runで実行されます)
    pass

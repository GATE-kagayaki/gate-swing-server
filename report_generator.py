import cv2
import mediapipe as mp
import numpy as np

# MediaPipe設定
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    3点(a, b, c)から角度bを計算する関数
    a, b, c はそれぞれ [x, y] または [x, y, z] の座標リスト/配列
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def analyze_swing(video_path):
    """
    動画を解析し、スイングの評価レポート（テキスト）を返します。
    """
    cap = cv2.VideoCapture(video_path)
    
    # スイングデータの格納用
    landmarks_history = []
    
    # 簡易的なフェーズ検知用（今回は全フレーム解析後の最大値などで判定）
    max_shoulder_rotation = 0
    address_spine_angle = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
        
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # 高速化のため画像をリサイズ（任意）
            # image = cv2.resize(image, (640, 480))
            
            # RGB変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 推論実行
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 必要なランドマークの座標を取得
                # 11:左肩, 12:右肩, 23:左腰, 24:右腰
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # アドレス時の前傾角度（簡易判定：最初の数フレーム）
                if frame_count < 10:
                    # 股関節-肩-垂直線 などの角度計算が必要だが、今回は簡易的に「腰と肩のY座標差」などでチェック
                    pass 

                # 肩の回転角度（捻転）を簡易計算
                # 2D画像上での肩の傾きを計算
                shoulder_angle = calculate_angle(
                    [left_shoulder[0], 0], left_shoulder, right_shoulder
                )
                
                # 最大値を更新（バックスイングの深さの指標として）
                if shoulder_angle > max_shoulder_rotation:
                    max_shoulder_rotation = shoulder_angle

                landmarks_history.append(landmarks)
            
            frame_count += 1
            
    cap.release()
    
    # --- レポート作成 ---
    report_lines = []
    report_lines.append("⛳ スイング診断レポート ⛳")
    report_lines.append(f"総フレーム数: {frame_count}")
    
    # 診断ロジック（サンプル）
    # ※実際はカメラアングルや左右打ちによってロジックを調整する必要があります
    if max_shoulder_rotation > 0:
        report_lines.append(f"★ 肩の回転: {int(max_shoulder_rotation)}度")
        if max_shoulder_rotation > 80:
            report_lines.append("  → 深い捻転ができています！素晴らしいです。")
        elif max_shoulder_rotation < 45:
            report_lines.append("  → 少し回転が浅いかもしれません。もっと背中をターゲットに向ける意識を持ってみましょう。")
        else:
            report_lines.append("  → 標準的な回転量です。")
    
    report_lines.append("\nお疲れ様でした！次回も継続して計測しましょう。")
    
    return "\n".join(report_lines)

   

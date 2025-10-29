"""
웹캠을 통해 실시간 나이/성별 예측을 수행하는 스크립트
nCube 서버 연동 지원 (서버가 없어도 독립적으로 작동 가능)
"""

import cv2
import numpy as np
import argparse
import socket
import json
from tensorflow.keras.models import load_model

# nCube 서버 설정
HOST = '127.0.0.1'
PORT = 3105

# nCube 연결 (선택적)
upload_client = None
try:
    upload_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    upload_client.connect((HOST, PORT))
    print(f"[OK] nCube 서버 연결 성공: {HOST}:{PORT}")
except Exception as e:
    print(f"[WARNING] nCube 서버 연결 실패: {e}")
    print("[INFO] 서버 없이 독립적으로 작동합니다.")

def send_cin(con, msg):
    """
    nCube 서버로 데이터를 전송합니다.
    
    Args:
        con: 연결 이름
        msg: 전송할 메시지
    """
    if upload_client is None:
        return
    
    try:
        cin = {'ctname': con, 'con': msg}
        msg_encoded = (json.dumps(cin) + '<EOF>')
        upload_client.sendall(msg_encoded.encode('utf-8'))
        print(f"[SEND] {msg} to {con}")
    except Exception as e:
        print(f"[ERROR] 서버 전송 실패: {e}")

def close_server_connection():
    """
    서버 연결을 닫습니다.
    """
    global upload_client
    if upload_client is not None:
        try:
            upload_client.close()
            print("[OK] 서버 연결 종료")
        except Exception as e:
            print(f"[WARNING] 서버 연결 종료 실패: {e}")
        finally:
            upload_client = None

# 성별 딕셔너리
gender_dict = {0: "Male", 1: "Female"}

# 나이 구간
age_ranges = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

def detect_faces(image):
    """
    이미지에서 얼굴을 감지합니다.
    
    Args:
        image: OpenCV 이미지 (BGR 형식)
    
    Returns:
        faces: 감지된 얼굴 목록 (x, y, w, h)
    """
    # Haar Cascade를 사용한 얼굴 감지
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    return faces

def predict_age_gender(model, face_image):
    """
    모델을 사용하여 나이와 성별을 예측합니다.
    
    Args:
        model: 훈련된 모델
        face_image: 얼굴 이미지 (BGR 형식, 224x224)
    
    Returns:
        gender_label: 성별 라벨 (Male/Female)
        age_label: 나이 구간 라벨
        gender_prob: 성별 확률
        age_prob: 나이 구간 확률
    """
    # 전처리
    # BGR to RGB (모델은 RGB를 기대)
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # 정규화 (0~255 -> 0~1)
    normalized_face = face_rgb.astype('float32') / 255.0
    
    # 배치 차원 추가
    input_img = np.expand_dims(normalized_face, axis=0)
    
    # 예측
    prediction = model.predict(input_img, verbose=0)
    
    # 멀티태스크 모델 출력 처리
    # prediction은 리스트: [gender_output, age_output]
    gender_probs = prediction[0][0]  # [Male_prob, Female_prob]
    age_probs = prediction[1][0]     # [age_0-2_prob, age_3-9_prob, ..., age_70+_prob]
    
    # 성별 예측
    gender_pred = int(np.argmax(gender_probs))
    gender_prob = float(gender_probs[gender_pred])
    
    # 나이 구간 예측
    age_pred = int(np.argmax(age_probs))
    age_prob = float(age_probs[age_pred])
    
    gender_label = gender_dict[gender_pred]
    age_label = age_ranges[age_pred] if age_pred < len(age_ranges) else f"{age_pred} years"
    
    return gender_label, age_label, gender_prob, age_prob

def draw_prediction(frame, x, y, w, h, gender_label, age_label, gender_prob, age_prob):
    """
    프레임에 예측 결과를 표시합니다.
    
    Args:
        frame: OpenCV 프레임
        x, y, w, h: 얼굴 좌표
        gender_label: 성별 라벨
        age_label: 나이 구간 라벨
        gender_prob: 성별 확률
        age_prob: 나이 구간 확률
    """
    # 얼굴 사각형
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 결과 텍스트
    result_text = f"{gender_label}, {age_label}"
    confidence_text = f"{int(gender_prob*100)}%/{int(age_prob*100)}%"
    
    # 텍스트 배경
    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_bg_x = x + 5
    text_bg_y = y - 70
    text_bg_w = text_size[0] + 10
    text_bg_h = 55
    
    cv2.rectangle(frame, (text_bg_x, text_bg_y), (text_bg_x + text_bg_w, text_bg_y + text_bg_h), 
                  (0, 0, 0), -1)
    
    # 결과 텍스트
    cv2.putText(frame, result_text, (x+10, y-45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 확률 텍스트
    cv2.putText(frame, confidence_text, (x+10, y-25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

def run_webcam(model, camera_id=0, flip=True, show_stats=True):
    """
    웹캠을 통해 실시간 나이/성별 예측을 수행합니다.
    
    Args:
        model: 훈련된 모델
        camera_id: 카메라 ID (기본값: 0)
        flip: 화면 좌우 반전 여부 (기본값: True)
        show_stats: 통계 정보 표시 여부 (기본값: True)
    """
    # 웹캠 초기화
    print(f"\n웹캠 초기화 중... (카메라 ID: {camera_id})")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"[ERROR] 카메라를 열 수 없습니다 (ID: {camera_id})")
        return
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[OK] 웹캠 준비 완료!")
    print("\n컨트롤:")
    print("  - 'q': 종료")
    print("  - 's': 스크린샷 저장")
    print("  - 'r': 통계 정보 토글")
    print("\n영상을 시작합니다...\n")
    
    # 통계 정보
    frame_count = 0
    faces_detected = 0
    show_stats_flag = show_stats
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                print("[WARNING] 프레임을 읽을 수 없습니다.")
                break
            
            # 화면 좌우 반전
            if flip:
                frame = cv2.flip(frame, 1)
            
            # 얼굴 감지
            faces = detect_faces(frame)
            
            # 감지된 얼굴에 대해 예측 수행
            for (x, y, w, h) in faces:
                # 얼굴 영역 추출
                roi = frame[y:y+h, x:x+w]
                
                # 224x224로 리사이즈
                face_resized = cv2.resize(roi, (224, 224))
                
                # 예측 수행
                gender_label, age_label, gender_prob, age_prob = predict_age_gender(model, face_resized)
                
                # 결과 표시
                draw_prediction(frame, x, y, w, h, gender_label, age_label, gender_prob, age_prob)
                
                # nCube 서버로 결과 전송
                send_cin("age_gender_prediction", f"{gender_label},{age_label}")
                
                faces_detected += 1
            
            # 통계 정보 표시
            if show_stats_flag:
                frame_count += 1
                stats_text = f"Frames: {frame_count} | Faces detected: {faces_detected}"
                cv2.putText(frame, stats_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 프레임 표시
            cv2.imshow("Real-time Age/Gender Prediction - Press 'q' to quit", frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[OK] 사용자 요청으로 종료합니다.")
                break
            elif key == ord('s'):
                # 스크린샷 저장
                screenshot_path = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"[OK] 스크린샷 저장: {screenshot_path}")
            elif key == ord('r'):
                # 통계 정보 토글
                show_stats_flag = not show_stats_flag
                print(f"[INFO] 통계 정보 표시: {'ON' if show_stats_flag else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n[INFO] 인터럽트로 종료합니다.")
    finally:
        # 리소스 해제
        cap.release()
        cv2.destroyAllWindows()
        close_server_connection()
        print("\n[OK] 웹캠 종료 완료")
        print(f"[INFO] 총 처리된 프레임: {frame_count}")
        print(f"[INFO] 감지된 얼굴 수: {faces_detected}")

def main():
    """
    메인 실행 함수
    """
    parser = argparse.ArgumentParser(description='웹캠을 통한 실시간 나이/성별 예측')
    parser.add_argument('-m', '--model', type=str, default='age_gender_model.h5',
                       help='모델 파일 경로 (기본값: age_gender_model.h5)')
    parser.add_argument('-c', '--camera', type=int, default=0,
                       help='카메라 ID (기본값: 0)')
    parser.add_argument('--no-flip', action='store_true',
                       help='화면 좌우 반전 비활성화')
    parser.add_argument('--no-stats', action='store_true',
                       help='통계 정보 미표시')
    
    args = parser.parse_args()
    
    # 모델 로드
    print("=" * 70)
    print("실시간 나이/성별 예측 - 웹캠 모드")
    print("=" * 70)
    print(f"\n모델 로딩: {args.model}")
    
    import os
    if not os.path.exists(args.model):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {args.model}")
        return
    
    try:
        model = load_model(args.model, compile=False)
        print("[OK] 모델 로드 완료!")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return
    
    # 웹캠 실행
    run_webcam(
        model=model,
        camera_id=args.camera,
        flip=not args.no_flip,
        show_stats=not args.no_stats
    )

if __name__ == "__main__":
    main()


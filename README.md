# AIoT Raspberry Pi 기반 실시간 나이/성별 예측

## 개요
TensorFlow Hub에 mobilenet_v2 모델을 사용하여, 라즈베리파이에서 실행되는 실시간 나이/성별 예측 시스템 구현. kaggle에 age_gender 데이터를 활용하여 모델 학습 진행. 기존 실습 자료인 facial_recognition.py를 베이스로 age_gender_prediction.py 구현.

## 파일 구조
```
├── age_gender_prediction.py    # 메인 실행 파일
├── age_gender_model.h5         # 나이/성별 예측 모델
├── age_gender_requirements.txt # 필요한 패키지 목록
└── README.md                   # 이 파일
```

## 데이터 전처리

### 1. 이미지 리사이즈
- **기존**: 48x48 그레이스케일 이미지
- **변경**: 224x224 RGB 컬러 이미지

### 2. 픽셀 데이터 파싱
- **기존**: 문자열
- **변경**: 배열

### 3. 이미지 채널 확장
- **기존**: 1채널(grayscale)
- **변경**: 3채널(RGB)

  ### 4. 나이 라벨 처리
- **기존**: 연속형(19, 23, 27)
- **변경**: 구간형(10-19, 20-29)

  ### 5. 데이터 정규화
- **기존**: 0~255
- **변경**: 0~1

## 주요 기능

### 1. **실시간 얼굴 감지 및 예측**
- 웹캠에서 실시간으로 얼굴 감지
- 감지된 얼굴의 나이와 성별 예측
- 결과를 화면에 표시

### 2. **nCube 연동**
- 예측 결과를 소켓 통신을 통해 외부 서버로 전송
- Host: 127.0.0.1, Port: 3105

## 모델 출력 형식
```python
gender_dict = {0: "Male", 1: "Female"}
age_ranges = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
```

**예상 출력 구조**:
- **총 13개 클래스**: 성별 2개 + 나이구간 11개
- 출력 형식: `[Male_prob, Female_prob, age_0-2_prob, age_3-9_prob, ..., age_90+_prob]`

## 실행 방법

1. **의존성 설치**:
   ```bash
   pip install -r age_gender_requirements.txt
   ```

2. **모델 파일 준비**:
   - `age_gender_model.h5` 파일을 프로젝트 폴더에 배치

3. **실행**:
   ```bash
   python age_gender_prediction.py
   ```

## nCube 통신

시스템은 다음 형식으로 nCube에 데이터를 전송합니다:
```json
{
  "ctname": "age_gender",
  "con": "Male,30-39"
}
```

## 라이센스 및 참고사항
- facial_recognition.py 기반으로 수정
- nCube 통신 프로토콜 유지
- 라즈베리파이 최적화 고려

# 나이/성별 예측 시스템 사용 가이드

## 개요
이 프로젝트는 라즈베리파이에서 실행되는 실시간 나이/성별 예측 시스템입니다. 기존 감정 인식 코드를 나이/성별 예측으로 변경한 AIoT 기반 시스템입니다.

## 파일 구조
```
├── age_gender_prediction.py    # 메인 실행 파일
├── age_gender_model.h5         # 나이/성별 예측 모델 (사용자가 준비해야 함)
├── age_gender_requirements.txt # 필요한 패키지 목록
└── README.md                   # 이 파일
```

## 주요 변경사항

### 1. 모델 입력 변경
- **기존**: 48x48 그레이스케일 이미지
- **변경**: 224x224 RGB 컬러 이미지

### 2. 출력 형식 변경
- **기존**: 감정 라벨 (7개 클래스)
- **변경**: 성별 (Male/Female) + 나이 구간

### 3. nCube 통신 메시지 변경
- **기존**: `send_cin("emotion", "Happy")`
- **변경**: `send_cin("age_gender", "Male,30-39")`

## 모델 준비 방법

### 옵션 1: 사전 훈련된 모델 사용
```python
# 예시: TensorFlow Hub에서 모델 다운로드
import tensorflow_hub as hub

# 나이/성별 예측 모델 (예시)
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)
model.save('age_gender_model.h5')
```

### 옵션 2: 커스텀 모델 훈련
```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# MobileNetV2 기반 모델 생성
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = GlobalAveragePooling2D()(base_model.output)
gender_output = Dense(2, activation='softmax', name='gender')(x)  # 성별
age_output = Dense(9, activation='softmax', name='age')(x)       # 나이 구간

model = Model(inputs=base_model.input, outputs=[gender_output, age_output])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## 모델 출력 형식 조정

코드의 예측 부분을 모델 출력에 맞게 조정해야 합니다:

```python
# 모델 출력이 [gender_logits, age_logits] 형태인 경우
if len(prediction) == 2:
    gender_pred = int(np.argmax(prediction[0]))
    age_pred = int(np.argmax(prediction[1]))
else:
    # 단일 출력인 경우 (예: [gender_prob, age_prob, ...])
    gender_pred = int(np.argmax(prediction[0][:2]))
    age_pred = int(np.argmax(prediction[0][2:11]))  # 나이 구간 9개
```

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

## 명령행 옵션
- `-f, --fullscreen`: 전체화면 모드
- `-d, --debug`: 디버그 정보 표시
- `-fl, --flip`: 비디오 신호 뒤집기

## nCube 통신

시스템은 다음 형식으로 nCube에 데이터를 전송합니다:
```json
{
  "ctname": "age_gender",
  "con": "Male,30-39"
}
```

## 성능 최적화 팁

1. **모델 크기 최적화**: MobileNet, EfficientNet-Lite 등 경량 모델 사용
2. **입력 크기 조정**: 필요에 따라 224x224에서 더 작은 크기로 변경
3. **프레임 스킵**: 모든 프레임이 아닌 일정 간격으로 예측 수행
4. **GPU 가속**: 라즈베리파이 4에서는 TensorFlow Lite 사용 고려

## 문제 해결

### 일반적인 문제들:
1. **메모리 부족**: 모델 크기를 줄이거나 배치 크기 감소
2. **느린 추론**: TensorFlow Lite로 모델 변환 고려
3. **정확도 저하**: 더 큰 입력 크기나 고품질 모델 사용

### 디버그 모드:
```bash
python age_gender_prediction.py --debug
```

## 라이센스 및 참고사항
- 기존 감정 인식 코드를 기반으로 수정
- nCube 통신 프로토콜 유지
- 라즈베리파이 최적화 고려

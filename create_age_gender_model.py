"""
나이/성별 예측 모델 생성 및 저장 스크립트
TensorFlow Hub의 사전 훈련된 모델을 활용하여 나이/성별 예측 모델을 생성합니다.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

def create_age_gender_model():
    """
    TensorFlow Hub의 사전 훈련된 모델을 기반으로 나이/성별 예측 모델을 생성합니다.
    최신 TensorFlow 버전 호환성을 고려합니다.
    """
    print("사전 훈련된 모델을 로딩 중...")
    
    # TensorFlow Hub에서 MobileNetV2 모델 로드 (경량화된 모델)
    # 실제 얼굴 이미지에 적합한 모델 사용
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    
    try:
        # Hub 모델 로드 - 최신 버전 호환성 고려
        hub_layer = hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)
        
        # 모델 구성 - InputLayer 명시적 정의로 호환성 문제 해결
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3), name='input_layer'),
            hub_layer,
            Dropout(0.3),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            # 나이와 성별을 동시에 예측하는 출력층
            Dense(11, activation='softmax', name='age_gender_output')  # 성별(2) + 나이구간(9)
        ])
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("모델 생성 완료!")
        print(f"모델 구조: {model.summary()}")
        
        return model
        
    except Exception as e:
        print(f"Hub 모델 로딩 실패: {e}")
        print("대안 모델을 생성합니다...")
        
        # 대안: 직접 MobileNetV2 기반 모델 생성
        return create_alternative_model()

def create_alternative_model():
    """
    TensorFlow Hub를 사용할 수 없을 때의 대안 모델
    최신 TensorFlow 버전 호환성을 고려합니다.
    """
    from tensorflow.keras.applications import MobileNetV2
    
    # MobileNetV2 베이스 모델 (사전 훈련된 가중치 사용)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 베이스 모델의 가중치를 고정 (전이 학습)
    base_model.trainable = False
    
    # 새로운 분류기 추가
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 나이/성별 예측 출력 (성별 2개 + 나이구간 9개 = 11개 클래스)
    outputs = Dense(11, activation='softmax', name='age_gender_output')(x)
    
    # 명시적 InputLayer로 호환성 문제 해결
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer')
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("대안 모델 생성 완료!")
    return model

def create_dummy_training_data():
    """
    모델 구조를 확인하기 위한 더미 데이터 생성
    실제 사용시에는 실제 얼굴 데이터셋으로 교체해야 합니다.
    """
    # 더미 이미지 데이터 (224x224x3)
    dummy_images = np.random.random((100, 224, 224, 3)).astype(np.float32)
    
    # 더미 라벨 (성별 2개 + 나이구간 9개)
    # 0-1: 성별 (Male, Female)
    # 2-10: 나이구간 (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)
    dummy_labels = np.random.randint(0, 11, (100,)).astype(np.int32)
    dummy_labels = tf.keras.utils.to_categorical(dummy_labels, 11)
    
    return dummy_images, dummy_labels

def save_model_for_inference(model, model_path='age_gender_model.h5'):
    """
    추론용 모델을 저장합니다.
    최신 TensorFlow 버전 호환성을 고려합니다.
    """
    try:
        # H5 형식으로 저장 (compile=False로 호환성 문제 해결)
        model.save(model_path, save_format='h5', include_optimizer=False)
        print(f"모델이 {model_path}에 저장되었습니다.")
        
        # 모델 정보 출력
        print(f"모델 입력 크기: {model.input_shape}")
        print(f"모델 출력 크기: {model.output_shape}")
        
        # 저장된 모델 테스트 로딩
        try:
            test_model = tf.keras.models.load_model(model_path, compile=False)
            print("저장된 모델 테스트 로딩 성공!")
            return True
        except Exception as test_e:
            print(f"저장된 모델 테스트 로딩 실패: {test_e}")
            return False
        
    except Exception as e:
        print(f"모델 저장 실패: {e}")
        print("대안 저장 방법을 시도합니다...")
        
        # 대안 저장 방법
        try:
            # SavedModel 형식으로 저장
            model.save(model_path.replace('.h5', '_savedmodel'), save_format='tf')
            print(f"모델이 SavedModel 형식으로 저장되었습니다.")
            return True
        except Exception as e2:
            print(f"대안 저장 방법도 실패: {e2}")
            return False

def main():
    """
    메인 실행 함수
    """
    print("=== 나이/성별 예측 모델 생성 시작 ===")
    
    # 1. 모델 생성
    model = create_age_gender_model()
    
    # 2. 더미 데이터로 모델 구조 확인
    print("\n모델 구조 확인을 위한 더미 학습...")
    dummy_images, dummy_labels = create_dummy_training_data()
    
    # 더미 학습 (실제 학습이 아닌 구조 확인용)
    try:
        model.fit(dummy_images, dummy_labels, epochs=1, batch_size=16, verbose=1)
        print("더미 학습 완료!")
    except Exception as e:
        print(f"더미 학습 중 오류 (정상적일 수 있음): {e}")
    
    # 3. 모델 저장
    print("\n모델 저장 중...")
    success = save_model_for_inference(model)
    
    if success:
        print("\n=== 모델 생성 및 저장 완료 ===")
        print("이제 age_gender_prediction.py에서 이 모델을 사용할 수 있습니다.")
        print("\n주의사항:")
        print("- 이 모델은 더미 데이터로 생성된 것으로 실제 성능이 낮습니다.")
        print("- 실제 사용을 위해서는 얼굴 이미지 데이터셋으로 재훈련이 필요합니다.")
        print("- 나이/성별 라벨링된 데이터셋을 사용하여 fine-tuning을 수행하세요.")
    else:
        print("모델 저장에 실패했습니다.")

if __name__ == "__main__":
    main()

"""
간단한 나이/성별 예측 모델 생성 스크립트
TensorFlow 호환성 문제를 해결하기 위한 최소한의 모델
"""

import tensorflow as tf
import numpy as np
import os

def create_simple_age_gender_model():
    """
    간단하고 호환성이 좋은 나이/성별 예측 모델을 생성합니다.
    """
    print("간단한 나이/성별 예측 모델을 생성합니다...")
    
    # 간단한 CNN 모델 생성
    model = tf.keras.Sequential([
        # 입력층 - 명시적으로 정의하여 호환성 문제 해결
        tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer'),
        
        # 첫 번째 컨볼루션 블록
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 두 번째 컨볼루션 블록
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 세 번째 컨볼루션 블록
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 네 번째 컨볼루션 블록
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 글로벌 평균 풀링
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # 완전 연결층
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # 출력층 (성별 2개 + 나이구간 9개 = 11개 클래스)
        tf.keras.layers.Dense(11, activation='softmax', name='age_gender_output')
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("모델 생성 완료!")
    print(f"모델 구조:")
    model.summary()
    
    return model

def create_dummy_data():
    """
    모델 테스트를 위한 더미 데이터 생성
    """
    print("더미 데이터를 생성합니다...")
    
    # 더미 이미지 데이터
    dummy_images = np.random.random((50, 224, 224, 3)).astype(np.float32)
    
    # 더미 라벨 (성별 2개 + 나이구간 9개)
    dummy_labels = np.random.randint(0, 11, (50,)).astype(np.int32)
    dummy_labels = tf.keras.utils.to_categorical(dummy_labels, 11)
    
    return dummy_images, dummy_labels

def save_model_safely(model, model_path='age_gender_model.h5'):
    """
    안전하게 모델을 저장합니다.
    """
    print(f"모델을 {model_path}에 저장합니다...")
    
    try:
        # H5 형식으로 저장 (옵티마이저 제외)
        model.save(model_path, save_format='h5', include_optimizer=False)
        print("H5 형식으로 저장 성공!")
        
        # 저장된 모델 테스트
        test_model = tf.keras.models.load_model(model_path, compile=False)
        print("저장된 모델 테스트 로딩 성공!")
        
        return True
        
    except Exception as e:
        print(f"H5 저장 실패: {e}")
        
        # 대안: SavedModel 형식으로 저장
        try:
            savedmodel_path = model_path.replace('.h5', '_savedmodel')
            model.save(savedmodel_path, save_format='tf')
            print(f"SavedModel 형식으로 저장 성공: {savedmodel_path}")
            return True
        except Exception as e2:
            print(f"SavedModel 저장도 실패: {e2}")
            return False

def main():
    """
    메인 실행 함수
    """
    print("=== 간단한 나이/성별 예측 모델 생성 시작 ===")
    
    # TensorFlow 버전 확인
    print(f"TensorFlow 버전: {tf.__version__}")
    
    # 1. 모델 생성
    model = create_simple_age_gender_model()
    
    # 2. 더미 데이터로 모델 테스트
    print("\n더미 데이터로 모델 테스트...")
    dummy_images, dummy_labels = create_dummy_data()
    
    try:
        # 더미 학습 (구조 확인용)
        model.fit(dummy_images, dummy_labels, epochs=1, batch_size=8, verbose=1)
        print("더미 학습 완료!")
    except Exception as e:
        print(f"더미 학습 중 오류 (정상적일 수 있음): {e}")
    
    # 3. 모델 저장
    print("\n모델 저장 중...")
    success = save_model_safely(model)
    
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

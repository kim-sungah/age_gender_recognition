"""
CSV 데이터를 사용하여 나이/성별 예측 모델을 훈련하는 스크립트
age_gender.csv 파일을 직접 사용하여 실제 데이터로 모델을 학습시킵니다.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import cv2
import os

def load_csv_data(csv_path='age_gender.csv', test_size=0.2):
    """
    CSV 파일에서 데이터를 로드하고 전처리합니다.
    
    Returns:
        X_train, X_test: 전처리된 이미지 데이터
        y_train, y_test: 나이 (regression) 또는 클래스 (classification) 라벨
    """
    print("CSV 데이터 로딩 중...")
    df = pd.read_csv(csv_path)
    
    print(f"총 샘플 수: {len(df)}")
    print(f"나이 범위: {df['age'].min()} ~ {df['age'].max()}")
    print(f"성별 분포: Male={len(df[df['gender']==0])}, Female={len(df[df['gender']==1])}")
    
    # 픽셀 데이터 파싱 및 이미지로 변환
    print("\n이미지 데이터 변환 중...")
    X = []
    ages = []
    genders = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"진행 중: {idx}/{len(df)}")
        
        try:
            # 픽셀 데이터를 배열로 변환
            pixels = np.array([int(p) for p in row['pixels'].split()])
            
            # 픽셀 수 확인 (48x48 = 2304)
            if len(pixels) != 2304:
                print(f"경고: 샘플 {idx}의 픽셀 수가 2304가 아닙니다: {len(pixels)}")
                continue
            
            # 48x48 이미지로 reshape
            img = pixels.reshape(48, 48).astype(np.uint8)
            
            # 3채널로 확장 (Grayscale to RGB)
            img_rgb = np.stack([img, img, img], axis=-1)
            
            # 224x224로 리사이즈
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            X.append(img_resized)
            
            ages.append(row['age'])
            genders.append(row['gender'])
            
        except Exception as e:
            print(f"에러: 샘플 {idx} 처리 중 오류 발생: {e}")
            continue
    
    # 배열이 비어있지 않은지 확인
    if len(X) == 0:
        raise ValueError("처리된 이미지가 없습니다!")
    
    X = np.array(X)
    ages = np.array(ages)
    genders = np.array(genders)
    
    print(f"성공적으로 처리된 샘플 수: {len(X)}")
    print(f"이미지 shape: {X.shape}")
    print(f"픽셀 값 범위: [{X.min()}, {X.max()}]")
    
    # 데이터 정규화 (0~255 -> 0~1)
    X = X.astype('float32') / 255.0
    
    # 데이터 분할
    X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
        X, ages, genders, test_size=test_size, random_state=42
    )
    
    print(f"\n훈련 세트: {len(X_train)} 개")
    print(f"테스트 세트: {len(X_test)} 개")
    
    return X_train, X_test, age_train, age_test, gender_train, gender_test

def create_age_gender_model():
    """
    나이와 성별을 예측하는 멀티태스크 모델 생성
    age_gender_prediction.py의 예상 형식에 맞춰서 생성
    """
    print("나이/성별 예측 모델 생성 중...")
    
    # 입력 레이어
    input_img = tf.keras.layers.Input(shape=(224, 224, 3))
    
    # 공유 백본
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    
    # 공유 헤드
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # 성별 출력 (2개 클래스)
    gender_output = Dense(2, activation='softmax', name='gender_output')(x)
    
    # 나이 구간 출력 (9개 클래스)
    age_output = Dense(9, activation='softmax', name='age_output')(x)
    
    # 모델 생성
    model = tf.keras.models.Model(inputs=input_img, outputs=[gender_output, age_output])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'gender_output': 'categorical_crossentropy', 'age_output': 'categorical_crossentropy'},
        loss_weights={'gender_output': 1.0, 'age_output': 1.0},
        metrics={'gender_output': 'accuracy', 'age_output': 'accuracy'}
    )
    
    return model

def age_to_category(age):
    """
    나이를 9개 구간으로 변환
    """
    if age <= 2:
        return 0
    elif age <= 9:
        return 1
    elif age <= 19:
        return 2
    elif age <= 29:
        return 3
    elif age <= 39:
        return 4
    elif age <= 49:
        return 5
    elif age <= 59:
        return 6
    elif age <= 69:
        return 7
    else:
        return 8

def train_age_gender_model():
    """
    나이와 성별을 동시에 예측하는 멀티태스크 모델 훈련
    age_gender_prediction.py와 호환되는 모델
    """
    print("\n=== 나이/성별 멀티태스크 모델 훈련 ===")
    
    # 데이터 로드
    X_train, X_test, age_train, age_test, gender_train, gender_test = load_csv_data()
    
    # 나이를 카테고리로 변환
    age_category_train = np.array([age_to_category(a) for a in age_train])
    age_category_test = np.array([age_to_category(a) for a in age_test])
    
    # 원-핫 인코딩
    age_category_train = tf.keras.utils.to_categorical(age_category_train, 9)
    age_category_test = tf.keras.utils.to_categorical(age_category_test, 9)
    gender_train_categorical = tf.keras.utils.to_categorical(gender_train, 2)
    gender_test_categorical = tf.keras.utils.to_categorical(gender_test, 2)
    
    # 모델 생성
    model = create_age_gender_model()
    model.summary()
    
    # 훈련
    print("\n모델 훈련 시작...")
    history = model.fit(
        X_train, 
        {'gender_output': gender_train_categorical, 'age_output': age_category_train},
        validation_data=(X_test, {'gender_output': gender_test_categorical, 'age_output': age_category_test}),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # 모델 저장 (age_gender_prediction.py가 사용할 이름)
    model.save('age_gender_model.h5', save_format='h5')
    print("\n나이/성별 모델 저장 완료: age_gender_model.h5")
    
    return model, history


def main():
    """
    메인 실행 함수
    """
    print("=" * 70)
    print("CSV 데이터를 사용한 나이/성별 예측 모델 훈련")
    print("(age_gender_prediction.py와 호환되는 모델)")
    print("=" * 70)
    
    # 나이/성별 멀티태스크 모델 훈련
    train_age_gender_model()

if __name__ == "__main__":
    main()


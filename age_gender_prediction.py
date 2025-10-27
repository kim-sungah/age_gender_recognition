import argparse
import os
import sys
import time

import cv2
import numpy as np
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
import sys
import subprocess

import socket, json

from tensorflow.keras.models import load_model

HOST = '127.0.0.1'
PORT = 3105

# nCube 연결 (선택적)
upload_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
upload_client.connect((HOST, PORT))

def send_cin(con,msg) :
    cin = {'ctname': con, 'con': msg}
    msg = (json.dumps(cin) + '<EOF>')
    upload_client.sendall(msg.encode('utf-8'))
        
    print (f"send {msg} to {con}")


if sys.platform == 'linux':
    from gpiozero import CPUTemperature

# input arg parsing
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fullscreen',
                    help='Display window in full screen', action='store_true')
parser.add_argument(
    '-d', '--debug', help='Display debug info', action='store_true')
parser.add_argument(
    '-fl', '--flip', help='Flip incoming video signal', action='store_true')
args = parser.parse_args()

model = load_model('age_gender_model.h5', compile=False)

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary for gender prediction
gender_dict = {0: "Male", 1: "Female"}

# age ranges for age prediction (if using classification)
age_ranges = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

def get_gpu_temp():
    temp = subprocess.check_output(['vcgencmd measure_temp | egrep -o \'[0-9]*\.[0-9]*\''],
                                    shell=True, universal_newlines=True)
    return str(float(temp))

# start the webcam feed
cap = cv2.VideoCapture(0)
zoom_scale = 3
while True:
    # time for fps
    start_time = time.time()

    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # if args.flip:
    #     frame = cv2.flip(frame, 1)
    if not ret:
        break

    height, width, _ = frame.shape
    centerX, centerY = int(height / 2), int(width / 2)
    radiusX, radiusY = int(height / (2 * zoom_scale)), int(width / (2 * zoom_scale))

    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY

    cropped = frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))


    # Use OpenCV's built‑in Haar cascade path so that the file is always found
    # without needing a local copy. cv2.data.haarcascades points to the
    # directory containing pretrained cascade XML files.
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    max_area = 0
    x_max = y_max = w_max = h_max = 0
    for (x, y, w, h) in faces:
        if w*h > max_area:
            x_max, y_max, w_max, h_max = x, y, w, h
            max_area = w*h

    if max_area > 0:
        cv2.rectangle(frame, (x_max, y_max-50), (x_max+w_max, y_max+h_max+10), (255, 0, 0), 2)
        
        # Extract face region in color (RGB)
        roi_color = frame[y_max:y_max + h_max, x_max:x_max + w_max]
        
        # Resize to model input size (typically 224x224 for age/gender models)
        resized_face = cv2.resize(roi_color, (224, 224))
        
        # Normalize pixel values to [0, 1]
        normalized_face = resized_face.astype('float32') / 255.0
        
        # Add batch dimension
        input_img = np.expand_dims(normalized_face, axis=0)
        
        # Make prediction
        prediction = model.predict(input_img)
        
        # 모델 출력 처리 (11개 클래스: 성별 2개 + 나이구간 9개)
        # 출력 형식: [Male_prob, Female_prob, age_0-2_prob, age_3-9_prob, ..., age_70+_prob]
        prediction_probs = prediction[0]
        
        # 성별 예측 (첫 2개 값)
        gender_pred = int(np.argmax(prediction_probs[:2]))
        
        # 나이 구간 예측 (나머지 9개 값)
        age_pred = int(np.argmax(prediction_probs[2:]))
        
        gender_label = gender_dict[gender_pred]
        age_label = age_ranges[age_pred] if age_pred < len(age_ranges) else f"{age_pred} years"
        
        # Display results
        result_text = f"{gender_label}, {age_label}"
        cv2.putText(frame, result_text, (x_max+20, y_max-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Send to nCube
        send_cin("age_gender_prediction", f"{gender_label},{age_label}")
    
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

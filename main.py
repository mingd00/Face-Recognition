from fastapi import FastAPI, File, UploadFile
from moviepy.editor import *
import cv2
import tempfile
import numpy as np
from scipy.ndimage import zoom
from fer import FER

path = './'
app = FastAPI()

# FER 모델 초기화
detector = FER()

# 얼굴 인식 함수
def detect_face(frame):
    cascPath = path + 'Models/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(detected_faces) > 0:
        return gray, detected_faces[0]
    else:
        return gray, None

# 얼굴 특징 추출 함수
def extract_face_features(gray, detected_face, offset_coefficients=(0.075, 0.05), shape_x=48, shape_y=48):
    if detected_face is None:
        return None
    x, y, w, h = detected_face
    horizontal_offset = int(np.floor(offset_coefficients[0] * w))
    vertical_offset = int(np.floor(offset_coefficients[1] * h))
    extracted_face = gray[
        y + vertical_offset : y + h,
        x + horizontal_offset : x - horizontal_offset + w
    ]
    new_extracted_face = zoom(
        extracted_face,
        (shape_x / extracted_face.shape[0], shape_y / extracted_face.shape[1])
    )
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face

# 영상 처리 함수
async def convert_video(file: UploadFile):
    video_data = await file.read()
    frames = []

    with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
        temp_video.write(video_data)
        temp_video.seek(0)
        cap = cv2.VideoCapture(temp_video.name)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 5프레임마다 저장
            if frame_count % 15 == 0:
                timestamp = frame_count / 30  # 현재 프레임의 시간 (초)
                frames.append((frame, timestamp)) 
            frame_count += 1
        cap.release()
    return frames

def process_frames(frames):
    total_score = 100
    minus_log = {
        'angry': [],
        'disgust': [],
        'fear': [],
        'happy': [],
        'sad': [],
        'surprise': [],
        'neutral': []
    }
    
    for frame, timestamp in frames:
        gray, detected_face = detect_face(frame)
        if detected_face is not None:
            extracted_face = extract_face_features(gray, detected_face)
            if extracted_face is not None:
                # FER 라이브러리로 감정 예측
                result = detector.detect_emotions(frame)  # FER 모델로 감정 예측
                if result:
                    emotions = result[0]['emotions']
                    max_emotion = max(emotions, key=emotions.get)
                    
                    if max_emotion in ['angry', 'disgust', 'fear', 'surprise']:
                        total_score -= 1
                        print(f"예측값: {max_emotion}으로 1점 감점, 현재 점수: {total_score}")
                        minus_log[max_emotion].append(f'{timestamp:.2f}')  # 감정별로 시간 기록
                    else:
                        print(f"예측값: {max_emotion}, 현재 점수: {total_score}")
            else:
                print("얼굴 추출 실패")
        else:
            print("얼굴 탐지 실패")
            
    # 비어 있는 리스트를 제외한 minus_log 필터링
    minus_log = {emotion: times for emotion, times in minus_log.items() if times}  # 비어 있는 리스트 제거

    final_result = {
        "total_score": total_score,
        "minus_log": minus_log  # 비어있는 감정 항목 제외
    }       
    print(final_result)
    return final_result

# FastAPI 엔드포인트
@app.post("/upload")
async def emotion_recognition_api(file: UploadFile = File(...)):
    frames = await convert_video(file)
    predictions = process_frames(frames)
    response = {"frame_results": predictions}
    return response

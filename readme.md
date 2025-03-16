# AI 기반 발표/면접 시스템 - 표정 인식 모델 

### 전체 Workflow

[main.py 코드 바로가기](https://github.com/mingd00/Face-Recognition/blob/main/main.py)

![image](https://github.com/user-attachments/assets/170eb1fc-b66a-4473-8258-0fb93a419bc2)

1. **프레임 추출**: 영상에서 0.5초 간격으로 프레임을 추출하여 이미지로 변환
2. **얼굴 탐지**: Haar Cascade 분류기를 활용해 얼굴 영역 검출
3. **표정 인식**: 얼굴 이미지를 표정 인식 모델에 입력하여 감정 예측 및 점수 반환

---

### 1️⃣ **Baseline 모델 비교**

[모델 비교 코드 바로가기](https://github.com/mingd00/Face-Recognition/blob/main/Notebooks/deepface_fer_model_test.ipynb)

- **FER model**: 정확도 **72.3%**, 빠른 처리 속도
- **Deepface model**: 정확도 **97.5%**, 하지만 **5배 이상 느린 속도**

> ✅ 속도와 정확도의 균형을 맞추기 위해 Custom CNN 모델과 EfficientNet 파인튜닝을 추가로 시도
> 

---

### 2️⃣ **학습 데이터 구성**

[학습 데이터 전처리 코드 바로가기](https://github.com/mingd00/Face-Recognition/blob/main/Notebooks/Data_Merging.ipynb)

- **FER2013 dataset** + **AI Hub dataset** 활용
- **총 35만 개의 이미지를 전처리** 했지만 Colab 환경에서 적정 크기 유지를 위해 **표정당 FER 5,000장 + AI Hub 5,000장 샘플링**

---

### 3️⃣ **Custom CNN Model**

[Custom CNN 코드 바로가기](https://github.com/mingd00/Face-Recognition/blob/main/Notebooks/Xception(with_total_data).ipynb)

- **구조**: Conv2D → MaxPooling2D → Flatten → Dense
- **하이퍼파라미터**: Adam Optimizer, Categorical Crossentropy
- **콜백**: EarlyStopping, ModelCheckpoint
- **결과**: 약 **60% 정확도**

---

### 4️⃣ **EfficientNetB0 Fine-Tuning**

[파인튜닝 코드 바로가기](https://github.com/mingd00/Face-Recognition/blob/main/Notebooks/EfficientNet_v2.ipynb)

1. **1단계 (전이 학습)**
    - **EfficientNetB0** + 커스텀 분류기 학습 (10 epoch)
    - 기본적인 **데이터 증강** 적용
2. **2단계 (파인튜닝)**
    - feature extractor까지 학습에 포함
    - **데이터 증강 강화** + **학습률 감소 스케줄링(ExponentialDecay)**
    - 추가 **50 epoch** 학습
    - **결과**: 약 **60% 정확도**

---

### 5️⃣ **최종 모델 선택**

- **FER 모델**의 **적절한 속도(실시간 적용 가능)** 및 **72.3% 정확도**를 고려하여 **최종 채택**

---

### 패키지 설치

```
pip install -r requirements.txt
```

---

### 실행

```
uvicorn main:app --reload
```

---

### 기술스택

- Python(Numpy, Pandas, Matplotlib)
- Tensorflow/Keras
- OpenCV
- FastAPI
- 

---

## 라이센스

https://github.com/fshnkarimi/FaceEmotionRecognition

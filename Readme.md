# 코드 파일 정리중! In the process of organizing files and Code

### 기본 시각화 코드
<details>
<summary>🧑‍💻code🧑‍💻</summary>

```py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("png2x") # svg, retina, png2x ...
mpl.style.use("seaborn-v0_8")
mpl.rcParams.update({"figure.constrained_layout.use": True})
sns.set_context("paper") 
sns.set_palette("Set2") 
sns.set_style("whitegrid") 

# 시스템 폰트패밀리에 따라 변경
plt.rc("font", family = "NanumSquareRound")
plt.rcParams["axes.unicode_minus"] = False
```
</details>

### 코드파일 목차 및 내용
<details>
<summary>🧑‍💻list🧑‍💻</summary>

### 전체 코드 파일

- 00_basics
  - [00_파이썬_기초.ipynb](/code/00_basics/00_파이썬_기초.ipynb)
      -  파이썬으로 할 수 있는 일
      -  파이썬 에 적합하지 않은 일
      -  스네이크 케이스와, 캐멀 케이스
      -  리스트와 배열의 차이
      -  실수와 정수의 난수 생성 with numpy
      -  `np.where(a, b, c)`
  - [01_선형과_비선형.ipynb](/code/00_basics/01_선형과_비선형.ipynb)
  - [02_도함수_계산.ipynb](/code/00_basics/02_도함수_계산.ipynb)
      -  Gradient
  - [03_통계.ipynb](/code/00_basics/03_통계.ipynb)
      -  통계와 데이터 분석의 기초 개념
      -  정규 분포 (Normal Distribution)
      -  표준화 (Standardization)
      -  가설(Hypothesis) 검정
      -  만약 정규분포를 따르지 않을 때
      -  파이썬 코드
  - [04_벡터화.ipynb](/code/00_basics/04_벡터화.ipynb)
      -  벡터라이제이션
      -  코사인 거리
      -  CountVectorizer, TfidfVectorizer
  - [05_토큰화.ipynb](/code/00_basics/05_토큰화.ipynb)
      -  토크나이제이션 (토큰화 : Tokenizaiton)
      -  IMDB를 이용한 영화평
  - [06_공분산과_상관계수.ipynb](/code/00_basics/06_공분산과_상관계수.ipynb)
      -  통계적 수치
      -  공분산
      -  표준 편차 ( std : standard deviation )
      -  Correlation
  - [07_data_유형.ipynb](/code/00_basics/07_data_유형.ipynb)
      -  정량적 데이터 (Quantitative Data)
      -  질적 데이터 (Qualitative Data)
      -  시계열 및 공간 데이터 (Temporal and Spatial Data)
      -  미디어 및 센서 데이터 (Media and Sensor Data)
      -  구조 및 관계 데이터 (Structured and Relational Data)
      -  행동 및 활동 데이터 (Behavioral and Activity Data)
      -  부가 및 설명 데이터 (Supplementary and Descriptive Data)
      -  복합 및 실시간 데이터 (Composite and Real-time Data)
      -  이상 및 특이 데이터 (Anomalous and Atypical Data)
      -  임베디드 및 트랜잭션 데이터 (Embedded and Transactional Data)
      -  횡단면 및 종단 데이터 (Cross-sectional and Longitudinal Data)
  - [08_Confusion_Matrix.ipynb](/code/00_basics/08_Confusion_Matrix.ipynb)
      -  Precision 과 recall 을 봐야한다.
- 01_machinelearing
  - [01_상관과_회귀.ipynb](/code/01_machinelearing/01_상관과_회귀.ipynb)
      -  상관계수 (Correlation Coefficient)
      -  회귀 계수 (Regression Coefficient)
      -  상관과 회귀의 차이
      -  결정 계수 $R^2$
      -  오차의 종류
      -  예시
  - [02_교차검증.ipynb](/code/01_machinelearing/02_교차검증.ipynb)
      -  교차검증(cross validation)
  - [03_GridSearch.ipynb](/code/01_machinelearing/03_GridSearch.ipynb)
      -  그리드 서치(Grid Search)
      -  RandomizedSearchCV
  - [04_Feature_Importance.ipynb](/code/01_machinelearing/04_Feature_Importance.ipynb)
      -  feature importance
      -  랜덤 포레스트 및 디시전 트리를 이용해서 정확도 구해보기
      -  랜덤 포레스트에서 가장 중요한 피쳐부터 하나씩 추가해서 비교해 보기
  - [05_PCA_주성분분석.ipynb](/code/01_machinelearing/05_PCA_주성분분석.ipynb)
      -  주성분 분석 (Principal Component Analysis, PCA)
      -  스케일링과 pca
      -  Breast Cancer dataset
      -  데이터 스케일링
      -  PCA
      -  Digit dataset
      -  PCA
  - [06_knn.ipynb](/code/01_machinelearing/06_knn.ipynb)
      -  [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) (K-Neighbors-Classifier)
      -  knn 파라미터 (scikit-learn)
      -  KNN에서의 최적의 K값 찾기
      -  적당한 k값 구하기 with 교차 검증
  - [07_decision_tree.ipynb](/code/01_machinelearing/07_decision_tree.ipynb)
      -  Decision Tree (DecisionTreeClassifier)
      -  불순도
      -  분류 vs 회귀 : DecisionTreeClassifier vs DecisionTreeRegressor
  - [08_SVM.ipynb](/code/01_machinelearing/08_SVM.ipynb)
      -  SVM (Support Vector Machine)
      -  주요 키워드
  - [09_ensemble.ipynb](/code/01_machinelearing/09_ensemble.ipynb)
      -  Ensemble
      -  Bagging (Bootstrap Aggragating)
      -  Bagging : Random Forest
      -  Bagging : Random Forest (Grid Search)
      -  Boosting
      -  종류
      -  Boosting : AdaBoost
      -  Gradient Boosting
      -  Gradient Boosting : AdaBoost
      -  Gradient Boosting : XGBoost
  - [10_회귀_예시.ipynb](/code/01_machinelearing/10_회귀_예시.ipynb)
      -  데이터 로드
      -  결측확인 (missingno as msno)
      -  데이터 스플릿
      -  상관 계수
      -  데이터 스케일링
      -  Logistic Regression
      -  SVR
      -  K-NN Regressor
      -  Decision Tree Regressor
      -  Random Forest Regressor
      -  XGBoost Regressor
      -  오답 확인
  - [11_분류_예시.ipynb](/code/01_machinelearing/11_분류_예시.ipynb)
      -  데이터 로드
      -  결측확인 (missingno as msno)
      -  데이터 스플릿
      -  상관 계수
      -  데이터 스케일링
      -  Logistic Regression(분류에 이용)
      -  SVC
      -  K-NN Classifier
      -  Decision Tree Classifier
      -  Random Forest Classifier
      -  XGBoost Classifier
      -  오답 확인
  - [12_Scaling.ipynb](/code/01_machinelearing/12_Scaling.ipynb)
- 02_DeepLearning
  - [01_deep_learning.ipynb](/code/02_DeepLearning/01_deep_learning.ipynb)
      -  딥러닝
      -  딥러닝 종류
      -  딥러닝의 구조
      -  단층 퍼셉트론 및 다층 퍼셉트론
      -  인공 신경망의 프로세스
      -  선형 회귀 모형과 신경망 모델의 차이
      -  출력 노드의 수
  - [02_활성화_함수.ipynb](/code/02_DeepLearning/02_활성화_함수.ipynb)
      -  활성화 함수
      -  **Sigmoid (시그모이드) 함수**:
      -  **Tanh (하이퍼볼릭 탄젠트) 함수**:
      -  **ReLU (Rectified Linear Unit) 함수**:
      -  **Leaky ReLU 함수**:
      -  **ELU (Exponential Linear Unit) 함수**:
      -  **SoftPlus 함수**:
      -  **GeLU (Gaussian Error Linear Unit) 함수**:
  - [03_비용함수.ipynb](/code/02_DeepLearning/03_비용함수.ipynb)
      -  비용함수
      -  회귀 문제
      -  분류 문제
      -  비용함수 예제 : 회귀 문제
      -  비용함수 예제 : 분류 문제
  - [04_역전파.ipynb](/code/02_DeepLearning/04_역전파.ipynb)
      -  역전파 (Back Propagation)
      -  체인 룰
  - [05_옵티마이저.ipynb](/code/02_DeepLearning/05_옵티마이저.ipynb)
      -  Optimizer (수치 최적화 알고리즘)
      -  기본 이론 : 경사 하강법(Gradient Descent)
      -  탐색 방향 기반 알고리즘
      -  학습률 기반 알고리즘
      -  다양한 옵티마이저 예제(torch)
  - [06_다양한_데이터.ipynb](/code/02_DeepLearning/06_다양한_데이터.ipynb)
      -  sklearn datasets
      -  tensorflow datasets
      -  seaborn datasets
      -  torchvision datasets
      -  cancer 데이터
      -  Digits 데이터
      -  MNIST 데이터
      -  Iris dataset
      -  wine 품질(quality) 데이터
      -  다중 분류
  - [07_다양한_기법들.ipynb](/code/02_DeepLearning/07_다양한_기법들.ipynb)
      -  딥러닝 간단한 층 쌓기
      -  콜백
      -  EarlyStopping
      -  결과 확인
      -  과적합 시작점 확인
  - [08_다양한_문제들.ipynb](/code/02_DeepLearning/08_다양한_문제들.ipynb)
      -  경사 소실 (gradient vanishing)
      -  데드 렐루 문제
      -  과적합 (Overfitting)
      -  과적합 해결책
      -  초기 가중치 문제(Weight Initialization Problem)
  - [09_CNN.ipynb](/code/02_DeepLearning/09_CNN.ipynb)
      -  CNN (Convolutional Neural Networks)
      -  공간 정보 추출(Spatial)
      -  패딩
      -  풀링
      -  CNN의 학습 과정
      -  CNN의 키워드
      -  Cifar10 데이터
      -  mnist 데이터
  - [10_RNN.ipynb](/code/02_DeepLearning/10_RNN.ipynb)
      -  RNN(Reccurent Neural Networks)
      -  시퀀스 데이터
      -  RNN과 FNN 모델의 차이점
      -  RNN을 사용한 텍스트 분석
      -  시작 토큰 및 종료 토큰
      -  RNN 순서
      -  Return Sequence True 파라미터
      -  imdb data 실습(return_seq_true with concat)
      -  imdb data 실습(return_seq_true with mean)
  - [11_LSTM.ipynb](/code/02_DeepLearning/11_LSTM.ipynb)
      -  LSTM
      -  기억 셀 ($C_t$)
      -  기억 셀의 업데이트
      -  기억 셀(Memory Cell)
      -  Gate
      -  GRU(Gated Recurrent Unit)
      -  imdb 실습
      -  bidirectional 및 stacked bidirectional
  - [12_AutoEncoder.ipynb](/code/02_DeepLearning/12_AutoEncoder.ipynb)
      -  AutoEncoder
      -  Encoder
      -  Latent Space(Code or z)
      -  Decoder
      -  Loss
      -  AE 응용
      -  Mnist data를 이용한 AE
      -  AE 를 이용한 데이터 증강
- 03_DeepLearning_기법들
  - [00_다양한_딥러닝_기법.ipynb](/code/03_DeepLearning_기법들/00_다양한_딥러닝_기법.ipynb)
      -  SoftMax 함수
      -  BatchNormalization
      -  EarlyStopping
      -  Dropout
      -  Learning Rate Scheduler
      -  Data Augmentation
      -  L2 Regularization
      -  Model Checkpoint
      -  Early Learning Rate Scheduler
      -  Gradient Clipping
      -  Transfer Learning(전이 학습)
      -  Attention Mechanism(어텐션)
  - [01_ReceptiveField.ipynb](/code/03_DeepLearning_기법들/01_ReceptiveField.ipynb)
      -  수용 영역
- 04_time_series
  - [01_time_series.ipynb](/code/04_time_series/01_time_series.ipynb)
      -  시계열 데이터
      -  다변량과 단변량
      -  시간 종속성 Time Dependence
      -  시계열 특성
      -  **정상성**
  - [02_ARIMA_분석.ipynb](/code/04_time_series/02_ARIMA_분석.ipynb)
      -  ARIMA
      -  자기회귀 AR(auto regressive)
      -  차분 (**integrated**)
      -  이동 평균(Moving Average) - q
      -  자기 상관성 확인 (Autocorrelation)
      -  Augmented Dickey-Fuller 를 이용한 정상성 검정
      -  차분에 대한 ADF
      -  statsmodels
  - [03_rag_feature.ipynb](/code/04_time_series/03_rag_feature.ipynb)
      -  Lag Feature (지연 피쳐)
      -  이동평균선
      -  regplot(회귀선)
  - [05_trend.ipynb](/code/04_time_series/05_trend.ipynb)
      -  다항 특성
      -  Moving Average Plots
      -  트렌드 예측 (Trend, 추세)
      -  잔차 모델링
      -  성능 평가
      -  지연 피쳐 및 이동 평균 생성
      -  모델 시각화
      -  example
      -  함수정의
      -  전처리
  - [06_Cycles.ipynb](/code/04_time_series/06_Cycles.ipynb)
      -  Serial Dependence
      -  Cycles(Serial Dependence 을 나타내는 일반적인 방법)
      -  Lagged Series and Lag plots
  - [07_seasonal.ipynb](/code/04_time_series/07_seasonal.ipynb)
      -  Seasonality
      -  계절성 확인
      -  주가 예측 모델 및 시각화
      -  지수 변화율 분석 및 주기성 시각화
      -  계절성 플롯(seasonal_plot)과 주기도(Periodogram)
      -  `scipy.signal.periodogram`
      -  주가 예측모델 (주기학습)
  - [08_Hybrid_model.ipynb](/code/04_time_series/08_Hybrid_model.ipynb)
      -  Hybrid Model
      -  **Components and Residuals**
      -  Hybrid Forecasting with Residuals(잔차를 사용한 복합 예측)
      -  Hybrids 알고리즘 디자인하기
      -  피쳐 변환 알고리즘 1 : Linear Regression
      -  타겟 변환 알고리즘 2 : Tree 모델 종류.
      -  하이브리드 모델 (Linear Regression + DecisionTreeRegressor 의 잔차 )
      -  트랜드(다항특성)를 고려하여 잔차와 함께 학습하기
  - [09_Forecast_stratagy.ipynb](/code/04_time_series/09_Forecast_stratagy.ipynb)
      -  예측 모델 정의하기
      -  예측 기원 (forecast origin)
      -  **forecast horizon**
      -  용어정리
      -  멀티 스텝 예측 전략!
- 05_sequence
  - [01_Sequence.ipynb](/code/05_sequence/01_Sequence.ipynb)
      -  크롤링
      -  불러오기
      -  가격 데이터 타입 변경
      -  날짜 데이터 타입 변경
      -  결측 확인
      -  데이터 정보 확인(고윳값)
      -  상관관계 분석
      -  분포 확인
      -  다변량 분석 - 산점도 행렬
  - [02_Sequence_모델링.ipynb](/code/05_sequence/02_Sequence_모델링.ipynb)
      -  Sequence 모델링
      -  전처리
      -  모델 설계 밑 구축
      -  예측 및 평가
  - [03_word_embed_cluster.ipynb](/code/05_sequence/03_word_embed_cluster.ipynb)
      -  전처리
      -  **Tf-idf** <br>
      -  **Word2Vec**
      -  **FastText**
- 06_visualization
  - [00_sns_시각화코드.ipynb](/code/06_visualization/00_sns_시각화코드.ipynb)
      -  여유 있을 때 배우면 좋은 것
      -  카토 그램
      -  Matplotlib 시각화 all in one
      -  라인 그래프
      -  산점도
      -  바 그래프
      -  히스토그램
      -  육각 히스토그램(hexbin)
      -  박스, 바이올린 플롯
      -  색상 선택하기
      -  파일저장
  - [01_PCA_시각화.ipynb](/code/06_visualization/01_PCA_시각화.ipynb)
      -  pca 시각화
  - [02_지도 시각화(folium).ipynb](/code/06_visualization/02_지도 시각화(folium).ipynb)
      -  데이터에서 지도 시각화 및 json 다루기
      -  지도 시각화
  - [03_boxplots.ipynb](/code/06_visualization/03_boxplots.ipynb)
      -  사분위 값
      -  IQR 계산하기
      -  boxplots
  - [04_cv2_그림그리기.ipynb](/code/06_visualization/04_cv2_그림그리기.ipynb)
      -  cv2 그림그리기
- 07_Pretrained_CNN
  - [00_img_prep.ipynb](/code/07_Pretrained_CNN/00_img_prep.ipynb)
      -  사진 불러오기
      -  이미지 가로 세로 맞추기 : 패딩
      -  이미지 가로 세로 맞추기 : 패딩
      -  워핑
  - [01_img_featuremap.ipynb](/code/07_Pretrained_CNN/01_img_featuremap.ipynb)
      -  특성 맵 확인하기
  - [02_LeNet.ipynb](/code/07_Pretrained_CNN/02_LeNet.ipynb)
      -  LeNet 구조
      -  CNN 의 1 by 1 필터
      -  lenet 실습
  - [03_AlexNet.ipynb](/code/07_Pretrained_CNN/03_AlexNet.ipynb)
      -  alexnet
  - [04_VGG16.ipynb](/code/07_Pretrained_CNN/04_VGG16.ipynb)
      -  feature map
      -  사전 학습 모형을 통한 이미지 분류 - VGG16
  - [05_inception(ggl)Net.ipynb](/code/07_Pretrained_CNN/05_inception(ggl)Net.ipynb)
      -  Inception Net
      -  Inception Net 구조
      -  인셉션 모듈 구조 확인
      -  inception net + 머신러닝 분류 알고리즘 적용
  - [06_ResNet.ipynb](/code/07_Pretrained_CNN/06_ResNet.ipynb)
      -  Residual Block
      -  Skip Connection (입력값 을 출력에 더하여 전달) 의 효과
      -  ResNet 모델 학습 과정
      -  ResNet 프리 트레인드 모델 이용하기
  - [07_mobile_Net.ipynb](/code/07_Pretrained_CNN/07_mobile_Net.ipynb)
      -  Mobile Net
      -  Depthwise Separable Convolution
      -  PointWise Convolution
      -  MoblieNet Pretrained Model
  - [08_DenseNet.ipynb](/code/07_Pretrained_CNN/08_DenseNet.ipynb)
      -  DenseNet
  - [09_EfficientNet.ipynb](/code/07_Pretrained_CNN/09_EfficientNet.ipynb)
      -  EfficientNet
  - [10_cnn_transfer_learning.ipynb](/code/07_Pretrained_CNN/10_cnn_transfer_learning.ipynb)
      -  Pretrained 모델 with 머신러닝
      -  분류 알고리즘 연결 및 예측
      -  파인 튜닝
- 08_pretrained_RNN
  - [01_time_rnn.ipynb](/code/08_pretrained_RNN/01_time_rnn.ipynb)
      -  ARIMA 모델(pmdarima)
      -  ARIMA
      -  RNN 모델
      -  LSTM 모델
  - [02_LSTM모델_설계.ipynb](/code/08_pretrained_RNN/02_LSTM모델_설계.ipynb)
      -  LSTM 모델 설계
  - [03_**Transformer**.ipynb](/code/08_pretrained_RNN/03_**Transformer**.ipynb)
      -  **Transformer**
      -  Transformer 전체적인 구성
      -  Positional Encoding layer
      -  Self Attention layer
      -  Encoder Decoder Attention
      -  실습 imdb
  - [04_BERTopic_En.ipynb](/code/08_pretrained_RNN/04_BERTopic_En.ipynb)
      -  Bert (Bidirectional Encoder Representations from Transformers)
      -  **BERTopic**
- 09_Object_Detection
  - [00_ReceptiveField.ipynb](/code/09_Object_Detection/00_ReceptiveField.ipynb)
      -  수용영역
      -  수용영역별 피쳐 크기
  - [01_Yolo.ipynb](/code/09_Object_Detection/01_Yolo.ipynb)
      -  YOLO (You Only Look Once)
      -  버전 별 비교
      -  YOLO with Robotics
  - [02_SSD.ipynb](/code/09_Object_Detection/02_SSD.ipynb)
      -  SSD (Single Shot MultiBox Detector)
      -  특징
      -  boxes
      -  학습 프로세스 순서
      -  용어정리
      -  mAP (Mean Average Precision)
      -  None Max Suppression
      -  SSD example
  - [05_Semantic_Upsampling_Transposed.ipynb](/code/09_Object_Detection/05_Semantic_Upsampling_Transposed.ipynb)
  - [06_Semantic_Segmentation.ipynb](/code/09_Object_Detection/06_Semantic_Segmentation.ipynb)
  - [07_Semantic_Performance.ipynb](/code/09_Object_Detection/07_Semantic_Performance.ipynb)
  - [08_OD_RCNN_Offset.ipynb](/code/09_Object_Detection/08_OD_RCNN_Offset.ipynb)
- 10_ChatGPT_API
  - [00_ChatGPT_API_Chatbot.ipynb](/code/10_ChatGPT_API/00_ChatGPT_API_Chatbot.ipynb)
  - [01_ChatGPT_API.ipynb](/code/10_ChatGPT_API/01_ChatGPT_API.ipynb)
- 11_Private_chat
  - [00_GPT_2.ipynb](/code/11_Private_chat/00_GPT_2.ipynb)
  - [01_llama_3.ipynb](/code/11_Private_chat/01_llama_3.ipynb)
      -  Llama 3
- 12_모델_응용
  - [00_NER.ipynb](/code/12_모델_응용/00_NER.ipynb)
      -  NER 모델
  - [01_CNN_for_Text.ipynb](/code/12_모델_응용/01_CNN_for_Text.ipynb)
  - [02_VQA.ipynb](/code/12_모델_응용/02_VQA.ipynb)
      -  VQA :  <br>
- 13.발표
  - [01_딥러닝에서_배치_크기의_역할.ipynb](/code/13.발표/01_딥러닝에서_배치_크기의_역할.ipynb)
      -  배치 사이즈란?
      -  배치 사이즈별 학습
      -  결론:
      -  배치 사이즈 선택 요령
  - [02_텐서_자료형.ipynb](/code/13.발표/02_텐서_자료형.ipynb)
      -  Tensor 자료형
      -  텐서 차원별 명명법
      -  파이 토치 에서의 사용
  - [03_옵티마이저_비교.ipynb](/code/13.발표/03_옵티마이저_비교.ipynb)
      -  옵티마이저 비교
  - [04_인공지능의_편향성과_차별.ipynb](/code/13.발표/04_인공지능의_편향성과_차별.ipynb)
      -  인공지능의 편향성과 차별문제
      -  종류
      -  원인
      -  Ai 차별
  - [05_인공지능의_창의성과_저작권.ipynb](/code/13.발표/05_인공지능의_창의성과_저작권.ipynb)
      -  창의성
      -  논점
      -  창의성 시험
      -  AI 드론
      -  저작권
      -  Nara AI Film
      -  결론
  - [06_딥러닝_모델의_해석가능성.ipynb](/code/13.발표/06_딥러닝_모델의_해석가능성.ipynb)
      -  딥러닝 모델의 해석 가능성
      -  해석
  - [08_데이터_활용과_개인정보_보호.ipynb](/code/13.발표/08_데이터_활용과_개인정보_보호.ipynb)
      -  데이터 활용과 개인정보 보호
      -  빅데이터란
      -  데이터 활용
      -  개인정보 보호
      -  개인정보 보호 중심 설계 :
  - [09_활성화_함수.ipynb](/code/13.발표/09_활성화_함수.ipynb)
      -  활성화 함수
      -  의미와 역할
      -  종류
      -  선택 기준
  - [10_인공지능의_윤리적_고려사항.ipynb](/code/13.발표/10_인공지능의_윤리적_고려사항.ipynb)
  - [11_transformer.ipynb](/code/13.발표/11_transformer.ipynb)
      -  Transformer
      -  어텐션 시각화
  - [12_alex_net.ipynb](/code/13.발표/12_alex_net.ipynb)
      -  Alex Net
      -  구조
  - [7.02.VQA.ipynb](/code/13.발표/7.02.VQA.ipynb)
- 14.Generative_Deep_Learning_2nd_공부내용
  - [00_2.3_다층_퍼셉트론_구현.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/00_2.3_다층_퍼셉트론_구현.ipynb)
  - [01_2.4_합성곱_신경망.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/01_2.4_합성곱_신경망.ipynb)
  - [02_3._VAE_in_book_extra(faceA).ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/02_3._VAE_in_book_extra(faceA).ipynb)
  - [03_3.1_VAE.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/03_3.1_VAE.ipynb)
      -  학습 출력
  - [04_3.2_multivariable_normal_distribution.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/04_3.2_multivariable_normal_distribution.ipynb)
  - [05_3.2_VAE_in_book.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/05_3.2_VAE_in_book.ipynb)
  - [06_3.오토인코더.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/06_3.오토인코더.ipynb)
  - [07_GAN.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/07_GAN.ipynb)
  - [08_vae_done.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/08_vae_done.ipynb)
  - [09_WGAN_GP-gptmade.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/09_WGAN_GP-gptmade.ipynb)
  - [10_WGAN_GP.ipynb](/code/14.Generative_Deep_Learning_2nd_공부내용/10_WGAN_GP.ipynb)


</details>


### AI study
- [Notion 링크](https://royal-offer-53a.notion.site/KDT-2024-05-2024-09-10bf678f80468069b4e1e2f0a631131a?pvs=4)
- [전체 파일 구조](markdown/00_files.md)

### 예제 데이터 셋
- [heart](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

### 목차
- [1. 기초](markdown/01_basics.md)
- [2. 머신러닝](markdown/02_ml.md)
- [3. 샘플링](markdown/03_sampling.md)
- [4. 딥러닝](markdown/04_dl.md)
- [5. 모델 평가](markdown/05_metrics.md)
- [6. 분류 및 회귀 문제](markdown/06_diversity.markdown)
- [7. 시계열](markdown/07_time_series.md)

### 프로젝트
- [project1](https://github.com/tommyjin2894/KDT_project1)
- [project2](https://github.com/tommyjin2894/KDT_project2)
- project3
    - [서비스](https://github.com/tommyjin2894/project_3_service)
    - [훈련 및 결과](https://github.com/tommyjin2894/project_3_git)

### 참고 링크
[roboflow](https://roboflow.com/) <br>
[ultraytics](https://docs.ultralytics.com/integrations/roboflow/) <br>
[Learn open cv](https://learnopencv.com/)  <br>
[supervisely](https://supervisely.com/) <br>
[superb ai](https://superb-ai.com/ko) <br>
[label studio](https://labelstud.io/) -> 오디오에서 감성 분석 가능 <br>

### segmentation tool
[Label Studio](https://labelstud.io/guide/) <br>
[Label Me](https://github.com/labelmeai/labelme) <br>
[Anylabeling](https://github.com/vietanhdev/anylabeling) <br>
[X-Anylabeling](https://github.com/CVHub520/X-AnyLabeling) <br>
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

## AI study
- ### [Notion 전체 내용 링크](https://royal-offer-53a.notion.site/KDT-2024-05-2024-09-134f678f80468007b265d54d5952da14)
- ### [Notion 정리 내용 링크](https://royal-offer-53a.notion.site/KDT-2024-05-2024-09-10bf678f80468069b4e1e2f0a631131a)

## 프로 젝트 정리 Git
- [project1](https://github.com/tommyjin2894/KDT_project1)
- [project2](https://github.com/tommyjin2894/KDT_project2)
- project3
    - [서비스](https://github.com/tommyjin2894/project_3_service)
    - [훈련 및 결과](https://github.com/tommyjin2894/project_3_git)

## 아래 내용 요약 목차
- [데이터 전처리 및 분석 단계](https://github.com/tommyjin2894/ai_study?tab=readme-ov-file#%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%EB%B0%8F-%EB%B6%84%EC%84%9D-%EB%8B%A8%EA%B3%84)
- [마이닝 알고리즘](https://github.com/tommyjin2894/ai_study?tab=readme-ov-file#%EB%A7%88%EC%9D%B4%EB%8B%9D-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-1)
- [샘플링 기법](https://github.com/tommyjin2894/ai_study?tab=readme-ov-file#%EC%83%98%ED%94%8C%EB%A7%81-%EA%B8%B0%EB%B2%95)
- [딥러닝](https://github.com/tommyjin2894/ai_study?tab=readme-ov-file#%EB%94%A5%EB%9F%AC%EB%8B%9D)
- [모델 평가](https://github.com/tommyjin2894/ai_study?tab=readme-ov-file#%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80)
- [분류 문제](https://github.com/tommyjin2894/ai_study?tab=readme-ov-file#%EB%B6%84%EB%A5%98-%EB%AC%B8%EC%A0%9C)
- [회귀 문제](https://github.com/tommyjin2894/ai_study?tab=readme-ov-file#%ED%9A%8C%EA%B7%80-%EB%AC%B8%EC%A0%9C)
- [시계열 데이터](https://github.com/tommyjin2894/ai_study?tab=readme-ov-file#%EC%8B%9C%EA%B3%84%EC%97%B4-%EB%8D%B0%EC%9D%B4%ED%84%B0)
## 내용 요약 정리


### 데이터 전처리 및 분석 단계
1. **데이터 탐색**:
   - 다양한 데이터셋(예: 부산 집값, Iris, HR, Titanic 데이터)을 분석할 때, 데이터의 구조와 특성을 먼저 이해.
   - 데이터의 결측 값, 이상치, 변수의 종류 등을 파악.

2. **결측치 처리**:
   - **Numerical 데이터**: 
     - 가장 간단한 방법은 평균이나 중앙값으로 결측치를 대체하는 것.
     - 다른 특성과의 관계를 분석하여 더 적합한 값을 찾아 채울 수도 있다.
   - **Categorical 데이터**:
     - 새로운 카테고리를 만들어 결측치를 대체할 수 있다.
     - 다른 특성과의 관계를 고려하여 결측치를 처리할 수 있다.

3. **Feature Engineering**:
   - 새로운 특성을 생성하여 데이터를 변환하거나 차원을 줄일 수 있다.
   - **Categorical 변수 처리**:
     - **Nominal (순서 없음)**: One-hot encoding을 사용하거나 데이터 타입을 `categorical`로 지정한다.
     - **Ordinal (순서 있음)**: 이러한 변수는 숫자(예: 자연수)로 변환하여 순서를 반영한다.

4. **이상치 처리**:
   - 각 특성에 대한 깊은 이해를 바탕으로 이상치인지 판단한다.
   - 다른 특성과의 관계를 분석하여 데이터가 실제로 이상한지 평가한다.

5. **데이터 시각화**:
   - Matplotlib, Seaborn 등을 사용해 데이터를 시각화.
   - 데이터의 패턴, 분포, 관계 등을 이해하기 위하여 적절한 그래프와 색상 등을 선택.
   - $x$축, $y$축 및 제목을 명확히 표시하고, 필요한 경우 추가 설명을 삽입.

6. **상관관계 분석**:
   - 두 변수 간의 관계를 수치화.
   - **수치형 데이터**: 피어슨 상관계수를 사용.
   - **순서형 데이터**: 스피어만 상관계수를 사용.

7. **스케일링**:
   - 데이터의 분포 범위를 균일하게 맞춤.
   - **방법**: Min-Max 스케일링, Standard 스케일링, Robust 스케일링 등.

8. **차원 축소**:
   - 차원이 높을 경우, 과적합을 방지하기 위해 차원을 축소.
   - **기법**: PCA, t-SNE 등. 모델 학습 전 정보 보존을 위해 차원을 줄이거나 데이터를 2D/3D로 시각화할 때 사용.

**주의사항**: 테스트 데이터의 전처리는 훈련 데이터에서 얻은 정보를 기준으로만 이루어져야 함. 이는 데이터 누수를 방지하기 위한 중요한 원칙.

### 마이닝 알고리즘
- 머신러닝 모델(지도 학습)   
    |모델|이름|설명|
    |---|---|---|
    |분류|Decision Tree|트리구조로 데이터를 분류, 조건 분기|
    |-|Random Forest|앙상블 기법중 baseline Bagging 중 하나 <br> 여러개의 DT로 구성|
    |-|KNN|가까운 K 개의 데이터를 기반으로 결정 <br> baseline L1 및 L2 거리|
    |-|SVM|클래스 간의 경계를 최대화하여 초평면을 찾는다.|
    |회귀|Linear Regression|선형 관계 모델링|
    |-|Logistic Regression|이진 분류를 위한 회귀 분석 기법,<br> baseline 확률로 출력값을 변환|
    |인공 신경망|NN|여러층의 뉴런|
    |기타|AdaBoost|약한 학습기 x N = 강한 학습기|
    |-|XGBoost|Gradient Boosting Machines 의 효율적이고 강력하게 개선|


- 비지도 학습
    |종류|이름|설명|
    |-|-|-|
    |클러스터링|k-means|비슷한 포인트를 가깝게 위치|
    |-|계층적 클러스터링|트리 구조로 조직화|
    |연관 규칙|Apriori 알고리즘|자주 발생 하는 연관 집합|
    |-|FP-Growth|Apriori 보다 효율적인 |
    |차원 축소|PCA|데이터를 압축, 저차원으로|
    |-|t-SNE|2~3 차원으로 시각화, 비슷한 데이터 그룹화|
        baseline 클러스터링 : 유사도 기준 L1(manhatten), L2(Euclidean) 으로 군집화


- 다양한 기법
    |종류|이름|설명|
    |---|---|---|
    |기법|K-fold 교차 검증|점수 평균|
    |-|Grid search|모든 경우의수를 본다|
    |-|Randomized search|랜덤한 경우의수를 본다|
    |앙상블|bagging<br> (bootstrap aggregating)|1. baseline N 개의 샘플을 뽑기<br>->집어넣고 N 개의 샘플을 뽑는다. <br> 2. 중복이 생길 수 있음|
    |-|Boosting|약한 학습기 X N = 강한 학습기 <br>AdaBoost, XGBoost, Lgith GBM, Cat     Boost 등|
    |-|Stacking|여러 개의 기초모델의 예측<br>종합하여 새로운 메타모델 생성|


- K-fold 교차 검증
    - 훈련 데이터를 k 개로 분할해 번갈아 가면서 훈련 평가
      |학습 모델|데이터1|데이터2|데이터3|데이터4|데이터5|
      | ---| --- | --- | --- | --- | --- |
      | 학습 1 | train | train | train | train | test |
      | 학습 2 | train | train | train | test | train |
      | 학습 3 | train | train | test | train | train |
      | 학습 4 | train | test | train | train | train |
      | 학습 5 | test | train | train | train | train |

### 샘플링 기법

- 데이터 불균형 성능평가
    - 데이터 불균형 판단기준 : 30%
        - 데이터 불 균형시 다양한 샘플링 및 다양한 metrics 설정
          - 정확도, 정밀도, 재현율, F1 스코어, AUC-ROC,


-  샘플링 기법
    - 임의 추출
    - 계통 추출 (공장)
    - 층화 추출 (나이 및 성별별 추출)
    - 군집 추출 (전국 -> 서울)
    - 다 단계 추출 (전국 -> 서울 -> 남성)
    - 비 확률적 추출 (임의 추출)
  
주의 : 편향적인 데이터가 되지 않게

### 딥러닝
- 종류
    |이름|특징|구조|
    |-|-|-|
    |단층 퍼셉트론|XOR 문제와 같은 비선형 문제를 해결할 수 없음<br>역전파는 존재하지 않았다|단층 구조|
    |다층 퍼셉트론 (MLP)|범용 근사기:<br>충분히 크고 복잡한 어떠한 문제라도 이론적으로 학습 가능|입력층, 은닉층(다수), 출력층|
    |CNN (Convolutional Neural Networks)|공간적 계층 구조를 통해 이미지 및 비디오 데이터의 특징 추출에 탁월함|Convolutional layer, Pooling layer, Fully Connected layer|
    |RNN (Recurrent Neural Networks)|시퀀스 데이터 처리에 강점,<br>시계열 및 자연어 처리에 유용|Recurrent 구조, Hidden state vector|
    |LSTM (Long Short-Term Memory)|장기 의존성 문제를 해결하기 위해 설계됨,<br>Forget-Input-Output Gate 및 Cell state(기억 셀)를 사용|LSTM Cell 구조, Gates (Forget, Input, Output), Cell state|
    |GRU (Gated Recurrent Unit)|LSTM의 경량화된 변형,<br>더 간단한 구조로 기억 셀 없이 Gate만 사용|GRU Cell 구조, Update Gate, Reset Gate|
    |AutoEncoder|데이터의 차원을 축소하고 재생성하여 데이터 압축 및 노이즈 제거,<br>특성 학습에 사용됨|Encoder -> Latent Space(z) -> Decoder|
    |Transformer|Attention 메커니즘을 사용하여 입력 시퀀스의 모든 요소를 동시적으로 처리,<br>장기 의존성 문제 해결|Self-Attention Mechanism, Encoder-Decoder 구조, Multi-Head Attention, Position-wise Feed-Forward Networks|
    |ResNet (Residual Networks)|Residual Block을 사용하여 매우 깊은 신경망을 학습,<br>Gradient Vanishing 문제 완화|Residual Block, Skip Connections, Convolutional Layers|
    |EfficientNet|모델의 크기와 계산 효율성을 조정하기 위한 Compound Scaling 사용,<br>높은 성능과 효율성 제공|EfficientNet Blocks, Compound Scaling, Swish Activation Function|
    |VAE (Variational Autoencoder)|잠재 공간의 확률 분포를 학습하여 새로운 샘플을 생성,<br>데이터의 확률적 특성을 모델링|Encoder, Latent Space (Probability Distribution), Decoder, Variational Objective|
    |GAN (Generative Adversarial Network)|생성자와 판별자 간의 경쟁을 통해 데이터 생성,<br>이미지 생성, 데이터 증강 등에 사용|Generator, Discriminator, Adversarial Training|

- 비용함수 및 손실함수
    - 손실 함수 : 데이터 포인트 하나에 대한 오차 함수
    - 비용 함수 : 전체 데이터에 대한 오차 함수

    |구분|이름|특징|구조|
    |-|-|-|-|
    |회귀 문제|단층 퍼셉트론|XOR 같은 비선형 문제에 대한 한계<br>역전파는 존재하지 않았다|단층 구조|
    |-|MSE|제곱, 이상치에 민감|$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$|
    |-|MAE|절대 값, 이상치에 둔감|$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \lvert y - \hat{y} \lvert$|
    |-|허브 손실|MSE + MAE|MSE + MAE 의 구조|
    |-|로그 코사인 유사도|이상치에 매우 강함|$\log - \cosh = \frac{1}{N} \sum^{N}_{i = 1} \log({\cosh (\hat{y}-y)})$|
    |분류 문제|Cross Entropy Error|이진 분류 : binary CEE<br>다중 분류 : Categorical CEE|$CEE = -\sum_{k=1}^i t_k\text{log}\hat{y}$|
    |-|힌지 손실|SVM 에서 사용<br>마진 오류 최소화||
    |-|제곱 힌지 손실|이상치의 민감||
    |-|포칼 손실|오답에 대한 가중치 부여||

- 활성화 함수
    |이름|공식|출력 범위
    |-|-|-|
    |Sigmoid|$\phi = \frac{1}{1+e^{-x}}$|0 ~ 1|
    |tanh|$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$|-1 ~ 1|
    |ReLU|$f(z) = max(0, z)$|$0 \leq f(x)$|
    |Leaky ReLU|$f(z) = max(\epsilon z, z)$|$0 \leq f(x)$|
    |ELU|$f(x) = x \space \text{if } x \geq 0$<br>$f(x) = \alpha (e^x - 1) \space \text{if } x < 0$|$0 \leq f(x)$|
    |SoftPlus|$f(z) =  \ln(1 + e^x)$|$0 \leq f(x)$|
    |GeLU|$0.5 \cdot x \cdot \left( 1 + \tanh \left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 \cdot x^3 \right) \right) \right)$|Free <br>ReLU 계열 그래프와 비슷|

- 옵티 마이저
    : 수치 최적화 알고리즘
    |이름|학습률|탐색 방향|알고리즘 기반|
    |-|-|-|-|
    |SGD|상수|기울기|탐색 방향
    |Momentum|상수|단기 누적 기울기|탐색 방향
    |AdaGrad|장기 파라미터 변화량과 반비례|기울기|학습 률
    |RMSProp|단기 파라미터 변화량과 반비례|기울기|학습 률
    |Adam|단기 파라미터 변화량과 반비례|단기 누적 Grad|학습 률

- 문제 및 완화법
    - 경사 소실 문제
        - ReLU 계열의 활성화 함수 사용 <br> (Dead ReLU 문제가 발생할 수 있음)
    - 과적합 문제
        |이름|내용|
        |-|-|
        |L1 규제|가중치의 절대값과 비례하는 비용 추가<br>가중치를 0으로 만들어 특성에 대한 영향 제거<br>(모델의 희소성 증가)|
        |L2 규제|가중치의 제곱에 비례하는 비용 추가<br>가중치의 값을 줄여 복잡성을 낮춘다<br>(가중치가 너무 커지는 것을 방지)<br>|
        |드롭 아웃|학습 과정 중 노드를 임의로 비활성|
        |Early Stop|더 이상 학습이 진행되지 않을떄 학습 중단|
        |데이터 증강|비슷한 데이터를 복제하여 학습 데이터로 만듬<br>테스트 할떄 증강 금지|

- 데이터 증강 기법
    - keras 변형 증강 : 케라스 내장 으로 각도 조절 및 크기 반전등을 이용하여 데이터를 증강
    - Auto Encoder 증강 : Auto Encoder 로 생성된 데이터를 이용한 증강

- 다양한 Pretraind 모델
    - CNN 기반
        |이름|내용|특징|레이어|
        |-|-|-|-|
        |LeNet|CNN 초기 모델|얀 르쿤에 의해 개발, 손글씨 인식에 사용|기본 CNN 구조 (Convolutional Layers, Pooling Layers)|
        |AlexNet|ReLU 활성화 함수, 데이터 증강, MaxPooling을 통한 벡터화, 드롭아웃, 다중 GPU 활용|ReLU 활용, 데이터 증강으로 성능 향상|Convolutional Layers, ReLU, MaxPooling, Dropout|
        |VGG-16|3x3 필터와 2x2 MaxPooling 활용, 구조 단순화, 규제 기법 적용|옥스포드 VGG 그룹에 의해 개발, 깊이 있는 네트워크|Convolutional Layers (3x3), MaxPooling (2x2), Fully Connected Layers|
        |InceptionNet<br>(Google Net)|Bottle neck 구조, Inception Module, Auxiliary classifier, Main classifier|Google에 의해 개발, 1x1 필터로 파라미터 수 감소|Inception Modules, 1x1, 3x3, 5x5 Convolutions, Pooling|
        |ResNet|Residual block을 통한 Skip Connection, 경사 소실 문제 완화|Microsoft에 의해 개발, VGG-19의 뼈대, Residual Blocks 사용|Residual Blocks, Skip Connections, Convolutional Layers|
        |MobileNet|Depthwise Separable Convolution, 각 채널별로 독립적인 연산 후 통합|Google의 Howard에 의해 개발, 성능 유지 및 속도 향상|Depthwise Separable Convolutions, 1x1 Convolutions|
        |DenseNet|Dense Block 구조, 모든 레이어의 input을 output에 Concat|ResNet과 비슷한 성능, Feature 재사용 증가|Dense Blocks, Convolutional Layers, Concatenation|
        |EfficientNet|최적의 Depth, Width, Resolution을 찾기 위한 Grid Search, 효율적인 모델 크기 및 성능|구글에 의해 개발, 모델 크기와 계산 효율성 최적화|Compound Scaling, Convolutional Layers, EfficientNet Blocks|

    - 자연어 처리 기반
        |이름|내용|특징|
        |-|-|-|
        |Transformer|Attention 메커니즘을 사용하여 입력 시퀀스의 모든 요소를 동시적으로 처리하며, 장기 의존성 문제를 해결하는 모델|Self-Attention, Multi-Head Attention, Encoder-Decoder 구조|
        |BERT (Bidirectional Encoder Representations from Transformers)|양방향 컨텍스트를 사용하여 자연어 이해 성능을 향상시킨 모델. Masked Language Modeling과 Next Sentence Prediction을 통해 사전 학습됨|Bidirectional Context, Pre-training and Fine-tuning, 다양한 NLP 작업에 활용|
        |GPT (Generative Pre-trained Transformer)|대규모 언어 모델로, 언어 생성과 번역을 포함한 다양한 NLP 작업에 강력한 성능을 발휘. Transformer 기반으로 대규모 데이터에서 사전 학습됨|Unidirectional Context, Language Modeling, Transfer Learning|



    - 객체 탐지 모델
        |Shots|이름|내용|특징|
        |-|-|-|-|
        |Two|R-CNN<br>(Regions with CNN features)|전통적인 객체 탐지 방법:<br>Selective Search로 영역을 제안-><br>CNN으로 피처 벡터로 변환-><br>분류 및 경계 상자를 예측|Two-stage detector,<br>Selective Search,<br>CNN-based feature extraction|
        |Two|Fast R-CNN|R-CNN의 개선, 전체 이미지에 대해 CNN을 한 번만 실행,<br>RoI Pooling로 각 제안 영역의 피처를 추출 분류 및 회귀|RoI Pooling,<br>End-to-end training,<br>Faster processing compared to R-CNN|
        |Two|Faster R-CNN|Region Proposal Network (RPN)과<br>Fast R-CNN을 결합|RPN for region proposals,<br>ROI Pooling|
        |One|YOLO<br>(You Only Look Once)|One-Shot. 빠른 속도와 높은 실시간 성능|Bounding box regression,<br>Class prediction|
        |One|SSD<br>(Single Shot MultiBox Detector)|다양한 크기 객체를 탐지<br>다양한 스케일의 특성을 활용|Multi-scale feature maps,<br>Default boxes|

        > RoI : Region of interest


### 모델 평가
  
  1. 정확도(Accuracy):
      - 일반적 평가 지표
      - 데이터가 균형적 일때
  
  2. 재현율(Recall), 정밀도(Precision), F1-score:
      - 데이터의 불균형 30% 이상일때
      - 재현율: 실제 양성 중 양성 비율 <br>
         : $$\frac{\text{TP}}{\text{FP} + {\text{FP}}}$$
      - 정밀도: 예측한 양성 중 실제 양성비율 <br>
         : $$\frac{\text{TP}}{\text{FP} + {\text{FN}}}$$
      - F1-score: 재현율과 정밀도의 조화 평균
  
  3. 혼동 행렬(Confusion Matrix):
      - 실제 값과 예측 값의 관계를 보여줌
      - 정확도, 재현율, 정밀도 지표
      - ROC 곡선 및 AUC(Area Under the Curve):
          - 이진 분류 시 평가 Metric
          - 임계값에 따른 True Positive Rate와 False Positive Rate를 나타냄
          - AUC 값이 1에 가까울수록 모델의 성능이 좋음, 0.5 는 넘어야 함
          - AUC = 0.5 -> 랜덤 분류기와 성능이 같다.
  
  4. R-squared(R2-Score):
  
      - 회귀 모델 평가에 사용되는 지표
      - 모델이 종속변수의 변동을 얼마나 잘 설명하는지 나타냄
      - 0에서 1 사이의 값을 가지며, 1에 가까울수록 모델 성능이 좋음

  5. 교차 검증(Cross-Validation):
  
      - k개의 폴드(fold)로 나누어 모델을 평가
      - 훈련 ,검증 데이터를 분리 후 일반화 성능 평가
      - 과적합을 방지하고, 안정성을 확인
  
  
  
  6. 도메인 지식 활용:
  
      - 데이터에 대한 도메인 이해 및 평가

### 분류 문제
|이름|내용|
|-|-|
|Mnist|손 글씨 분류|
|CIFAR|사진 대상 분류|
|텍스트, 표정, 감성|주로 시퀀스 context 해석 문제|
|일 대 다 분류|단계별로 하나씩 분류|

### 회귀 문제
|이름|내용|
|-|-|
|주택 가격 예측|가격 예측|
|주식 가격 예측|가격 예측|
|온도 예측|기상 데이터로 온도 예측|

### 시계열 데이터
- 시간 순서에 따라 발생하는 연속적인 데이터를 기반으로 미래를 예측하는 데이터.

- 다변량, 단변량
    - **다변량 (Multivariate)**: 여러 종속 변수(여러 개의 정답 라벨).
    - **다변수 (Multivariable)**: 여러 독립 변수(다양한 입력).
    - **단변량 (Univariate)**: 단일 정답 라벨을 가지는 데이터.

- 시간의 종속성 (Time Dependence)
    - 과거 데이터를 기반으로 미래를 예측하는 특성.
    - 불규칙성이 있으면 예측이 어려워짐.

- 시계열 종속성 (Serial Dependence)
    - 시계열 데이터가 이전 값에 의존하는 패턴. 이전 데이터가 현재와 미래의 데이터에 영향을 미치는 경우를 설명함.

- Cycles
    - 데이터가 주기적으로 반복되는 현상을 설명. 반드시 시간에 따라 반복되지 않아도 되며, 자연스럽게 반복되는 현상.

- 시계열 데이터의 주요 특성
    - **계절성 (Seasonality)**: 일일, 주간, 연간 등 주기적으로 변화하는 패턴.
    - **추세 (Trend)**: 장기적인 변화 경향을 나타냄. 이동 평균 플롯을 통해 추세를 시각화할 수 있음.

- 다항 특성 (Polynomial Features)
    - 시계열 데이터를 다항식으로 표현할 수 있음. 예를 들어, $degree$가 1일 때, 2일 때, n일 때의 다항식 모델이 각각 있음.
        - $y = w \times time + b$ (일차식)
        - $y = w_0 \times time^2 + w_1 \times time + b$ (이차식)
        - 다항식 모델로 예측 가능.

- 정상성 (Stationarity)
    - 시계열 분석이 용이하려면 데이터가 정상성을 가져야 함.
    - **정상성 판단 기준**:
        - 시각적으로: 상승 또는 하락이 지속되지 않음, 변동폭 일정.
        - 통계적으로: 평균과 분산이 일정하며, 공분산이 시간과 무관하게 유지됨.

    - **비정상적 데이터**는 변환을 통해 정상성을 가지게 한 후 분석을 진행해야 함.

> 시계열 → 정상성 검증 → 정상성을 띄도록 변환 → 정확도 향상

### 참고 링크
- [roboflow](https://roboflow.com/) <br>
- [ultraytics](https://docs.ultralytics.com/integrations/roboflow/) <br>
- [Learn open cv](https://learnopencv.com/)  <br>
- [supervisely](https://supervisely.com/) <br>
- [superb ai](https://superb-ai.com/ko) <br>
- [label studio](https://labelstud.io/) -> 오디오에서 감성 분석 가능 <br>

- segmentation tool
    - [Label Studio](https://labelstud.io/guide/) <br>
    - [Label Me](https://github.com/labelmeai/labelme) <br>
    - [Anylabeling](https://github.com/vietanhdev/anylabeling) <br>
    - [X-Anylabeling](https://github.com/CVHub520/X-AnyLabeling) <br>
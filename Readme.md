### 노션 및 파일 구조
- [Notion 링크](https://royal-offer-53a.notion.site/KDT-2024-05-2024-09-10bf678f80468069b4e1e2f0a631131a?pvs=4)

- [전체 파일 구조](mds/file_hirachy.md)

### 기본 시각화 코드
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
<!--------------------------------------->

### [파이썬 기본 코드 연습](mds/1_python_basic_codes.md)

### 데이터 불균형 성능평가
- 데이터 불균형 판단기준 : 30%
    - 데이터 불 균형시 다양한 샘플링 및 다양한 metrics 설정
      - 정확도, 정밀도, 재현율, F1 스코어, AUC-ROC,

### 회귀 성능 평가
- RMSE, MAE 등 

### 😊결과를 표로 잘 정리하기😊

<!--------------------------------------->


# 마이닝 알고리즘

### 내용 정리
머신러닝 모델(지도 학습)
    
|모델|이름|설명|
|---|---|---|
|분류|Decision Tree|트리구조로 데이터를 분류, 조건 분기|
|-|Random Forest|앙상블 기법중 baseline Bagging 중 하나 <br> 여러개의 DT로 구성|
|-|KNN|가까운 K 개의 데이터를 기반으로 결정 <br> baseline L1 및 L2 거리|
|-|SVM|클래스 간의 경계를 최대화하여 초평면을 찾는다.|
|회귀|Linear Regression|선형 관계 모델링|
|-|Logistic Regression|이진 분류를 위한 회귀 분석 기법,<br> baseline 확률로     출력값을 변환|
|인공 신경망|NN|여러층의 뉴런|
|기타|AdaBoost|약한 학습기 x N = 강한 학습기|
|-|XGBoost|Gradient Boosting Machines 의 효율적이고 강력하게 개선|


비지도 학습

 |종류|이름|설명|
 |-|-|-|
 |클러스터링|k-means|비슷한 포인트를 가깝게 위치|
 |-|계층적 클러스터링|트리 구조로 조직화|
 |연관 규칙|Apriori 알고리즘|자주 발생 하는 연관 집합|
 |-|FP-Growth|Apriori 보다 효율적인 |
 |차원 축소|PCA|데이터를 압축, 저차원으로|
 |-|t-SNE|2~3 차원으로 시각화, 비슷한 데이터 그룹화|

    baseline 클러스터링 : 유사도 기준 L1(manhatten), L2(Euclidean) 으로 군집화


다양한 기법

 |종류|이름|설명|
 |---|---|---|
 |기법|K-fold 교차 검증|점수 평균|
 |-|Grid search|모든 경우의수를 본다|
 |-|Randomized search|랜덤한 경우의수를 본다|
 |앙상블|bagging<br> (bootstrap aggregating)|1. baseline N 개의 샘플을 뽑기<br>->집어넣고 N 개의 샘플을 뽑는다. <br> 2. 중복이 생길 수 있음|
 |-|Boosting|약한 학습기 X N = 강한 학습기 <br>AdaBoost, XGBoost, Lgith GBM, Cat     Boost 등|
 |-|Stacking|여러 개의 기초모델의 예측<br>종합하여 새로운 메타모델 생성|


K-fold 교차 검증

    - 훈련 데이터를 k 개로 분할해 번갈아 가면서 훈련 평가
     |학습 모델|데이터1|데이터2|데이터3|데이터4|데이터5|
     | ---| --- | --- | --- | --- | --- |
     | 학습 1 | train | train | train | train | test |
     | 학습 2 | train | train | train | test | train |
     | 학습 3 | train | train | test | train | train |
     | 학습 4 | train | test | train | train | train |
     | 학습 5 | test | train | train | train | train |

    


### code
```
전처리, 트레인 테스트 데이터 분할, 마이닝 알고리즘, 교차 검증, PCA, 그리드 서치, 랜더마이즈드 서치
```

## 다양한 샘플링 기법

### 내용 정리
다양한 샘플링 기법 설명
  
### 샘플링 기법
- 임의 추출
- 계통 추출 (공장)
- 층화 추출 (나이 및 성별별 추출)
- 군집 추출 (전국 -> 서울)
- 다 단계 추출 (전국 -> 서울 -> 남성)
- 비 확률적 추출 (임의 추출)
  
주의 : 편향적인 데이터가 되지 않게
  
  

### codes
```
다양한 샘플링 기법 코드
```

# 딥러닝

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

### 비용함수 및 손실함수
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



### 활성화 함수

|이름|공식|출력 범위
|-|-|-|
|Sigmoid|$\phi = \frac{1}{1+e^{-x}}$|0 ~ 1|
|tanh|$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$|-1 ~ 1|
|ReLU|$f(z) = max(0, z)$|$0 \leq f(x)$|
|Leaky ReLU|$f(z) = max(\epsilon z, z)$|$0 \leq f(x)$|
|ELU|$f(x) = x \space \text{if } x \geq 0$<br>$f(x) = \alpha (e^x - 1) \space \text{if } x < 0$|$0 \leq f(x)$|
|SoftPlus|$f(z) =  \ln(1 + e^x)$|$0 \leq f(x)$|
|GeLU|$0.5 \cdot x \cdot \left( 1 + \tanh \left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 \cdot x^3 \right) \right) \right)$|Free <br>ReLU 계열 그래프와 비슷|

### 옵티 마이저
: 수치 최적화 알고리즘
|이름|학습률|탐색 방향|알고리즘 기반|
|-|-|-|-|
|SGD|상수|기울기|탐색 방향
|Momentum|상수|단기 누적 기울기|탐색 방향
|AdaGrad|장기 파라미터 변화량과 반비례|기울기|학습 률
|RMSProp|단기 파라미터 변화량과 반비례|기울기|학습 률
|Adam|단기 파라미터 변화량과 반비례|단기 누적 Grad|학습 률


### 문제 및 완화법
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

### 데이터 증강 기법

- keras 변형 증강 : 케라스 내장 으로 각도 조절 및 크기 반전등을 이용하여 데이터를 증강
- Auto Encoder 증강 : Auto Encoder 로 생성된 데이터를 이용한 증강

## 다양한 Pretraind 모델
### CNN 기반
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

### 자연어 처리 기반

|이름|내용|특징|
|-|-|-|
|Transformer|Attention 메커니즘을 사용하여 입력 시퀀스의 모든 요소를 동시적으로 처리하며, 장기 의존성 문제를 해결하는 모델|Self-Attention, Multi-Head Attention, Encoder-Decoder 구조|
|BERT (Bidirectional Encoder Representations from Transformers)|양방향 컨텍스트를 사용하여 자연어 이해 성능을 향상시킨 모델. Masked Language Modeling과 Next Sentence Prediction을 통해 사전 학습됨|Bidirectional Context, Pre-training and Fine-tuning, 다양한 NLP 작업에 활용|
|GPT (Generative Pre-trained Transformer)|대규모 언어 모델로, 언어 생성과 번역을 포함한 다양한 NLP 작업에 강력한 성능을 발휘. Transformer 기반으로 대규모 데이터에서 사전 학습됨|Unidirectional Context, Language Modeling, Transfer Learning|



### 객체 탐지 모델

|Shots|이름|내용|특징|
|-|-|-|-|
|Two|R-CNN<br>(Regions with CNN features)|전통적인 객체 탐지 방법:<br>Selective Search로 영역을 제안-><br>CNN으로 피처 벡터로 변환-><br>분류 및 경계 상자를 예측|Two-stage detector,<br>Selective Search,<br>CNN-based feature extraction|
|Two|Fast R-CNN|R-CNN의 개선, 전체 이미지에 대해 CNN을 한 번만 실행,<br>RoI Pooling로 각 제안 영역의 피처를 추출 분류 및 회귀|RoI Pooling,<br>End-to-end training,<br>Faster processing compared to R-CNN|
|Two|Faster R-CNN|Region Proposal Network (RPN)과<br>Fast R-CNN을 결합|RPN for region proposals,<br>ROI Pooling|
|One|YOLO<br>(You Only Look Once)|One-Shot. 빠른 속도와 높은 실시간 성능|Bounding box regression,<br>Class prediction|
|One|SSD<br>(Single Shot MultiBox Detector)|다양한 크기 객체를 탐지<br>다양한 스케일의 특성을 활용|Multi-scale feature maps,<br>Default boxes|

> RoI : Region of interest

## 모델 평가 하기

### 모델 평가
  
  1. 정확도(Accuracy):
      - 일반적 평가 지표
      - 데이터가 균형
  
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

<!-------------------------------------------------------------------------------------------------------> 

## 분류 및 회귀 문제
여러 종류의 분류 회귀 문제 유형

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

## 시계열
시계열 이론

### 참고 링크
[roboflow](https://roboflow.com/) <br>
[ultraytics](https://docs.ultralytics.com/integrations/roboflow/) <br>
learn open cv .com <br>
supervisely <br>
superb ai <br>
labelstudio.com -> 오디오에서 감성 분석 가능 <br>

### segmentation
[Label Studio](https://labelstud.io/guide/) <br>
[Label Me](https://github.com/labelmeai/labelme) <br>
[Anylabeling](https://github.com/vietanhdev/anylabeling) <br>
[X-Anylabeling](https://github.com/CVHub520/X-AnyLabeling) <br>
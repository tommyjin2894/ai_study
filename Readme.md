### [Notion 링크](https://royal-offer-53a.notion.site/KDT-2024-05-2024-09-10bf678f80468069b4e1e2f0a631131a?pvs=4)

### [전체 파일 구조](file_hirachy.md)

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
 
## 기본 시각화 코드
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

## 파이썬 기본 코드 연습

### 데이터 불균형 성능평가
- 데이터 불균형 판단기준 : 30%
    - 데이터 불 균형시 다양한 샘플링 및 다양한 metrics 설정
      - 정확도, 정밀도, 재현율, F1 스코어, AUC-ROC,

### 회귀 성능 평가
- RMSE, MAE 등 

### 😊결과를 표로 잘 정리하기😊

<!--------------------------------------->


## 마이닝 알고리즘

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

    <details>
    <summary>K-fold 교차 검증

    - 훈련 데이터를 k 개로 분할해 번갈아 가면서 훈련 평가
     |학습 모델|데이터1|데이터2|데이터3|데이터4|데이터5|
     | ---| --- | --- | --- | --- | --- |
     | 학습 1 | train | train | train | train | test |
     | 학습 2 | train | train | train | test | train |
     | 학습 3 | train | train | test | train | train |
     | 학습 4 | train | test | train | train | train |
     | 학습 5 | test | train | train | train | train |

    


### codes

전처리

    ```py
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    # StandardScaler
    model_std = StandardScaler()

    # MinMaxScaler
    model_minmax = MinMaxScaler()

    # RobustScaler
    model_robust = RobustScaler()
    ```


트레인 테스트 데이터 분할

    ```py
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,y_data,
        test_size=0.3,
        random_state=42,
        )
        # stratify=y_data
        # y라벨의 비율 유지
    ```


마이닝 알고리즘

    ```py
    # 머신러닝 라이브러리
    import sklearn
    # Main Models
    from sklearn.neighbors import KNeighborsClassifier # KNN
    from sklearn.tree import DecisionTreeClassifier # 의사결정나무
    from sklearn.linear_model import LogisticRegression # 로지스틱 회귀
    from sklearn.svm import SVC # 서포트 벡터 분류
    from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트 분류
    from sklearn.ensemble import GradientBoostingClassifier # 그래디언트 부스팅 분류
    from sklearn.naive_bayes import GaussianNB # 가우시안 나이브 베이즈
    from xgboost import XGBRegressor # XGB 회귀

    # Extras
    from sklearn.svm import NuSVC # Nu 서포트 벡터 분류
    from sklearn.svm import LinearSVC # 선형 서포트 벡터 분류
    from sklearn.ensemble import AdaBoostClassifier # AdaBoost 분류
    from sklearn.ensemble import ExtraTreesClassifier # Extra Trees 분류
    from sklearn.ensemble import HistGradientBoostingClassifier # 히스토그램 기반 그래디언트 부스팅 분류
    from sklearn.ensemble import BaggingClassifier # 배깅 분류
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # 선형 판별 분석
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis # 이차 판별 분석
    from sklearn.linear_model import RidgeClassifier # 릿지 분류
    from sklearn.linear_model import Perceptron # 퍼셉트론
    from sklearn.neural_network import MLPClassifier # 다층 퍼셉트론 분류
    from sklearn.gaussian_process import GaussianProcessClassifier # 가우시안 프로세스 분류
    from sklearn.naive_bayes import ComplementNB # 보완 나이브 베이즈
    from sklearn.naive_bayes import BernoulliNB # 베르누이 나이브 베이즈
    import xgboost as xgb # xgb (별칭)


    ```


교차 검증

    ```py
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score

    # 전처리기 na값 자동채움과
    # 랜덤 포레스트의 모델을 파이프라인으로 구축,
    # 동일한 결과를 위한 random_state=0
    my_pipe = Pipeline(steps=[
        ('preprocessor', SimpleImputer()) ,
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))
    ])

    #neg_mab_error 의 결과는 -으로 나오기 때문에 -1 을 곱해준다.
    scores = -1 * cross_val_score(
        my_pipe, X, y,
        cv=4,
        scoring='neg_mean_absolute_error')

    print(scores.mean())

    ```


PCA

    ```py
    import pandas as pd
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    train = pd.read_csv("table.csv")
    feature_df = train[train.columns[1:6]]
    pca = PCA(n_components = 2).fit_transform(feature_df)

    colors = {0:"blue", 1:"red"}
    c = train["attrition"].replace(colors)
    ```
    그래프 그리기
    ```py
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(pca[:,0], pca[:,1], alpha=0.6, color=c)
    ax.set(xlabel=R"X", ylabel=R"Y", title="PCA");
    ```


그리드 서치, 랜더마이즈드 서치

    ```py
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import StratifiedKFold

    def grid_search(x_train, y_train, params, base_model):
        model_base = GridSearchCV( # or Randomized Search
            # n_iter=10 # for Randomized Search
            base_model,
            params,
            cv = StratifiedKFold(3,shuffle=True, random_state = 209), # Cross Valid
            return_train_score=True,
            n_jobs = -1 # CPU or GPU?
            )

        model_base.fit(x_train, y_train)

        best_model = model_base.best_estimator_
        best_pred = best_model.predict(x_test)
        print("최고 정확도", metrics.accuracy_score(best_pred,y_test))
        return best_model, grid_model.cv_results_ # 최고성능 모델과 ,교차검증 결과

    params = {} # dict 형식 {"파라미터": list,}

    ```

<br>

<!------------------------------------------------------------------------------------------------------->

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
다양한 샘플링 기법
  
  ### 샘플링 기법 코드
  
  ```py
  # 언더 샘플링
  from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
  from imblearn.over_sampling import RandomOverSampler, SMOTE
  from imblearn.combine import SMOTEENN
  
  RandomUnderSampler
  EditedNearestNeighbours 
  
  # 오버 샘플링
  RandomOverSampler
  SMOTE
  
  # Both
  SMOTEENN
  ```
  


<!------------------------------------------------------------------------------------------------------->

## 딥러닝
다양한 딥러닝 모델 구조

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



비용 함수

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



활성화 함수

### 비용함수 및 손실함수
- 손실 함수 : 데이터 포인트 하나에 대한 오차 함수
- 비용 함수 : 전체 데이터에 대한 오차 함수
- 종류 :
    |이름|공식|출력 범위
    |-|-|-|
    |Sigmoid|$\phi = \frac{1}{1+e^{-x}}$|0 ~ 1|
    |tanh|$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$|-1 ~ 1|
    |ReLU|$f(z) = max(0, z)$|$0 \leq f(x)$|
    |Leaky ReLU|$f(z) = max(\epsilon z, z)$|$0 \leq f(x)$|
    |ELU|$f(x) = x \space \text{if } x \geq 0$<br>$f(x) = \alpha (e^x - 1) \space \text{if } x < 0$|$0 \leq f(x)$|
    |SoftPlus|$f(z) =  \ln(1 + e^x)$|$0 \leq f(x)$|
    |GeLU|$0.5 \cdot x \cdot \left( 1 + \tanh \left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 \cdot x^3 \right) \right) \right)$|Free <br>ReLU 계열 그래프와 비슷|



옵티마이저

### 옵티 마이저
- 옵티 마이저 : 수치 최적화 알고리즘
- 종류 :
    |이름|학습률|탐색 방향|알고리즘 기반|
    |-|-|-|-|
    |SGD|상수|기울기|탐색 방향
    |Momentum|상수|단기 누적 기울기|탐색 방향
    |AdaGrad|장기 파라미터 변화량과 반비례|기울기|학습 률
    |RMSProp|단기 파라미터 변화량과 반비례|기울기|학습 률
    |Adam|단기 파라미터 변화량과 반비례|단기 누적 Grad|학습 률



딥러닝 문제 해결 기법

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



### codes
                                                                        
다양한 layers

    - 기본 라이브러리
        ```py
        import tensorflow as tf
        from tensorflow.keras import datasets, layers, models, optimizers
        ```
    - CNN
        ```py
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(classes, activation='softmax'))
        ```
    -  RNN
        ```py
        model = models.Sequential()
        model.add(layers.Embedding(max_features, 64, input_length=max_len))
        model.add(layers.SimpleRNN(32, activation='tanh', return_sequences=False))
        model.add(layers.Dense(16, activation='tanh'))
        model.add(layers.Dense(2, activation = 'softmax'))
        model.summary()
        ```
    - LSTM
        ```py
        model1 = models.Sequential()
        model1.add(layers.Embedding(max_features, 128, input_length=maxlen))

        # return_sequence=True - 모든 스테이트를 내보냄
        model1.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
        model1.add(layers.Bidirectional(layers.LSTM(64)))
        model1.add(layers.Dense(2, activation = 'softmax'))
        model1.summary()
        ```
    


Auto Encoder

    ```py
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input, Embedding, Flatten
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.regularizers import l1 # 정규화 과적합 방지
    from tensorflow.keras.optimizers import Adam

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    ```

    ```py
    input_size = 784
    hidden_size = 128
    code_size = 32 # 잠재 공간 벡터의 크기

    input_img = Input(shape=(input_size,)) # 인풋

    hidden_1 = Dense(hidden_size, activation='relu')(input_img) # 인코더 부분

    code = Dense(code_size, activation='relu')(hidden_1) # 잠재 공간

    hidden_2 = Dense(hidden_size, activation='relu')(code) # 디코더 부분(인코더와 같다)
    output_img = Dense(input_size, activation='sigmoid')(hidden_2)
    # 인코더 부분과 디코더 부분 둘다 있어야 한다면,
    # 출력 층의 사이즈는 입력층의 사이즈와 같아야 한다.

    # 인코더부분과 디코더 부분의 결합
    autoencoder = Model(input_img, output_img)
    ```
    



seq2seq

    ```py
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense

    # 인코더
    encoder_inputs = Input(shape=(None, 50))
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # 디코더
    decoder_inputs = Input(shape=(None, 50))
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(50, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # 모델 컴파일
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 모델 요약
    model.summary()
    ```
    ```

    ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
    ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
    │ input_layer         │ (None, None, 50)  │          0 │ -                 │
    │ (InputLayer)        │                   │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ input_layer_1       │ (None, None, 50)  │          0 │ -                 │
    │ (InputLayer)        │                   │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ lstm (LSTM)         │ [(None, 256),     │    314,368 │ input_layer[0][0] │
    │                     │ (None, 256),      │            │                   │
    │                     │ (None, 256)]      │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ lstm_1 (LSTM)       │ [(None, None,     │    314,368 │ input_layer_1[0]… │
    │                     │ 256), (None,      │            │ lstm[0][1],       │
    │                     │ 256), (None,      │            │ lstm[0][2]        │
    │                     │ 256)]             │            │                   │
    ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    │ dense (Dense)       │ (None, None, 50)  │     12,850 │ lstm_1[0][0]      │
    └─────────────────────┴───────────────────┴────────────┴───────────────────┘
    ```
    


Transformer

    ```py
    from tensorflow.keras import layers

    class EncoderBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super().__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )

            # 레이어 정규화
            self.layernorm1 = layers.LayerNormalization()
            self.layernorm2 = layers.LayerNormalization()

            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
    ```

    ```py
    # 토큰 및 위치 임베딩 정의
    class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, maxlen, vocab_size, embed_dim):

            super().__init__()
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

        def call(self, x):
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions
    ```

    ```py
    # 모델 설계

    embed_dim = 32  # 각 토큰의 임베딩 벡터 크기
    num_heads = 2  # 어텐션 헤드의 수
    ff_dim = 32  # 완전연결층의 노드 수

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    x = EncoderBlock(embed_dim, num_heads, ff_dim)(x, training=True)
    x = EncoderBlock(embed_dim, num_heads, ff_dim)(x, training=True)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    ```

    ```py
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    ```

    ```py
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ input_layer_10 (InputLayer)     │ (None, 200)            │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ token_and_position_embedding_2  │ (None, 200, 512)       │    10,342,400 │
    │ (TokenAndPositionEmbedding)     │                        │               │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ encoder_block_8 (EncoderBlock)  │ (None, 200, 512)       │     6,336,544 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ global_average_pooling1d_2      │ (None, 512)            │             0 │
    │ (GlobalAveragePooling1D)        │                        │               │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_31 (Dropout)            │ (None, 512)            │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_22 (Dense)                │ (None, 20)             │        10,260 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_32 (Dropout)            │ (None, 20)             │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_23 (Dense)                │ (None, 2)              │            42 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
    ```

<!------------------------------------------------------------------------------------------------------->

## 데이터 증강 기법

keras 변형 증강

    ```py
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

    # 별표 위주로 쓰임
    tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0, #***
        width_shift_range=0.0, #***
        height_shift_range=0.0, #***
        brightness_range=None, #***
        shear_range=0.0,
        zoom_range=0.0, #***
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False, #***
        vertical_flip=False, #***
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None
    )
    ```

    ```py
    # 이미지 로드 (예시로 하나의 이미지를 사용)
    img = load_img('image.jpg')  # 이미지 경로
    x = img_to_array(img)  # 이미지를 배열로 변환
    x = x.reshape((1,) + x.shape)  # (1, height, width, channels) # 배치 차원 추가

    ```


AE 학습 증강

    ```py
    import os
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
    ```
    - 함수정의
        ```py
        # 이미지 파일 로드 및 전처리 - 로컬
        def load_images_local(folder_path, target_size=(128, 128)):
            images = []
            filenames = os.listdir(folder_path)
            for filename in filenames:
                try:
                    img_path = os.path.join(folder_path, filename)
                    img = load_img(img_path, target_size=target_size)
                    img = img_to_array(img) / 255.0
                    images.append(img)
                except:
                    pass
            return np.array(images)

            # 이미지 증강 및 저장
        def augment_images(autoencoder, images, save_dir):
            decoded_images = autoencoder.predict(images)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, img_array in enumerate(decoded_images):
                img = array_to_img(img_array)
                img.save(os.path.join(save_dir, f'augmented_image_{i}.png'))
        ```
    - 증강 시작
        ```py
        data_folder = '../data/data_mw/woman'  # 이미지 폴더
        save_folder = '../data/data_mw_add/woman_new'  # 증강된 이미지를 저장할 폴더

        # 이미지 로드 - 로컬
        images = load_images_local(data_folder)

        # 오토인코더 모델 구성 및 훈련
        autoencoder = build_autoencoder(input_shape=(128, 128, 3))
        autoencoder.fit(images, images, epochs=20, batch_size=20)

        # 이미지 증강 및 저장
        augment_images(autoencoder, images, save_folder)
        ```




## 다양한 Pretraind 모델
CNN 기반

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




자연어 처리 기반

|이름|내용|특징|
|-|-|-|
|Transformer|Attention 메커니즘을 사용하여 입력 시퀀스의 모든 요소를 동시적으로 처리하며, 장기 의존성 문제를 해결하는 모델|Self-Attention, Multi-Head Attention, Encoder-Decoder 구조|
|BERT (Bidirectional Encoder Representations from Transformers)|양방향 컨텍스트를 사용하여 자연어 이해 성능을 향상시킨 모델. Masked Language Modeling과 Next Sentence Prediction을 통해 사전 학습됨|Bidirectional Context, Pre-training and Fine-tuning, 다양한 NLP 작업에 활용|
|GPT (Generative Pre-trained Transformer)|대규모 언어 모델로, 언어 생성과 번역을 포함한 다양한 NLP 작업에 강력한 성능을 발휘. Transformer 기반으로 대규모 데이터에서 사전 학습됨|Unidirectional Context, Language Modeling, Transfer Learning|



객체 탐지 모델

|Shots|이름|내용|특징|
|-|-|-|-|
|Two|R-CNN<br>(Regions with CNN features)|전통적인 객체 탐지 방법:<br>Selective Search로 영역을 제안-><br>CNN으로 피처 벡터로 변환-><br>분류 및 경계 상자를 예측|Two-stage detector,<br>Selective Search,<br>CNN-based feature extraction|
|Two|Fast R-CNN|R-CNN의 개선, 전체 이미지에 대해 CNN을 한 번만 실행,<br>RoI Pooling로 각 제안 영역의 피처를 추출 분류 및 회귀|RoI Pooling,<br>End-to-end training,<br>Faster processing compared to R-CNN|
|Two|Faster R-CNN|Region Proposal Network (RPN)과<br>Fast R-CNN을 결합|RPN for region proposals,<br>ROI Pooling|
|One|YOLO<br>(You Only Look Once)|One-Shot. 빠른 속도와 높은 실시간 성능|Bounding box regression,<br>Class prediction|
|One|SSD<br>(Single Shot MultiBox Detector)|다양한 크기 객체를 탐지<br>다양한 스케일의 특성을 활용|Multi-scale feature maps,<br>Default boxes|

> RoI : Region of interest



### codes
- 이미지 분류
    이미지 기본 전처리

        ```py
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # 이미지 데이터 전처리
        def preprocess_image(image_size=(224, 224), batch_size=32):
            datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )

            train_generator = datagen.flow_from_directory(
                'path/to/dataset',
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )

            validation_generator = datagen.flow_from_directory(
                'path/to/dataset',
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )

            return train_generator, validation_generator

        ```
    

    다양한 CNN based models

        ```py
        from tensorflow.keras.applications import (
            VGG16, VGG19,
            ResNet50, ResNet101, ResNet152, 
            InceptionV3, InceptionResNetV2,
            Xception,
            DenseNet121, DenseNet169, DenseNet201,
            MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
            NASNetMobile, NASNetLarge,
            EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, 
            EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        
        # 모델 딕셔너리 생성
        models_dict = {
            'VGG16': VGG16,
            'VGG19': VGG19,
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'ResNet152': ResNet152,
            'InceptionV3': InceptionV3,
            'InceptionResNetV2': InceptionResNetV2,
            'Xception': Xception,
            'DenseNet121': DenseNet121,
            'DenseNet169': DenseNet169,
            'DenseNet201': DenseNet201,
            'MobileNet': MobileNet,
            'MobileNetV2': MobileNetV2,
            'MobileNetV3Small': MobileNetV3Small,
            'MobileNetV3Large': MobileNetV3Large,
            'NASNetMobile': NASNetMobile,
            'NASNetLarge': NASNetLarge,
            'EfficientNetB0': EfficientNetB0,
            'EfficientNetB1': EfficientNetB1,
            'EfficientNetB2': EfficientNetB2,
            'EfficientNetB3': EfficientNetB3,
            'EfficientNetB4': EfficientNetB4,
            'EfficientNetB5': EfficientNetB5,
            'EfficientNetB6': EfficientNetB6,
            'EfficientNetB7': EfficientNetB7
        }
        
        # 모델 생성 함수
        def create_model(model_name, input_shape=(224, 224, 3), num_classes=1000):
            base_model = models_dict[model_name](weights='imagenet', include_top=False, input_shape=input_shape)
        
            # 모델 구조 정의
            model = Sequential()
            model.add(base_model)
            model.add(GlobalAveragePooling2D())
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))
            
            return model
        
        # 모델 생성 예시
        model_name = 'ResNet50'
        input_shape = (224, 224, 3)
        num_classes = 10  # 데이터셋에 따른 클래스 수
        model = create_model(model_name, input_shape, num_classes)
        
        # 모델 요약 출력
        model.summary()
        
        ```

        
- LM Models
    BERTopic
        : 텍스트의 토픽 추출 및 시각화 - 트랜스 포머 기반, 대량 문서 자동 토픽 추출, 토픽 사이의 관계 파악<br>
        : 주요 기능 - 자동 토픽 수 검출, 유사한 토픽 제거, 시각화, 동적 토픽 모델링(시간에 따라 변하는 트렌드 추척)<br>

        ```py
        !pip install update BERTopic
        ``` 

        ```py
        from bertopic import BERTopic

        import pandas as pd
        import numpy as np

        # 데이터 읽어오기
        df = pd.read_csv('topic_example.csv', engine='python')

        # 내용 처리 하기
        docs = df['text'].to_list()
        docs = [re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 .]', '', s) for s in docs]
        docs = [re.sub(r'\s+', ' ', s) for s in docs]

        model = BERTopic(language='korean', nr_topics=10, calculate_probabilities=True)
        topics, probabilities = model.fit_transform(docs)

        # 토픽 요약
        model.get_topic_info()

        # 토픽 (3) 의 상세 내용 확인
        model.get_topic(3)

        # 다양한 시각화
        model.visualize_barchart(top_n_topics=8) # 바 차트
        model.visualize_topics() # 주제간 거리 차트
        model.visualize_hierarchy(top_n_topics=10) # 계층적 클러스터링
        model.visualize_heatmap(top_n_topics=10) # 히트맵
        model.visualize_distribution(model.probabilities_[0], min_probability=0.015) # 특정 문서 주제 분포 시각화
        ```
        외부 모델
        ```py
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        topic_model = BERTopic(embedding_model=embedding_model)

        topics, probabilities = topic_model.fit_transform(docs)

        # 임베딩 벡터 만들기
        embeddings = embedding_model.encode(docs)
        print("임베딩 차원:", embeddings.shape)
        print("첫 번째 문서의 임베딩 벡터:", embeddings[0])
        ```
        
    
        
    GPT

        ```py
        import torch
        from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')

        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        text = '''우리 삶에 가장 필요한 덕목은 무엇일까? 그건 바로 취'''

        input_ids = tokenizer.encode(text, return_tensors='pt')
        gen_ids = model.generate(input_ids,
                                   max_length=100,
                                   repetition_penalty=2.0,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id,
                                   bos_token_id=tokenizer.bos_token_id,
                                   use_cache=True)
        generated = tokenizer.decode(gen_ids[0])
        print(generated)
        ```
        ```
        우리 삶에 가장 필요한 덕목은 무엇일까? 그건 바로 취업과 승진이다.
        그런데 이게 왜 중요한지 알 수 없다.
        이렇게 취업난에 허덕이는 청년들이 어떻게 하면 좋은 직장을 구할까 고민하는 것은 당연한 일이다.
        하지만 이런 고민을 하는 이유는 뭘까?
        바로 '취업' 때문이다.
        취업을 위해선 무엇보다 자신의 적성과 능력에 맞는 일자리를 찾아야 한다.
        그래야 자신이 원하는 직장에 갈 확률이 높아진다.
        또한 자기계발을 위한 노력도 필요하다.
        자신의 능력을 최대한 발휘할 기회를 만들어주는 것이
        ```
    

- [Object Detection Models](__KDT__season_3/09_Object_Detection/)

## 모델 평가 하기

### 내용 정리

모델 평가
  
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



<!-------------------------------------------------------------------------------------------------------> 

## 시계열
시계열 이론


<!-------------------------------------------------------------------------------------------------------> 

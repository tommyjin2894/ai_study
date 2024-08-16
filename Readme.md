# 정리

## 모델 설계
<details>
<summary>설명</summary>

- baseline baseline 모델을 설정하고, 이보다 좋은성능 내기
- 데이터 불균형 30% 정도 일때 부터 조치를 취해야함
    - 언더 샘플링
    ```py
        RandomUnderSampler
        EditedNearestNeighbours 
    ```
    - 오버 샘플링
    ```py
        RandomOverSampler
        SMOTE
    ```
    - Both
    ```py
        SMOTEENN
    ```
- 데이터 불균형 시 baseline 다양하게  metrics 설정
    - 정확도, 정밀도, 재현율, F1 점수, AUC-ROC, 회귀 - RMSE, MAE 등 
- 표로 잘 정리하기

</details>

## 다양한 샘플링 기법
<details>
<summary>설명</summary>

- 임의 추출
- 계통 추출 (공장)
- 층화 추출 (나이 및 성별별 추출)
- 군집 추출 (전국 -> 서울)
- 다 단계 추출 (전국 -> 서울 -> 남성)
- 비 확률적 추출 (임의 추출)

주의 : 편향적인 데이터가 안되게
</details>


## 마이닝 알고리즘
<details>
<summary>설명</summary>

- 머신러닝 모델(지도 학습) <br>
    |모델|이름|설명|
    |---|---|---|
    |분류|Decision Tree|트리구조로 데이터를 분류, 조건 분기|
    |-|Random Forest|앙상블 기법중 baseline Bagging 중 하나 <br> 여러개의 DT로 구성|
    |-|KNN|가까운 K 개의 데이터를 기반으로 결정 <br> baseline L1 및 L2 거리|
    |-|SVM|클래스 간의 경계를 최대화하여 초평면을 찾는다.|
    |회귀|Linear Regression|선형 관계 모델링|
    |-|Logistic Regression|이진 분류를 위한 회귀 분석 기법,<br> baseline 확률로 출력값을 변환|
    |인공 신경망|NN|여러층의 뉴런|
    |-|CNN|Convolution NN|
    |-|RNN|Recurrent NN|
    |-|LSTM|Long, Short Term Memory|
    |-|Auto Encoder|Encoder->Latent Space->Decoder|
    |-|Transformer|Self Attention, ED Attention|
    |기타|AdaBoost|약한 학습기 $\times$ N = 강한 학습기|
    |-|XGBoost|Gradient Boosting Machines 의 효율적이고 강력하게 개선|

<br>

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
<br>

- 기법
    |종류|이름|설명|
    |---|---|---|
    |기법|K-fold 교차 검증|점수 평균|
    |-|Grid search|모든 경우의수를 본다|
    |-|Randomized search|랜덤한 경우의수를 본다|
    |앙상블|bagging <br> (baseline Bootstrap baseline Aggregatbaseline ing)|1.baseline N 개의 샘플을 뽑기 ->집어넣고 baseline N 개의 샘플을 뽑는다. <br> 2. 중복이 생길 수 있음|
    |-|Boosting|약한 학습기 $ \times $ N = 강한 학습기 <br>AdaBoost, XGBoost, Lgith GBM, Cat Boost 등|
    |-|Stacking|여러 개의 기초모델의 예측을 종합하여 새로운 메타모델 생성|

    <details>
    <summary>K-fold 교차 검증</summary>
    
    - 훈련 데이터를 k 개로 분할해 번갈아 가면서 훈련 평가
        |||||||
        | ---   | --- | --- | --- | --- | --- |
        | 학습 1 | train | train | train | train | test |
        | 학습 2 | train | train | train | test | train |
        | 학습 3 | train | train | test | train | train |
        | 학습 4 | train | test | train | train | train |
        | 학습 5 | test | train | train | train | train |

    </details>

</details>

## 분류 및 회귀 문제
<details>
<summary>설명</summary>

### 분류 문제
|이름|내용|
|-|-|
|Mnist|손 글씨 분류|
|CIFAR|사진 대상 분류|
|텍스트<br>표정<br>감성<br>|주로 시퀀스 context 문제|
|일 대 다 분류|단계별로 하나씩 분류|

### 회귀 문제
|이름|내용|
|-|-|
|주택 가격 예측|가격 예측|
|주식 가격 예측|가격 예측|
|온도 예측|기상 데이터로 온도 예측|

</details>
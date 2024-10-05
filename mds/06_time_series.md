# 시계열 데이터

### 시계열 데이터란?
- 시간 순서에 따라 발생하는 연속적인 데이터를 기반으로 미래를 예측하는 데이터.

### 다변량, 단변량
- **다변량 (Multivariate)**: 여러 종속 변수(여러 개의 정답 라벨).
- **다변수 (Multivariable)**: 여러 독립 변수(다양한 입력).
- **단변량 (Univariate)**: 단일 정답 라벨을 가지는 데이터.

### 시간의 종속성 (Time Dependence)
- 과거 데이터를 기반으로 미래를 예측하는 특성.
- 불규칙성이 있으면 예측이 어려워짐.

### 시계열 종속성 (Serial Dependence)
- 시계열 데이터가 이전 값에 의존하는 패턴. 이전 데이터가 현재와 미래의 데이터에 영향을 미치는 경우를 설명함.

### Cycles
- 데이터가 주기적으로 반복되는 현상을 설명. 반드시 시간에 따라 반복되지 않아도 되며, 자연스럽게 반복되는 현상.

### 시계열 데이터의 주요 특성
- **계절성 (Seasonality)**: 일일, 주간, 연간 등 주기적으로 변화하는 패턴.
- **추세 (Trend)**: 장기적인 변화 경향을 나타냄. 이동 평균 플롯을 통해 추세를 시각화할 수 있음.

### 다항 특성 (Polynomial Features)
- 시계열 데이터를 다항식으로 표현할 수 있음. 예를 들어, $degree$가 1일 때, 2일 때, n일 때의 다항식 모델이 각각 있음.
    - $y = w \times time + b$ (일차식)
    - $y = w_0 \times time^2 + w_1 \times time + b$ (이차식)
    - 다항식 모델로 예측 가능.

### 정상성 (Stationarity)
- 시계열 분석이 용이하려면 데이터가 정상성을 가져야 함.
- **정상성 판단 기준**:
    - 시각적으로: 상승 또는 하락이 지속되지 않음, 변동폭 일정.
    - 통계적으로: 평균과 분산이 일정하며, 공분산이 시간과 무관하게 유지됨.
    
- **비정상적 데이터**는 변환을 통해 정상성을 가지게 한 후 분석을 진행해야 함.

---
## 시계열 → 정상성 검증 → 정상성을 띄도록 변환 → 정확도 향상
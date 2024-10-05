
# 모델 평가 하기

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
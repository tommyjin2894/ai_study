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

## AI study
- [Notion 링크](https://royal-offer-53a.notion.site/KDT-2024-05-2024-09-10bf678f80468069b4e1e2f0a631131a?pvs=4)
- [전체 파일 구조](markdown/00_files.md)

### 예제 데이터 셋
- [heart](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

### 목차
- [0. 파일](markdown/00_files.md)
- [1. 기초](markdown/01_basics.md)
- [2. 머신러닝](markdown/02_ml.md)
- [3. 샘플링](markdown/03_sampling.md)
- [4. 딥러닝](markdown/04_dl.md)
- [5. 모델 평가](markdown/05_metrics.md)
- [6. 분류 및 회귀 문제](markdown/06_diversity.markdown)
- [7. 시계열](markdown/07_time_series.md)

## 참고 링크
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
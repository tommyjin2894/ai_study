### 링크
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
# 내용정리

### [파이썬 기본 코드 연습](mds/1_python_basic_codes.md)
### [머신러닝](mds/01_ml.md)
### [샘플링](mds/02_sampling.md)
### [딥러닝](mds/03_dl.md)
### [모델 평가](mds/04_metrics.md)
### [분류 및 회귀 문제](mds/05_diversity.mds)
### [시계열](mds/06_time_series.md)

# 참고 링크
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
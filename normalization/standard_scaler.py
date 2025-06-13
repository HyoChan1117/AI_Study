from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Scale이란?
# 각 feature의 숫자 크기 범위
# Feature의 Scale을 맞추는 이유?
# - Gradient Descent의 수렴 속도를 높이기 위해
# - Feature의 중요도를 동일하게 하기 위해
# - 모델의 성능을 높이기 위해
# - 모델의 안정성을 높이기 위해

# 각각의 특성마다 

scaler = StandardScaler()

values = np.arange(10).reshape(-1, 1)  # 2차원 배열로 변환

# 현재 데이터셋의 평균, 분산, 표준편차를 계산하는 역할 -> fit()
fit_values = scaler.fit(values)

print(fit_values.mean_, fit_values.var_, fit_values.scale_)
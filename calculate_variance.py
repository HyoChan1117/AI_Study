import numpy as np
from sklearn.preprocessing import StandardScaler

# 특성들의 단위 일치화 : Scaling
# 분산 : 평균으로 얼마나 떨어져 있는지

x = np.arange(10)

# X의 합계와 평균
x_sum = sum(x)
x_avg = x_sum / len(x)

# 분산
variance = 0.0

# 분산 계산
for item in x:
    variance += (item - x_avg)**2
    
# 분산을 데이터 개수로 나누기
variance /= len(x)

print(variance)

# numpy를 이용한 평균과 분산 계산
np_avg = x.mean()  # 평균
np_variance = x.var()  # 분산
np_std = np.sqrt(np_variance)  # 표준편차

print(np_avg, np_variance, np_std)
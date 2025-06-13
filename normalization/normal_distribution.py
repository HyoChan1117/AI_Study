import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 평균이 0이고, 표준편차가 1인 정규분포 난수 생성기
# 난수 발생의 범위가 없는 이유
# 평균 0, 표준편차 1인 정규분포를 따르기 때문
# 표준편차, 분산 : 전체 데이터의 흩어진 정도를 스칼라 값으로 나타낸 것
np.random.randn()

# 정규분포를 따르는 10000개의 랜덤 값 생성
# *10 -> 표준편차 10
# +50 -> 평균 50
values = [ np.random.randn() * 10 + 50 for _ in range(10000) ]

plt.hist(values, bins=20, edgecolor='black', alpha=0.7, color='black')

plt.show()
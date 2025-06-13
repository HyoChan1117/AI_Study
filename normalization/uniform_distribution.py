import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Uniform distribution
# 각 데이터의 값이 동일한 확률로 발생하는 분포
values = [ np.random.randint(1, 100) for _ in range(100000)]

plt.hist(values, bins=20, edgecolor='black', alpha=0.7, color='black')

plt.show()
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 단위 행렬
one_hot = np.eye(4)

# 
y_list = [0, 1, 0, 3, 2, 3]

# encoding
one_hot_value = one_hot[y_list]

# decoding
np_decoded = np.argmax(one_hot_value, axis=1)

# axis=1은 행(row) 방향으로 최대값을 찾는다는 의미
# axis=0은 열(column) 방향으로 최대값을 찾는다는 의미
# axis=None은 전체 배열에서 최대값을 찾는다는 의미
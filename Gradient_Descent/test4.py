import numpy as np
import matplotlib.pyplot as plt

# 순서
# 데이터셋의 목적
# 1. 학습
# 2. 평가

# Data set
# input
# input data, features
# H(x) -> input data : x1 ~ xn
x_train = [ np.random.rand() * 10 for _ in range(50) ]
y_train =  [ val + np.random.rand() * 5 for val in x_train ]  # y -> label, y^ or y- -> prediction value
# 선형 회귀를 위한 테스트 데이터를 만들기 위한 


print(x_train)

print(y_train)

plt.scatter(x_train, y_train, color = "blue")
plt.show()

# output
# label
# f(x1) -> f(x2)
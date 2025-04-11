import numpy as np

num_of_samples = 5
num_of_features = 2

np.random.seed(0)  # 랜덤 시드 고정
np.set_printoptions(suppress=True, precision=2)  # 소수점 이하 1자리까지 출력

# data set
# feature 값 2개
# H(x) = w1 * x1 + w2 * x2 + b
# H(x) = 5X + 3X + 4
X = np.random.rand(num_of_samples, num_of_features) * 10  # 0.0 ~ 10.0
x_true = [5, 3]
b_true = 4
noise = np.random.rand(num_of_samples) * 0.5  # 노이즈 추가

y = X[:, 0] * 5 + X[:, 1] * 3 + b_true + noise  # y = H(x) + noise
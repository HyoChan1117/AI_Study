import numpy as np

num_features = 3
num_samples = 2

np.random.seed(0)  # 랜덤 시드 고정
np.set_printoptions(suppress=True, precision=3)  # 소수점 이하 3자리까지 출력

# h(x) = wx1 + wx2 + wx3 + b
# ndarray([x1, x2, x3])
# 3개의 샘플, 2개의 특성
X = np.random.rand(num_samples, num_features) * 10  # 0.0 ~ 10.0

print(X)

w_true = np.random.randint(1, 10, num_features)  # 1 ~ 10 사이의 정수 3개
b_true = np.random.randn() * 0.5  # 평균 0, 표준편차 0.5인 정규분포에서 랜덤 샘플링


y = X[:, 0] * w_true[0] + X[:, 1] * w_true[1] + X[:, 2] * w_true[2] + b_true  # y = wx + b + noise (실제 y 값)
y_ = X @ w_true + b_true   # 행렬 곱셈을 통한 y 계산
print(f"y: {y}, y_: {y_}")

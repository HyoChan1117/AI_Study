import numpy as np

num_features = 4
num_samples = 1000

np.random.seed(0)

X = np.random.rand(num_samples, num_features) * 2

w_true = np.random.randint(1, 11, (num_features, 1))  # 1 ~ 10 사이의 정수 4개
b_true = np.random.randn() * 0.5  # 평균 0, 표준편차 0.5인 정규분포에서 랜덤 샘플링

y = X @ w_true + b_true  # y = wx + b + noise (실제 y 값)

# 초기의 w값은 달라야 하기 때문에 랜덤으로 설정
w = np.random.rand(num_features, 1)  # 초기 w: (4, 1)
b = np.random.rand()  # 초기 b: 스칼라

# 하이퍼파라미터 설정
learning_rate = 0.01
epochs = 10000

# 학습 루프
for epochs in range(epochs):
    # 예측값
    y_pred = X @ w + b  # 예측값: (1000, 1)

    # 오차 계산
    error = y_pred - y  # 오차: (1000, 1)

    # 기울기 계산
    gradient_w = (X.T @ error) / num_samples # shape: (4, 1)
    gradient_b = error.mean()  # 스칼라

    # 파라미터 업데이트
    w -= gradient_w * learning_rate
    b -= gradient_b * learning_rate
    
print(f"w_true : {w_true}, b_true : {b_true}")
print(f"w : {w}, b : {b}")
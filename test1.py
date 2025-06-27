from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
dataset = load_digits()
X = dataset.data    # (1797, 64)
y = dataset.target  # (1797,)

# 2. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # (1437, 64)
X_test = scaler.transform(X_test)         # (360, 64)

# 4. 가중치 및 바이어스 초기화
w = np.random.randn(X_train.shape[1], 10)  # (64, 10)
b = np.random.randn(10)                    # (10,)

# 학습률
learning_rate = 0.01

# 5. logit
z = (X_train @ w) + b    # (1437, 10)

# 6. 예측값
exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
prediction = exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 7. 정답값 원-핫 인코딩
y_one_hot = np.eye(10)[y_train]    # (1437, 10)

# 8. 손실 계산
loss = -np.mean(y_one_hot * np.log(prediction + 1e-15), axis=1)

# 9. 경사(Gradient) 계산
error = prediction - y_one_hot  # (1437, 10)
gradient_w = (X_train.T @ error) / len(X_train)   # (64, 10)
gradient_b = np.mean(error, axis=0)               # (10,)

# 10. 파라미터 업데이트
w = w - learning_rate * gradient_w
b = b - learning_rate * gradient_b

print(loss)
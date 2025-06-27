import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=5, suppress=True)

# 1. 데이터셋 로드
dataset = load_breast_cancer()
X = dataset.data     # (569, 30)
y = dataset.target   # (569, 0)

# 2. 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)   # (455, 30), (114, 30), (455,), (114,)

# 3. 특성 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # (455, 30)
X_test = scaler.transform(X_test)         # (114, 30)

# 4. 가중치 및 바이어스 초기화
w = np.random.randn(X_train.shape[1], 1)  # (30, 1)
b = np.random.randn()

# 학습률
learning_rate = 0.01

# 5. logit
z = X_train @ w + b   # (455, 1)

# 6. 예측값
prediction = 1 / (1 + np.exp(-z))   # (455, 1)

# 7. y_train 1차원 벡터를 2차원으로 변경
y_train = y_train.reshape(-1, 1)

# 8. 오차
error = prediction - y_train   # (455, 1)

# 9. 경사(Gradient) 값 구하기
gradient_w = (X_train.T @ error) / len(X_train)   # (30, 1)
gradient_b = error.mean()

# 10. 파라미터 업데이트
w = w - learning_rate * gradient_w   # (30, 1)
b = b - learning_rate * gradient_b

# 11. 손실값
loss = -np.mean(y_train * np.log(prediction) + (1 - y_train) * np.log(1 - prediction))

print(loss)
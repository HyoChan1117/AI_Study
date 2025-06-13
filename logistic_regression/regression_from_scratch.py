import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 분할
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# 2. 훈련/테스트 셋 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)   # stratify=y: 클래스 비율 유지

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)  # (455, 1)
y_test = y_test.reshape(-1, 1)  # (114, 1)

w = np.random.randn(X_train.shape[1], 1)  # (30, 1)
b = np.random.randn()  # (1,)

learning_rate = 0.01
epochs = 100000

for epoch in range(epochs):
    # prediction
    # z = w * x + b
    z = X_train @ w + b  # (455, 1)

    # sigmoid
    prediction = 1 / (1 + np.exp(-z))  # (455, 1)
    # error
    error = prediction - y_train  # (455, 1)

    # gradient
    gradient_w = (X_train.T @ error) / len(X_train)  # (30, 1)
    gradient_b = error.mean()  # (1,)

    # Upate parameters
    w = w - learning_rate * gradient_w  # (30, 1)
    b = b - learning_rate * gradient_b  # (1,)

    # calculate loss
    loss = -np.mean(y_train * np.log(prediction) + (1 - y_train) * np.log(1 - prediction))

print(f"Loss: {loss:.4f}")



# w -> 30개, b -> 1개
np.set_printoptions(precision=15, suppress=True)
test_z = X_test @ w + b  # (114, 1)
test_prediction = 1 / (1 + np.exp(-test_z))  # (114, 1)
test_result = (test_prediction >= 0.5).astype(int)

print(test_result.shape)  # (114, 1)
print(y_test.shape)  # (114, 1)

accuracy = np.mean(test_result == y_test)
print(accuracy)


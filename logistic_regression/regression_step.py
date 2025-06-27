import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 유방암 데이터셋 로드
dataset = load_breast_cancer()
X = dataset.data         # 입력 특성 (shape: [569, 30])
y = dataset.target       # 정답 레이블 (0 또는 1)

# 2. 훈련/테스트 데이터 분할 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. 특성 정규화 (평균 0, 표준편차 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 특성 개수 (30개)
num_features = X_train.shape[1]

# 4. 가중치 및 편향 초기화
w = np.random.randn(num_features, 1)  # (30, 1)
b = np.random.randn()                 # 스칼라

# 학습률 설정
learning_rate = 0.01

# 출력 시 소수점 이하 5자리까지 표현
np.set_printoptions(precision=5, suppress=True)

# 5. 정답 레이블 shape 변경: (455, 1)
y_train = y_train.reshape(-1, 1)

# 6. 선형 조합 계산: z = Xw + b
z = X_train @ w + b  # (455, 1)

# 7. 시그모이드 함수 적용 → 확률 출력
# sigmoid(z) = 1 / (1 + exp(-z))
prediction = 1 / (1 + np.exp(-z))

print(prediction)  # 예측 확률값
print(prediction.shape, y_train.shape)  # (455, 1), (455, 1)

# 8. 오차 계산: 예측값 - 실제값
error = prediction - y_train  # (455, 1)

# 9. 경사(Gradient) 계산
# ∂Loss/∂w = X.T @ error / n
gradient_w = (X_train.T @ error) / len(X_train)  # (30, 1)
# ∂Loss/∂b = 평균 오차
gradient_b = error.mean()                        # 스칼라

print(gradient_w.shape, gradient_b.shape)  # (30, 1), ()

# 10. 가중치 및 편향 업데이트 (경사 하강법 1회)
w = w - learning_rate * gradient_w
b = b - learning_rate * gradient_b

# 11. 손실 함수(Loss) 계산 - 이진 크로스 엔트로피(Binary Cross Entropy)
# Loss = -[y*log(p) + (1-y)*log(1-p)]
loss = -np.mean(y_train * np.log(prediction) + (1 - y_train) * np.log(1 - prediction))

print(loss)

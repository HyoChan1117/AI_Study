from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
digits = load_digits()  # 사이킷런에서 제공하는 손글씨 숫자 데이터셋
features = digits.data  # (1797, 64): 8x8 이미지 벡터
labels = digits.target  # (1797,): 0~9 클래스 정수

# 2. 학습/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)  # (1437, 64)
X_test_std = scaler.transform(X_test)   # (360, 64)

np.set_printoptions(precision=5, suppress=True)  # 소수점 5자리까지 출력

num_features = X_train_std.shape[1]  # 특성 개수: 64
num_samples = X_train_std.shape[0]  # 샘플 개수: 1437
num_classes = 10  # 클래스 개수: 10 (0~9)
# 왜 1797개가 아닌 1437개일까?
# 1797개 중 20%인 360개가 테스트셋으로 분리되었기 때문

# 4. 가중치 및 편향 초기화
# pixels 64개에 대해 각 클래스(0~9)에 대한 가중치 (행: 특성, 열: 클래스)
# 정답인 클래스에 대한 가중치가 높아야 하므로, 각 클래스에 대해 64개의 가중치가 필요
w = np.random.randn(num_features, 10)  # (64, 10): 10개의 클래스
print(f"w shape: {w.shape}")  # (64, 10)

# 각 클래스에 대한 편향
b = np.random.randn(10)  # (10,): 각 클래스에 대한 편향 - 행 벡터
learning_rate = 0.01
epochs = 10000

# 5. 소프트맥스 회귀 학습
for epoch in range(epochs):
    # 5-1. 예측 계산
    # X (1437, 64) @ w (64, 10) + b (10,)
    logit = X_train_std @ w + b  # (1437, 10) / (64, 10)
    logit_max = np.max(logit, axis=1, keepdims=True)  # (1437, 1) - 오버플로우 방지용
    # keepdims=True: 차원 유지 - reshape 없이 할 수 있음
    exp_logit = np.exp(logit - logit_max)  # (1437, 10) - 오버플로우 방지
    logit -= logit_max  # (1437, 10) - 오버플로우 방지
    exp_logit = np.exp(logit)  # (1437, 10)
    exp_logit_sum = np.sum(exp_logit, axis=1, keepdims=True)  # (1437, 1) - 각 행의 합

    # 5-2. 소프트맥스 함수 적용
    softmax = exp_logit / exp_logit_sum  # (1437, 10)
    
    # 5-3. 원-핫 인코딩
    y_one_hot = np.eye(10)[y_train]  # (1437, 10)

    # 5-4. 손실 계산
    # error = softmax(1437, 10) - y_one_hot(1437, 10)
    # 에러 = 예측값 - 실제값(원-핫 인코딩)
    loss = -np.mean(np.sum(y_one_hot * np.log(softmax + 1e-15), axis=1))

    # 5-5. 그래디언트 계산
    error = softmax - y_one_hot  # (1437, 10)
    gradient_w = (X_train_std.T @ error) / num_samples  # (64, 10)
    gradient_b = np.mean(error, axis=0)  # (10,) 

    # 5-6. 파라미터 업데이트
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b
    
    # 6. Loss
    loss = -np.sum(np.log(softmax +  1e-15) * y_one_hot) / num_samples
    print(f"Final Loss: {loss:.4f}")
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
# 7. 테스트셋에 대한 예측
test_logit = X_test_std @ w + b  # (360, 10)
test_logit_max = np.max(test_logit, axis=1, keepdims=True)  # (360, 1) - 오버플로우 방지용
test_logit -= test_logit_max  # (360, 10) - 오버플로우 방지
test_exp_logit = np.exp(test_logit)  # (360, 10)
test_exp_logit_sum = np.sum(test_exp_logit, axis=1, keepdims=True)  # (360, 1) - 각 행의 합
test_softmax = test_exp_logit / test_exp_logit_sum  # (360, 10)
test_predictions = np.argmax(test_softmax, axis=1)  # (360,)

# 8. 정확도 계산
accuracy = np.mean(test_predictions == y_test)  # (360,)
print(f"Test Accuracy: {accuracy:.4f}")

for idx in range(0, 10):
    print(f"{labels[idx]}: {[idx]}")
import numpy as np

# 설정
# 샘플 5개, 특성 2개 데이터셋
num_of_samples = 5
num_of_features = 2

# 랜덤 시드 고정 및 소수점 2자리로 출력 설정
np.random.seed(0)
np.set_printoptions(suppress=True, precision=2)

# 데이터 생성
X = np.random.rand(num_of_samples, num_of_features) * 10

# 정답 가중치 및 편향 설정 (실제 값)
w_true = [5, 3]
b_true = 4

# 실제 값에 노이즈 추가
# y = wx + b + noise (실제 y 값)
noise = np.random.rand(num_of_samples) * 0.5
y = X[:, 0] * w_true[0] + X[:, 1] * w_true[1] + b_true + noise

# 초기 가중치 및 편향 설정 (학습 대상)
w = np.random.rand(num_of_features)  # 초기 w: (2,)
b = np.random.rand()          # 초기 b: 스칼라

# 학습률 및 에폭 설정
learning_rate = 0.01
epochs = 10000

# 학습 루프
for epoch in range(epochs):
    # 예측값 계산: y_pred = X @ w + b
    y_pred = X @ w + b

    # 오차 계산
    error = y_pred - y  # shape: (5,)

    # 손실 (MSE)
    loss = np.mean(error ** 2)

    # 기울기 계산 (벡터화된 연산)
    gradient_w = (2 / num_of_samples) * (X.T @ error)  # shape: (2,)
    gradient_b = (2 / num_of_samples) * np.sum(error)  # 스칼라

    # 파라미터 업데이트
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    # 중간 출력 (선택 사항)
    if epoch % 1000 == 0:
        print(f"[{epoch:>5} epoch] loss: {loss:.4f}  w: {w.round(2)}  b: {b:.2f}")

# 학습 결과 출력
print("\n✅ 최종 결과:")
print(f"예측한 가중치 w: {w.round(2)}")
print(f"예측한 절편 b: {b:.2f}")
print(f"실제 가중치 w_true: {w_true}")
print(f"실제 절편 b_true: {b_true}")

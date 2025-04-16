import numpy as np

# 실제 정답 가중치와 절편
w_true = np.array([5.0, 3.0])
b_true = 4.0

# 학습 데이터
samples = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 5.0]
])
y = np.dot(samples, w_true) + b_true  # 정답값 y 계산

# 학습 파라미터 초기화
w = np.array([0.2, 0.3])  # 초기 가중치
b = 0.1                   # 초기 절편
learning_rate = 0.01
epochs = 1000

# 학습 시작
for epoch in range(epochs):
    gradient_w = np.zeros_like(w)
    gradient_b = 0.0
    
    for dp, target in zip(samples, y):
        predict_y = np.dot(w, dp) + b
        error = predict_y - target
        gradient_w += error * dp
        gradient_b += error

    # 평균 기울기
    gradient_w /= len(samples)
    gradient_b /= len(samples)
    
    # 업데이트
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    # 손실 출력
    loss = np.mean((samples @ w + b - y) ** 2)
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"[{epoch:>2} epoch] loss: {loss:.4f}  w: {w.round(3)}  b: {b:.3f}")

# 최종 결과 출력
print("\n✅ 최종 결과:")
print(f"예측한 가중치 w: {w.round(3)}")
print(f"예측한 절편 b: {b:.3f}")
print(f"실제 가중치 w_true: {w_true}")
print(f"실제 절편 b_true: {b_true}")

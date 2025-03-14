import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# 초기 가중치 및 학습률 설정
w = 4
learning_rate = 0.01
epochs = 100  # 반복 횟수
avg = []

# -------------------------------------

for _ in range(epochs):
    loss = 0.0  # 매 epoch마다 손실값 초기화
    slope_sum = 0
    
    for x_train, y_train in zip(x, y):
        # 예측값 계산
        pred_val = w * x_train

        # 손실 함수 (MSE: Mean Squared Error)
        loss += (pred_val - y_train) ** 2

        # 기울기(Gradient) 계산
        slope_sum += x_train * (w * x_train - y_train)
    
    # 경사 하강법을 사용하여 w 업데이트
    w = w - learning_rate * (slope_sum / len(x))
    
    # 평균 손실값 저장 (Cost Function)
    avg.append(loss / len(x))

# 손실값 시각화
plt.plot(range(1, epochs + 1), avg)
plt.xlabel("Epochs")
plt.ylabel("Loss (Cost)")
plt.title("Gradient Descent Loss Reduction")
plt.show()

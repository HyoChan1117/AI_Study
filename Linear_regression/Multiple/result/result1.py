import numpy as np

num_features = 3  # 특성 수
num_samples = 5  # 샘플 수

np.random.seed(1)  # 랜덤 시드 설정
np.set_printoptions(suppress=True, precision=3)  # 과학적 표기법 사용 안함
X = np.random.rand(num_samples, num_features)  # 0~10 사이의 랜덤한 값 5개 생성

# h(x) = wx1 + wx2 + wx3 + b
w_true = np.random.randint(1, 10, num_features)  # 실제 가중치
b_true = np.random.randn() * 0.5  # 실제 절편

# 행렬 곱셈을 통한 y 계산
y = X @ w_true + b_true

# print(y)

# learning
w = np.random.rand(num_features)  # 초기 가중치
b = np.random.randn()  # 초기 절편
learning_rate = 0.01  # 학습률
epoches = 10000  # 에폭 수

  # print(f"w : {w}, b : {b}")
for epoch in range(epoches):
  # prediction
  predicition = X @ w + b  # 예측값
  # print(f"predicition : {predicition}")

  # error
  error = predicition - y  # 오차
  # print(f"error : {error}")
  # print(f"\n{X.T}")
  # print(f"\n{X.T @ error}")

  # gradient
  gradient = X.T @ error / num_samples # 기울기

  w = w - learning_rate * gradient  # 가중치 업데이트
  b = b - learning_rate * error.mean()  # 절편 업데이트


print(f"w_true : {w_true}, b_true : {b_true}")
print(f"w : {w}, b : {b}")





from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error  # 평균 제곱 오차
import numpy as np

np.set_printoptions(suppress=True, precision=1)
X = np.random.rand(100, 1) * 10
# H(x) = w * x + b

y = 2.5 * X + np.random.randn(100, 1) * 2
y = y.ravel()  # y를 1차원으로 변환

# 모델 생성 후 하이퍼파라미터 설정
model = SGDRegressor(
    max_iter=100, # max_iter : 최대 반복 횟수
    learning_rate='constant', # constant : 고정된 학습률
    eta0=0.001, # eta0 : 초기 학습률
    penalty=None, # penalty : 규제 항 (L2, L1, elasticnet)
    random_state=0 # random_state : 랜덤 시드
)

# 모델 학습
model.fit(X, y)

# 평가
# loss
y_pred = model.predict(X)  # 학습이 끝난 후 예측

mse = mean_squared_error(y, y_pred)  # 평균 제곱 오차
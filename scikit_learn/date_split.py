# 라이브러리 불러오기
from sklearn.model_selection import train_test_split  # train_test_split : 데이터셋을 학습용과 테스트용으로 나누는 함수
import numpy as np  # numpy : 행렬 연산을 위한 라이브러리

# 랜덤 시드 고정
np.random.seed(2)  # 랜덤 시드를 고정하여 매번 동일한 결과를 얻도록 설정

# 데이터셋 만들기
# X : 입력 값 (특징)
# 10개의 데이터 포인트, 각 데이터 포인트는 2개의 특징을 가짐
# np.random.rand(10, 2) : 0~1 사이의 랜덤한 값으로 이루어진 10x2 행렬 생성
# np.random.rand(10, 2) * 5 : 0~5 사이의 랜덤한 값으로 이루어진 10x2 행렬 생성
X = np.random.rand(10, 2) * 5

# Y : 출력 값 (정답)
# 10개의 데이터 포인트에 대한 정답을 랜덤하게 생성
# np.random.randint(0, 2, size=10) : 0~1 사이의 랜덤한 정수로 이루어진 10개의 데이터 포인트 생성
y = np.random.randint(0, 2, size=10)

# test_size : 테스트 데이터셋의 비율 (0.2 = 20%)
# train_test_split : 데이터를 학습용과 테스트용으로 나누는 함수
# random_state : 랜덤 시드 (랜덤하게 데이터를 나누되, 항상 동일하게 나눠지도록 고정)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=2)

# 데이터셋 나누기 결과 출력
# X_train : 학습용 데이터셋 (80%)
print(f"X_train.shape: {X_train.shape}")
print(X_train)

# X_test : 테스트 데이터셋 (20%)
print(f"X_test.shape: {X_test.shape}")
print(X_test)

# y_train : 학습용 데이터셋에 대한 정답 (80%)
print(f"y_train.shape: {y_train.shape}")
print(y_train)

# y_test : 테스트 데이터셋에 대한 정답 (20%)
print(f"y_test.shape: {y_test.shape}")
print(y_test)
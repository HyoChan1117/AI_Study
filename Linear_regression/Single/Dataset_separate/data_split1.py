# 라이브러리 불러오기
from sklearn.model_selection import train_test_split  # 데이터셋을 학습용과 테스트용으로 나누는 함수
import numpy as np   # 행렬 연산을 위한 라이브러리

# 랜덤 시드 고정
np.random.seed(2)

# 데이터셋 생성
# X : 입력값 (feature) - 특징 2개일 경우
X = np.random.rand(10, 2) * 5

# y : 출력값 (label)
y = np.random.randint(0, 2, size=10)

# 데이터셋 나누기
# 학습 데이터셋 : 80%, 테스트 데이터셋 : 20%
# test_size=0.2
# random_state=2
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=2)

# 데이터셋 나누기 결과 출력
# 학습용 입력 데이터
print(f"X_train.shape : {X_train.shape}")
print(X_train)

# 테스트용 입력 데이터
print(f"X_test.shape : {X_test.shape}")
print(X_test)

# 학습용 출력 데이터 (정답)
print(f"y_train.shape : {y_train.shape}")
print(y_train)

# 테스트용 출력 데이터 (정답)
print(f"y_test.shape : {y_test.shape}")
print(y_test)
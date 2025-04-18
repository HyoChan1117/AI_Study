# 매트릭스 연산의 예
import numpy as np

np.random.seed(0)

x = np.array([[1, 2], [3, 4]])
y = np.array([[2], [3]])

print(x.shape)
print(y.shape)

print(f"x:\n{x}\ny:\n{y}\nx @ y:\n{x @ y}") # 결과 : (2, 1)
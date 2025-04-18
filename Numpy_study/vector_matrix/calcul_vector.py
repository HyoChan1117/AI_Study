# 벡터 연산의 예
import numpy as np

np.random.seed(0)

x = np.random.randint(1, 4, (2, ))
y = np.random.randint(1, 4, (2, ))

print(f"x:\n{x}\ny:\n{y}\nx + y:\n{x + y}") # 결과 : 벡터
print(f"x:\n{x}\ny:\n{y}\nx - y:\n{x - y}") # 결과 : 벡터
print(f"x:\n{x}\ny:\n{y}\nx * y:\n{x * y}") # 결과 : 벡터
print(f"x:\n{x}\ny:\n{y}\nx @ y:\n{x @ y}") # 결과 : 스칼라
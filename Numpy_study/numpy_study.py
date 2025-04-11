# Numpy는 수치 계산을 위한 파이썬 라이브러리
# 행렬에 대한 연산을 지원원
import numpy as np

bar = np.zeros((2))
foo = np.zeros((3, 2))
pos = np.zeros((2, 3, 2))

print(bar.shape)
print(f"bar.shape: {bar.shape}")
print(f"foo.shape: {foo.shape}")
print(f"pos.shape: {pos.shape}")


print(bar)
print(foo)
print(pos)
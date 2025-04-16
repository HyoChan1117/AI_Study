import numpy as np

# h(x) = wx1 + wx2 + wx3 + b
# ndarray([x1, x2, x3])
# 5개의 샘플, 3개의 특성
X = np.random.rand(5, 3) * 10  # 0.0 ~ 10.0

kin = np.array(1)
bar = np.array([1, 2, 3])
foo = np.array([[1], [2], [3]])

print(f"{kin.shape}, {bar.shape}, {foo.shape}")
print(f"{type(kin)}, {type(bar)}, {type(foo)}")
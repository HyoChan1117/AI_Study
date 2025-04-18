import numpy as np

kin = np.array(1)
bar = np.array([1, 2, 3])
foo = np.array([[1], [2], [3]])

print(f"{kin.shape}, {bar.shape}, {foo.shape}")
print(f"{type(kin)}, {type(bar)}, {type(foo)}")
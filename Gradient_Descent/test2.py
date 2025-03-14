import numpy as np
import numpy as np
import random

x = np.random.rand(50, 1) * 10  # 0.0 ~ 10.0
y = [2*val + np.random.rand()*4 for val in x]

print(x)
print()
print(y)
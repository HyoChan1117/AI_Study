import numpy as np
from sklearn.preprocessing import StandardScaler

x = np.arange(0, 11)

max = x.max()
min = x.min()

values = [ (item - min) / (max - min) for item in x ]

print(values)
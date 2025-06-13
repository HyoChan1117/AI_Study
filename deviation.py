import numpy as np
from sklearn.preprocessing import StandardScaler

x = np.arange(10)

print(x.mean(), x.std())

mean = x.mean()

values = [ item - mean for item in x ]
print(values, sum(values))
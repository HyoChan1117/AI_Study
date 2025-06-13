from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 단위 벡터
one_hot = np.eye(4)

print(one_hot, "\n\n")

y_list = [0, 1, 0, 3, 2, 3]
my_list = [0, 1, 0, 3]
print(one_hot[y_list])

one_hot_value = one_hot[y_list]

# Decoding 작업
print(np.argmax(one_hot_value, axis=1))
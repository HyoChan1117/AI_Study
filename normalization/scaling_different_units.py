from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True, precision=30)  # 과학적 표기법을 사용하지 않도록 설정

values1 = np.array((160, 170, 190, 180)).reshape(-1, 1)  # 2차원 배열로 변환
values2 = np.array((400000000, 70000000, 200000000, 30000000)).reshape(-1, 1)  # 2차원 배열로 변환

scaler1 = StandardScaler()
scaler2 = StandardScaler()

# 전체 데이터셋의 평균을 0, 표준편차 1로 맞추는 역할
# 서로 다른 단위이더라도 평균과 편차는 동일하게 맞춰줌
fit_values1 = scaler1.fit_transform(values1)
fit_values2 = scaler2.fit_transform(values2)

print(fit_values1)
print(fit_values2)

# fit_values1 = scaler1.fit(values1)
# fit_values2 = scaler2.fit(values2)

# print(fit_values1.mean_, fit_values1.var_, fit_values1.scale_)
# print(fit_values2.mean_, fit_values2.var_, fit_values2.scale_)
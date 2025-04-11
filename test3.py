samples = []
y = []

w = [0.2, 0.3]
b = 0.1

gradient_w = [0.0, 0.0]
gradient_b = 0.0

# 모든 샘플 순회
for dp, y in zip(samples, y):
    # 2. 예측 값
    predict_y = w * dp + b
    
    # 1. Error : 예측 값 - 정답
    error = predict_y - y
    
    # 기울기 값 누적 -> w의 기울기 : sum(Error * each f)/샘플의 개수
    gradient_w += error * dp
    
    # b의 기울기
    gradient_b += error

# update gradient of each W
# update gradient of b


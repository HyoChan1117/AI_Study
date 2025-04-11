
samples = []
y = []

w = 0.2
b = 0.1

gradient_w = 0.0
gradient_b = 0.0

epochs = 50

for _ in range(epochs):
    for f, y_ in zip(samples, y):
        predict_y = w * f + b
        
        # Error = 예측값 - 정답값
        error = predict_y - y_
        
        # w의 기울기 : sum(Error * each f) / 샘플의 개수
        gradient_w = error * f
        
        # b의 기울기 : sum(Error) / 샘플의 개수
        gradient_b = error
        
    w = w - gradient_w  # w 업데이트
    b = b - gradient_b  # b 업데이트
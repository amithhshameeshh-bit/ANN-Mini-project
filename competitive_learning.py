import numpy as np

data = np.random.rand(100,2)

w = np.random.rand(2,2)
lr = 0.1

for epoch in range(50):
    for x in data:
        distances = np.linalg.norm(w - x, axis=1)
        winner = np.argmin(distances)
        w[winner] += lr * (x - w[winner])

print("Final Weights:", w)
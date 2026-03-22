from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500)
model.fit(X, y)

print("Accuracy:", model.score(X, y))
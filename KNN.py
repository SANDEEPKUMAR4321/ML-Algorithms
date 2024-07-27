import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

# Example usage
if __name__ == "__main__":
    # Generating synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Train the KNN model
    model = KNN(k=3)
    model.fit(X, y)
    predictions = model.predict(X)

    # Printing the accuracy
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy}")

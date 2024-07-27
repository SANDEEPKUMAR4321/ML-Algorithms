import numpy as np
from scipy.optimize import minimize

class SVM:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        m, n = X.shape
        y = y * 2 - 1  # Convert to -1, 1
        self.alpha = np.zeros(m)
        self.w = np.zeros(n)
        self.b = 0

        def loss(alpha):
            return 0.5 * np.sum(np.outer(alpha, alpha) * np.outer(y, y) * np.dot(X, X.T)) - np.sum(alpha)

        def zerofun(alpha):
            return np.dot(alpha, y)

        cons = {'type': 'eq', 'fun': zerofun}
        bounds = [(0, self.C) for _ in range(m)]
        result = minimize(loss, self.alpha, bounds=bounds, constraints=cons)
        self.alpha = result.x

        support_vectors = self.alpha > 1e-5
        self.w = np.sum(self.alpha[support_vectors][:, None] * y[support_vectors][:, None] * X[support_vectors], axis=0)
        self.b = np.mean(y[support_vectors] - np.dot(X[support_vectors], self.w))

    def predict(self, X):
        return (np.dot(X, self.w) + self.b > 0).astype(int)

# Example usage
if __name__ == "__main__":
    # Generating synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Train the SVM model
    model = SVM(C=1.0)
    model.fit(X, y)
    predictions = model.predict(X)

    # Printing the accuracy
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy}")

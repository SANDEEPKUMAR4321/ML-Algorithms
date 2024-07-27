import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Normalize X
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = (X - self.mean) / self.std
        
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        self.weights = np.zeros(X.shape[1])  # Initialize weights

        for _ in range(self.n_iters):
            y_predicted = X.dot(self.weights)  # Predictions
            error = y - y_predicted  # Error
            self.weights -= self.learning_rate * X.T.dot(error)  # Update weights
            self.bias = self.weights[0]  # Bias term

    def predict(self, X):
        X = (X - self.mean) / self.std  # Normalize new data
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        return X.dot(self.weights)

# Sample data
X = np.array([[5], [7], [8], [7.5], [2], [12], [2], [9], [4], [11], [12], [9], [6]])
y = np.array([13, 18, 21, 18, 5, 30, 4, 26, 11, 28, 31, 24, 17])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print("Slope (m):", model.weights[1])
print("Intercept (b):", model.bias)

# Make a prediction
new_data = np.array([[10]])
predicted_value = model.predict(new_data)
print("Predicted value for", new_data[0][0], ":", predicted_value[0])

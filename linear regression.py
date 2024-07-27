import numpy as np
import matplotlib.pyplot as plt
def compute_cost(X, y, weights):
    m = len(y)
    predictions = np.dot(X, weights)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        predictions = np.dot(X, weights)
        gradients = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradients
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)      
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")  
    return weights, cost_history
def plot_graph(X, y, weights):
    plt.scatter(X[:, 1], y, color='blue', marker='o', label='Training data')
    predictions = np.dot(X, weights)
    plt.plot(X[:, 1], predictions, color='red', linewidth=2, label='Linear regression')
    plt.xlabel('Feature Value')
    plt.ylabel('Target Value')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()
# Generating synthetic data
np.random.seed(0)
num_points = 100
x = np.linspace(0, 4, num_points)
y = 2 * x + 1 + np.random.normal(0, 1.5, num_points)  # true function is y = 2x + 1
X = np.stack((np.ones(num_points), x), axis=1)
initial_weights = np.zeros(X.shape[1])
# Train the linear regression model
learning_rate = 0.11
iterations = 1000
final_weights, cost_history = gradient_descent(X, y, initial_weights, learning_rate, iterations)
# Plotting the graph
plot_graph(X, y, final_weights)
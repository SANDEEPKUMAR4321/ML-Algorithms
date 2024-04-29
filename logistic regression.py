import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    error = (-y * np.log(predictions)) - ((1 - y) * np.log(1 - predictions))
    cost = sum(error) / m
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        gradients = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradients
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
        
        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    return weights, cost_history

def plot_decision_boundary(X, y, weights):
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='winter')
    
    x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_value = -(weights[0] + weights[1] * x_value) / weights[2]
    plt.plot(x_value, y_value, "r")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# Generating synthetic data
np.random.seed(0)
num_points = 100
x1 = np.random.normal(2, 1, num_points)
x2 = np.random.normal(3, 1, num_points)
y = np.zeros(num_points)

x1_2 = np.random.normal(6, 1, num_points)
x2_2 = np.random.normal(7, 1, num_points)
y_2 = np.ones(num_points)

x1 = np.concatenate((x1, x1_2))
x2 = np.concatenate((x2, x2_2))
y = np.concatenate((y, y_2))

X = np.stack((np.ones(2 * num_points), x1, x2), axis=1)
initial_weights = np.zeros(X.shape[1])

# Train the logistic regression model
learning_rate = 0.1
iterations = 10000
final_weights, cost_history = gradient_descent(X, y, initial_weights, learning_rate, iterations)

# Plotting the decision boundary
plot_decision_boundary(X, y, final_weights)

import numpy as np
class LinearRegression:
  """
  This class implements linear regression from scratch using NumPy.
  """
  def __init__(self, learning_rate=1.11, n_iters=100):  
    """
    Initializes the model with learning rate and number of iterations.
    Args:
      learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.01.
      n_iters (int, optional): The number of iterations for gradient descent. Defaults to 1000.
    """
    self.learning_rate = learning_rate
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    """
    Fits the model to the training data.
    Args:
      X (numpy.ndarray): The training data as a 2D array.
      y (numpy.ndarray): The target values as a 1D array.
    """
    
    # Add a column of ones for the bias term
    
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    self.weights = np.zeros(X.shape[1])  # Initialize weights with zeros
    print(self.weights)
    # Gradient descent loop
    for _ in range(self.n_iters):
      # Calculate predictions
      y_predicted = X.dot(self.weights)
      ##print(X)

      # Calculate errors
      error = y - y_predicted

      # Update weights and bias using gradient descent
      self.weights -= self.learning_rate * X.T.dot(error)
      self.bias = self.weights[0]  # Extract bias from weights
      #print(self.weights[0],self.weights[1])
    ##print(X)
  def predict(self, X):
    """
    Predicts the target values for new data points.
    Args:
      X (numpy.ndarray): The new data points as a 2D array.
    Returns:
      numpy.ndarray: The predicted target values.
    """
    # Add a column of ones for the bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))
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
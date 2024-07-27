
                          MACHINE LEARNING ALGORITHMS

        Linear Regression and Logistic Regression Implementation in Python
This repository contains implementations of Linear Regression and Logistic Regression algorithms in Python using NumPy and Matplotlib.

                              Table of Contents:
      1.Linear Regression:
          Introduction
          Usage
          Functions
      2.Logistic Regression:
          Introduction
          Usage
          Functions

      3.Decision Tree](#decision-tree)
          Introduction
          Usage
          Functions
  
      4.Support Vector Machine 
          Introduction
          Usage
          Functions

      5.k-Nearest Neighbors 
          Introduction
          Usage
          Functions
      6.Random Forest
          Introduction
          Usage
          Functions


                              1.LINEAR REGRESSION
                              
      Introduction : Linear Regression is a simple algorithm used for predictive modeling. 
      It assumes a linear relationship between the input variables (X) and the single output variable (y).
      
      Usage:
      import numpy as np
      import matplotlib.pyplot as plt
      
      # Generate synthetic data
      np.random.seed(0)
      num_points = 100
      x = np.linspace(0, 4, num_points)
      y = 2 * x + 1 + np.random.normal(0, 1.5, num_points)
      
      X = np.stack((np.ones(num_points), x), axis=1)
      initial_weights = np.zeros(X.shape[1])
      
      # Train the linear regression model
      learning_rate = 0.01
      iterations = 1000
      final_weights, cost_history = gradient_descent(X, y, initial_weights, learning_rate, iterations)
      
      # Plotting the graph
      plot_graph(X, y, final_weights)
      
      Functions:
      compute_cost(X, y, weights): Computes the cost for given X, y, and weights.
      gradient_descent(X, y, weights, learning_rate, iterations): Performs gradient descent to learn weights.
      plot_graph(X, y, weights): Plots the data points and the linear regression line.

      

                              2.LOGISTIC REGRESSION
                          
      Introduction:Logistic Regression is used for binary classification problems. 
      It models the probability that each input belongs to a particular class.

      Usage:
      import numpy as np
      import matplotlib.pyplot as plt
      
      # Generate synthetic data
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

      Functions:
      sigmoid(z): Computes the sigmoid of z.
      compute_cost(X, y, weights): Computes the cost for given X, y, and weights.
      gradient_descent(X, y, weights, learning_rate, iterations): Performs gradient descent to learn weights.
      plot_decision_boundary(X, y, weights): Plots the decision boundary for logistic regression.


                              3.DECISION TREE

      Introduction:Decision Trees are a type of supervised learning algorithm used for classification and regression. 
      The algorithm splits the data into subsets based on the feature values, recursively creating a tree structure.

      Usage:
      from decision_tree import DecisionTree
      # Generating synthetic data
      import numpy as np
      np.random.seed(0)
      X = np.random.rand(100, 2)
      y = (X[:, 0] + X[:, 1] > 1).astype(int)
      
      # Train the decision tree model
      model = DecisionTree(depth=3)
      model.fit(X, y)
      predictions = model.predict(X)
      
      # Printing the accuracy
      accuracy = np.mean(predictions == y)
      print(f"Accuracy: {accuracy}")

      Functions:
      fit(X, y): Fits the decision tree model to the training data.
      predict(X): Predicts the target values for the given input data.

      
                              4.SUPPORT VECTOR MACHINE  (SVM)
      Introduction:Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression. 
      It works by finding the hyperplane that best separates the classes in the feature space.

      Usage:
      from svm import SVM
      # Generating synthetic data
      import numpy as np
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

      Functions:
      fit(X, y): Fits the SVM model to the training data.
      predict(X): Predicts the target values for the given input data.

      
                              5.k-Nearest Neighbors (KNN)
      Introduction:
      k-Nearest Neighbors (KNN) is a supervised learning algorithm used for classification and regression. 
      It works by finding the k nearest neighbors of a data point and using their labels to predict the target value.

      Usage:
      from knn import KNN
      # Generating synthetic data
      import numpy as np
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

      Functions:
      fit(X, y): Fits the KNN model to the training data.
      predict(X): Predicts the target values for the given input data.

      
                              5.RANDOM FOREST
      Introduction:Random Forest is an ensemble learning method that constructs multiple decision trees during training,
      and outputs the mean prediction of the individual trees.

      Usage:
      from random_forest import RandomForest
      # Generating synthetic data
      import numpy as np
      np.random.seed(0)
      X = np.random.rand(100, 2)
      y = (X[:, 0] + X[:, 1] > 1).astype(int)
      
      # Train the random forest model
      model = RandomForest(n_estimators=10, max_depth=3)
      model.fit(X, y)
      predictions = model.predict(X)
      
      # Printing the accuracy
      accuracy = np.mean(predictions == y)
      print(f"Accuracy: {accuracy}")

      Functions:
      fit(X, y): Fits the random forest model to the training data.
      predict(X): Predicts the target values for the given input data.

      
            Requirements
                NumPy
                Matplotlib
                SciPy
                               
      License:
      This project is licensed under the MIT License. See the LICENSE file for details.

import numpy as np

class DecisionTree:
    def __init__(self, depth=1):
        self.depth = depth
        self.tree = {}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=self.depth)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(set(y)) == 1:
            return np.mean(y)
        feature, threshold = self._best_split(X, y)
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth-1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth-1)
        return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_feature, best_threshold, best_gini = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                gini = self._gini_index(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_feature, best_threshold, best_gini = feature, threshold, gini
        return best_feature, best_threshold

    def _gini_index(self, left, right):
        def gini(arr):
            m = len(arr)
            return 1 - sum((np.sum(arr == c) / m) ** 2 for c in np.unique(arr))
        return (len(left) * gini(left) + len(right) * gini(right)) / (len(left) + len(right))

    def _predict_single(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['feature']] < tree['threshold']:
                return self._predict_single(x, tree['left'])
            else:
                return self._predict_single(x, tree['right'])
        else:
            return tree

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = [DecisionTree(depth=self.max_depth) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        m, n = X.shape
        for tree in self.trees:
            indices = np.random.choice(m, m, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(tree_preds, axis=0))

# Example usage
if __name__ == "__main__":
    # Generating synthetic data
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

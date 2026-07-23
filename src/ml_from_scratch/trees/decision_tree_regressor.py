import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = None


    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        _, n_features = X.shape
        self.n_features = n_features

        self.root = self._build_tree(X, y, 0)
        return self

    def predict(self, X):
        if self.root is None:
            raise ValueError("The model must be fitted before prediction.")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features:
            raise ValueError("X must contain the same number of features used during training.")

        predictions = []

        for x in X:
            prediction = self._traverse_tree(x, self.root)
            predictions.append(prediction)

        return np.asarray(predictions)

    def _mse(self, y):
        if len(y) == 0:
            raise ValueError("Labels cannot be empty!")
        mean = np.mean(y)
        squared_errors = np.square(y - mean)
        return np.sum(squared_errors) / len(y)

    def _weighted_mse(self, y_left, y_right):
        num_left = len(y_left)
        num_right = len(y_right)
        total = num_left + num_right

        mse_left = self._mse(y_left)
        mse_right = self._mse(y_right)

        return (num_left / total) * mse_left + (num_right / total) * mse_right

    def _mse_reduction(self, y, y_left, y_right):
        parent_mse = self._mse(y)
        child_mse = self._weighted_mse(y_left, y_right)
        return parent_mse - child_mse

    def _leaf_value(self, y):
        return np.mean(y)

    def _split(self, X_column, threshold):
        cond = X_column <= threshold
        left_indices = np.where(cond)[0]
        right_indices = np.where(~cond)[0]
        return left_indices, right_indices
    
    def _candidate_thresholds(self, X_column):
        X_column = np.unique(X_column)
        thresholds = []
        for i in range(len(X_column) - 1):
            thresholds.append(np.mean([X_column[i], X_column[i + 1]]))

        return np.asarray(thresholds)

    def _best_split(self, X, y):
        best_mse_reduction = 0
        best_feature_index = -1
        best_threshold = None
        best_left_indices = []
        best_right_indices = []

        for feature_index in range(X.shape[1]):
            X_column = X[:, feature_index]
            thresholds = self._candidate_thresholds(X_column)
            for threshold in thresholds:
                left_indices, right_indices = self._split(X_column, threshold)
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                y_left = y[left_indices]
                y_right = y[right_indices]
                mse_reduction = self._mse_reduction(y, y_left, y_right)
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices
        

        return (
            best_mse_reduction,
            best_feature_index,
            best_threshold,
            best_left_indices,
            best_right_indices,
        )

    def _build_tree(self, X, y, depth):
        if self._mse(y) == 0:
            return Node(value=self._leaf_value(y))
        if len(y) < self.min_samples_split:
            return Node(value=self._leaf_value(y))
        if self.max_depth is not None:
            if depth >= self.max_depth:
                return Node(value=self._leaf_value(y))

        _ , feature_index, threshold, left_indices, right_indices = self._best_split(X, y)
        
        if feature_index == -1:
            return Node(value=self._leaf_value(y))
        
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature_index=feature_index, threshold=threshold, left=left_tree, right=right_tree)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

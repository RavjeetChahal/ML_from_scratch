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
    
class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features = n_features


    def predict(self, X):
        pass

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        sum_squared_proportions = np.sum(np.square(counts/(np.sum(counts))))
        return 1 - sum_squared_proportions


    def _weighted_impurity(self, y_left, y_right):
        num_l = len(y_left)
        num_r = len(y_right)
        total = num_l + num_r
        gini_l = self._gini(y_left)
        gini_r = self._gini(y_right)
        return ((num_l / total) * gini_l) + ((num_r / total) * gini_r)
    
    def _information_gain(self, y, y_left, y_right):
        gini_parent = self._gini(y)
        gini_after = self._weighted_impurity(y_left, y_right)
        return gini_parent - gini_after
    

    def _split(self, X_column, threshold):
        cond_matrix = X_column <= threshold
        left_indices = np.where(cond_matrix)[0]
        right_indices = np.where(~cond_matrix)[0]
        return left_indices, right_indices
    
    def _candidate_thresholds(self, X_column):
        X_column = np.unique(X_column, sorted=True)
        thresholds = []
        for i in range(len(X_column) - 1):
            thresholds.append(np.mean([X_column[i], X_column[i + 1]]))

        return np.asarray(thresholds)
    
    def _best_split(self, X, y):
        best_ig = 0
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
                ig = self._information_gain(y, y_left, y_right)
                if ig > best_ig:
                    best_ig = ig
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return (best_ig, best_feature_index, best_threshold, best_left_indices, best_right_indices)

dt = DecisionTreeClassifierScratch()
X_column = np.array([3, 3, 3])
threshold = 3.5

print(dt._candidate_thresholds(X_column))
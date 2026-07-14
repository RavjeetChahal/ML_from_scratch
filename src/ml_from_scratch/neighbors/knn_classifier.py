import numpy as np

class KNeighborsClassifierScratch:
    def __init__(self, n_neighbors=5):
        if n_neighbors <= 0:
            raise ValueError("Number of neighbors must be greater than 0!")
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    # KNN is a lazy learner, fitting only stores the training set for distance comparison during prediction
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("Data size does not match label size!")
        
        if self.n_neighbors > X.shape[0]:
            raise ValueError("Number of neighbors must not exceed number of examples!")
        
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        self._check_is_fitted()

        predictions = []
        X = np.asarray(X)

        for x in X:
            prediction = self._predict_one(x)
            predictions.append(prediction)

        return np.asarray(predictions)

    # Returns one distance from x to each stored training example
    def _euclidean_distance(self, x):
        diff = self.X_train - x
        squared_diff = np.square(diff)
        squared_distances = np.sum(squared_diff, axis=1)
        distances = np.sqrt(squared_distances)
        return distances
    
    def _predict_one(self, x):
        distances = self._euclidean_distance(x)
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        labels, counts = np.unique(nearest_labels, return_counts=True)
        majority = np.argmax(counts)
        prediction = labels[majority]
        return prediction
    
    def _check_is_fitted(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fit to data before predicting!")
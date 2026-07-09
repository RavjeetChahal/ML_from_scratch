import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-8):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.losses = []

    # Learn weights and bias by minimizing mean squared error loss with gradient descent
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []

        for i in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

            if i > 0:
                loss_change = np.abs(self.losses[i - 1] - loss)
                if loss_change < self.tolerance:
                    break

            error = y_pred - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights = self.weights - dw * self.learning_rate
            self.bias = self.bias - db * self.learning_rate
        return self

    # Checking to ensure model is fit before we can use it for predictions
    def _check_is_fitted(self):
        if self.weights is None or self.bias is None:
            raise ValueError("Fit method must be called to learn weights and bias first!")
    
    # Mean squared error: keeps errors positive and penalizes larger errors more
    def _compute_loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    
    # Predict continuous target values using the learned linear model
    def predict(self, X):
        self._check_is_fitted()
        
        X = np.asarray(X)
        
        return X @ self.weights + self.bias
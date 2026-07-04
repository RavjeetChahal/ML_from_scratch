import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-8):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
        self.tolerance = tolerance

    # Convert raw logits into probabilities between 0 and 1
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Learn weights and bias by minimizing binary cross-entropy with gradient descent
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for i in range(self.n_iterations):
            z = X @ self.weights + self.bias
            p = self._sigmoid(z)
            loss = self._compute_loss(y, p)
            self.losses.append(loss)
            if i > 0:
                loss_change = self.losses[i - 1] - loss
                if loss_change < self.tolerance:
                    break

            error = p - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            new_weights = self.weights - dw * self.learning_rate
            new_bias = self.bias - db * self.learning_rate

            self.weights = new_weights
            self.bias = new_bias
        
        return self

    # Method to compute binary cross-entropy loss
    def _compute_loss(self, y, p):
        # Using np.clip to ensure the predicted probability is not exactly 0 or 1, otherwise we would get log(0) which results in -inf
        # which can break loss
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss
    
    # Method to ensure fit() has been called
    def _check_is_fitted(self):
        if self.weights is None or self.bias is None:
            raise ValueError("Fit method must be called to learn weights and bias first!")
        
    # Return predicted probabilities for one or more examples
    def predict_proba(self, X):
        self._check_is_fitted()

        z = X @ self.weights + self.bias
        p = self._sigmoid(z)
        return p
    
    # Return class predictions for one or more examples
    def predict(self, X):
        self._check_is_fitted()
        
        proba = self.predict_proba(X)
        predictions = proba >= .5
        return predictions * 1

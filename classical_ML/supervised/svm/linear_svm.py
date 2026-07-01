import numpy as np
from typing import Optional

class LinearSVM:
    """
    Support Vector Machine classifier using Mini Batch Gradient Descent
    Optimizes the primal formulation of SVM using hinge loss:
    L = (lambda / 2) * || W ||^2 + 1/N * sum(1 - y_i * (W @ x_i + b))
    """
    def __init__(
        self,
        learning_rate : float = 0.001,
        batch_size : int = 32,
        num_epochs : int = 1000,
        lambda_param : float = 0.01
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_param = lambda_param
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None


    def fit(
        self,
        X : np.ndarray,
        y : np.ndarray
    ):
        """
        SVM model training
        X : dimension (n_sample, n_features)
        y : dimension (n_samples)
        """
        n_samples, n_features = X.shape

        #convert the labels to {1, -1}
        y_label = np.where(y <= 0, -1, 1) 

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.num_epochs):
            permutations = np.random.permutation(n_samples)#shuffling data every epoch
            X_permuted = X[permutations]
            y_permuted = y_label[permutations]
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_permuted[i : i + self.batch_size]
                y_batch = y_permuted[i : i + self.batch_size]

                # Calculate margins
                margins = y_batch * (np.dot(X_batch, self.w) + self.b)
                violating_indices = margins < 1

                # grad_init
                dw = self.lambda_param * self.w
                db = 0.0

                # gradient inclusion for the datapoints violating the constraints
                if np.any(violating_indices):
                    X_sv = X_batch[violating_indices]
                    y_sv = y_batch[violating_indices]
                    curr_bs = len(X_sv)
                    
                    dw -= np.dot(y_sv, X_sv) / curr_bs
                    db -= np.sum(y_sv) / curr_bs

                # Update weights
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

                

    def predict(
        self,
        X : np.ndarray
    ):
        """
        predicts classlabels for samples in X
        Args: X : samples to predict
        returns : Predicted class labels {-1, 1}.
        """
        val = np.dot(X, self.w) + self.b
        return np.sign(val)

        
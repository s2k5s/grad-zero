import numpy as np
from classical_ML.supervised.tree_based.decision_tree import DecisionTree
class RandomForest:
    def __init__(self, num_trees, max_depth, min_samples_leaf, max_features, feature_type=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.max_features = max_features
        self.feature_type = feature_type

    def sampler(self, X, y):
        num_samples = X.shape[0]
        indices = np.random.choice(num_samples, num_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        for i in range(self.num_trees):
            tree = DecisionTree(
                self.max_depth,
                self.min_samples_leaf,
                feature_type=self.feature_type,
                max_features=self.max_features,
                eval_type="gini",
                random_forest=True,
            )
            X_sampled, y_sampled = self.sampler(X, y)
            tree.fit(X_sampled, y_sampled)
            self.trees.append(tree)

    def accumulate(self, y):
        unique, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts) 
        pred = unique[idx]
        return pred
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees]) #(num_trees, num_samples)
        tree_preds = tree_preds.T #(num_samples, num_trees)

        pred = np.array([self.accumulate(row) for row in tree_preds]) #(num_samples,)
        return pred


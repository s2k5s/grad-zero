import numpy as np
class TreeNode:
    """
    a node is eithere a leaf or a branch, leaf holds the value and brach holds the routing logic
    """
    def __init__(self, feature=None, feature_type=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.feature_type = feature_type
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.value is not None
    


class DecisionTree:
    def __init__(self, max_depth, min_samples_leaf, feature_type=None, max_features=None, eval_type="gini", random_forest=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.feature_type = feature_type
        self.max_features = max_features
        self.eval_type = eval_type
        self.random_forest = random_forest
        self.root = None
    
    def gini_impurity(self, y):
        """
        if we pick two elements from a node with replacement, what is the probability they belong to different classes
        for a pure node --> this prob is 0
        for an evenly distributed node --> this prob is max
        we want lowest possible diversity hence we want lowest possible gini
        gini = 1 - (p1^2 + p2^2 .... + pn^2)
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / np.sum(counts)
        prob_squared = probs * probs
        squared_sum = np.sum(prob_squared)
        gini = 1 - squared_sum
        return gini
    
    def entropy(self, y):
        """
        information needed to accurately describe a sample. 
        for a pure node --> entropy = 0 as single class
        for a node with uniform distribution --> max entropy 
        entropy : - Σ pi * log(pi)
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / np.sum(counts)
        logprobs = np.log2(probs)
        prob_logprob = probs * logprobs
        weighted_logprob_sum = np.sum(prob_logprob)
        entropy = - weighted_logprob_sum
        return entropy
    
    def impurity(self, y):
        if self.eval_type == "gini":
            return self.gini_impurity(y)
        return self.entropy(y)

    def numerical_split(self, x, y):
        """
        handling the splitting of numerical features
        """
        parent_impurity = self.impurity(y)
        # sort based on the feature
        sorting_order = np.argsort(x)
        x_sorted = x[sorting_order]
        y_sorted = y[sorting_order]
        unique_x_sorted = np.unique(x_sorted)
        midpoints = (unique_x_sorted[1:] + unique_x_sorted[:-1]) / 2
        best_info_gain = float('-inf')
        threshold = 0

        for midpoint in midpoints:
            mask_smaller = x_sorted < midpoint
            y_smaller = y_sorted[mask_smaller]
            y_larger = y_sorted[~mask_smaller]
            weight_smaller = np.sum(mask_smaller)
            weight_larger = len(y) - weight_smaller

            impurity_smaller = self.impurity(y_smaller)
            impurity_larger = self.impurity(y_larger)

            avg_children_impurity = (impurity_smaller * weight_smaller + impurity_larger * weight_larger) / len(y)
            info_gain = parent_impurity - avg_children_impurity
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                threshold = midpoint


        return threshold, best_info_gain
    
    def categorical_split(self, x, y):
        """
        handling the splitting of categorical features
        """

        parent_impurity = self.impurity(y)
        unique = np.unique(x)
        best_info_gain = float('-inf')
        category = None
        for categ in unique:
            positive_mask = (x == categ)
            y_positive = y[positive_mask]
            y_negative= y[~positive_mask]
            weight_positive = len(y_positive)
            weight_negative = len(y_negative)

            impurity_positive = self.impurity(y_positive)
            impurity_negative = self.impurity(y_negative)

            avg_children_impurity = (weight_positive * impurity_positive + weight_negative * impurity_negative) / len(y)
            info_gain = parent_impurity - avg_children_impurity

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                category = categ

        return category, best_info_gain


        





    def build_tree(self, X, y, depth):
        """
        recursively builds a decision tree
        """

        if self.random_forest:
            if self.max_features is None:
                sampling_feature_dim = X.shape[1]
            elif self.max_features == "sqrt":
                sampling_feature_dim = int(np.sqrt(X.shape[1]))
            else:
                sampling_feature_dim = X.shape[1]
        else:
            sampling_feature_dim = X.shape[1]


        unique = np.unique(y)
        if depth == self.max_depth or len(y) <= self.min_samples_leaf or len(unique) == 1:
            uniques, counts = np.unique(y, return_counts=True)
            idx = np.argmax(counts)
            val = uniques[idx]
            return TreeNode(
                value=val 
            )
        best_feature, feature_type, best_score, threshold = None, None, float('-inf'), None
        
        feature_indices = np.random.choice(X.shape[1], sampling_feature_dim, replace=False)

        for i in feature_indices:
            curr_feature = X[:, i]
            if self.feature_type[i] == "Numerical":
                split_threshold, score = self.numerical_split(curr_feature, y)
                if best_score < score:
                    best_feature = i
                    best_score = score
                    feature_type = "Numerical"
                    threshold = split_threshold
            else:
                split_class, score = self.categorical_split(curr_feature, y)
                if best_score < score:
                    best_feature = i
                    best_score = score
                    feature_type = "Categorical"
                    threshold = split_class

        
        if best_feature is None or best_score <=0:
            uniques, counts = np.unique(y, return_counts=True)
            idx = np.argmax(counts)
            val = uniques[idx]
            return TreeNode(
                value=val 
            )

        mask = None
        if feature_type == "Numerical":
            mask = X[:, best_feature] < threshold
        else:
            mask = X[:, best_feature] == threshold
        left_subtree = self.build_tree(X[mask], y[mask], depth+1)
        right_subtree = self.build_tree(X[~mask], y[~mask], depth+1)
        return TreeNode(
            feature=best_feature,
            feature_type=feature_type,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree
        )


    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)
        return self
    
    def traverse(self, X, root):
        if root.is_leaf():
            return root.value
        feature_type = root.feature_type
        feature = root.feature
        if feature_type == "Numerical":
            if X[feature] < root.threshold:
                return self.traverse(X, root.left)
            else:
                return self.traverse(X, root.right)
            
        else:
            if X[feature] == root.threshold:
                return self.traverse(X, root.left)
            else:
                return self.traverse(X, root.right)
            
        


    def predict(self, X):
        """
        batch prediction
        """
        y_pred = np.array([self.traverse(X_row, self.root) for X_row in X])
        return y_pred




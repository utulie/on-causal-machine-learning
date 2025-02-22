import numpy as np
from collections import defaultdict

# Define causal tree node
class TreeNode:
    def __init__(self, effect=None, feature=None, threshold=None, left=None, right=None):
        self.effect = effect  # Treatment effect at the leaf node
        self.feature = feature  # Index of the splitting feature
        self.threshold = threshold  # Splitting threshold
        self.left = left  # Left subtree
        self.right = right  # Right subtree

# Causal tree construction function
def build_causal_tree(X, W, Y, min_samples_leaf=20, max_depth=3, current_depth=0):
    """
    X: feature (n_samples, n_features)
    W: treatment (0/1)
    Y: outcome
    """
    n_samples, n_features = X.shape
    
    # Termination condition
    if (n_samples < min_samples_leaf) or (current_depth >= max_depth):
        effect = _calculate_effect(Y[W==1], Y[W==0])
        return TreeNode(effect=effect)
    
    # Find the best split
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_idx = X[:, feature] <= threshold
            right_idx = ~left_idx
            
            # Calculate the gain in treatment effect variance
            gain = _effect_variance_gain(Y[left_idx], W[left_idx], Y[right_idx], W[right_idx])
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    # Recursively build subtrees
    if best_gain > 0:
        left_idx = X[:, best_feature] <= best_threshold
        left = build_causal_tree(X[left_idx], W[left_idx], Y[left_idx], 
                                min_samples_leaf, max_depth, current_depth+1)
        right = build_causal_tree(X[~left_idx], W[~left_idx], Y[~left_idx], 
                                 min_samples_leaf, max_depth, current_depth+1)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)
    else:
        effect = _calculate_effect(Y[W==1], Y[W==0])
        return TreeNode(effect=effect)

def _calculate_effect(y_treated, y_control):
    """Calculate treatment effect (ATE)"""
    if len(y_treated) == 0 or len(y_control) == 0:
        return 0.0
    return np.mean(y_treated) - np.mean(y_control)

def _effect_variance_gain(y_left, w_left, y_right, w_right):
    """Variance gain of treatment effect difference"""
    effect_left = _calculate_effect(y_left[w_left==1], y_left[w_left==0])
    effect_right = _calculate_effect(y_right[w_right==1], y_right[w_right==0])
    return (effect_left - effect_right)**2

# Causal forest
class CausalForest:
    def __init__(self, n_trees=50, min_samples_leaf=20, max_depth=5):
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, W, Y):
        """Train the forest"""
        for _ in range(self.n_trees):
            # Bootstrap sampling
            n_samples = X.shape[0]
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]
            W_boot = W[idx]
            Y_boot = Y[idx]
            
            # Build a single tree
            tree = build_causal_tree(X_boot, W_boot, Y_boot, 
                                    self.min_samples_leaf, self.max_depth)
            self.trees.append(tree)
    
    def predict_effect(self, X):
        """Predict treatment effect"""
        effects = []
        for x in X:
            tree_effects = []
            for tree in self.trees:
                node = tree
                while node.left:  # Traverse the tree until reaching a leaf node
                    if x[node.feature] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                tree_effects.append(node.effect)
            effects.append(np.mean(tree_effects))
        return np.array(effects)

# generate simulated data
def generate_data(n_samples=1000, true_effect_type='linear'):
    np.random.seed(42)
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    W = np.random.binomial(1, 0.5, size=n_samples)
    
    # True treatment effect function
    if true_effect_type == 'nonlinear':
        tau = 2 * X[:, 0] * np.exp(-X[:, 1]**2)  
    else:
        tau = 3 * X[:, 0] + 2 * X[:, 1]  
    
    Y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + W * tau + np.random.normal(0, 0.2, size=n_samples)
    return X, W, Y, tau

# Evaluation function
def evaluate_model(true_effect, pred_effect):
    mse = np.mean((true_effect - pred_effect)**2)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"True Effect Range: [{np.min(true_effect):.2f}, {np.max(true_effect):.2f}]")
    print(f"Predicted Effect Range: [{np.min(pred_effect):.2f}, {np.max(pred_effect):.2f}]")


if __name__ == "__main__":
    # Generate data
    X, W, Y, true_tau = generate_data(n_samples=2000, true_effect_type='nonlinear')
    
    # Train causal forest
    cf = CausalForest(n_trees=50, min_samples_leaf=20, max_depth=5)
    cf.fit(X, W, Y)
    
    # Predict treatment effect
    pred_tau = cf.predict_effect(X)
    
    # Evaluate performance
    evaluate_model(true_tau, pred_tau)

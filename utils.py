"""
Utility functions for data preprocessing and encoding.
"""

import numpy as np


def one_hot_encode(y, num_classes):
    """
    Convert class labels to one-hot encoded matrix.
    
    Args:
        y: Array of class labels, shape (m,) with values 0 to num_classes-1
        num_classes: Number of classes
    
    Returns:
        one_hot: One-hot encoded matrix, shape (m, num_classes)
        Example: y=[0, 1, 2, 0] -> [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]
    """
    m = len(y)
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot


def normalize_data(X):
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        X: Feature matrix, shape (m, n)
    
    Returns:
        X_normalized: Standardized features
        mean: Per-feature mean (for later denormalization if needed)
        std: Per-feature standard deviation
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    X_normalized = (X - mean) / std
    
    return X_normalized, mean, std


def train_test_split(X, y, test_ratio=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix, shape (m, n)
        y: Label array, shape (m,)
        test_ratio: Fraction of data to use for testing (e.g., 0.2 for 80/20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
    np.random.seed(random_state)
    
    m = len(X)
    test_size = int(m * test_ratio)
    
    # Randomly shuffle indices
    indices = np.random.permutation(m)
    
    # Split
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test utilities
    y_test = np.array([0, 1, 2, 0, 1])
    one_hot = one_hot_encode(y_test, num_classes=3)
    print("One-hot encoding test:")
    print(one_hot)
    
    X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    X_norm, mean, std = normalize_data(X_test)
    print(f"\nNormalization test:")
    print(f"Original mean: {np.mean(X_test, axis=0)}")
    print(f"Normalized mean: {np.mean(X_norm, axis=0)}")
    print(f"Original std: {np.std(X_test, axis=0)}")
    print(f"Normalized std: {np.std(X_norm, axis=0)}")

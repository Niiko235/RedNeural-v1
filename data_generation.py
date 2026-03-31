"""
Data generation module for 3-class classification problem.
Creates synthetic 3D data with three well-separated clusters.
"""

import numpy as np


def generate_3d_classification_data(num_samples_per_class=100, random_state=42):
    """
    Generate synthetic 3D data with 3 well-separated clusters (classes).
    
    Args:
        num_samples_per_class: Number of samples per class
        random_state: Random seed for reproducibility
    
    Returns:
        X: Array of shape (num_samples*3, 3) - features [x, y, z]
        y: Array of shape (num_samples*3,) - class labels [0, 1, 2]
    """
    np.random.seed(random_state)
    
    # Define cluster centers (3D points)
    center_class_0 = np.array([-2.0, -2.0, -2.0])
    center_class_1 = np.array([2.0, 2.0, 2.0])
    center_class_2 = np.array([2.0, -2.0, 0.0])
    
    # Generate data with Gaussian noise around cluster centers
    std_dev = 0.8
    
    class_0 = np.random.normal(center_class_0, std_dev, (num_samples_per_class, 3))
    class_1 = np.random.normal(center_class_1, std_dev, (num_samples_per_class, 3))
    class_2 = np.random.normal(center_class_2, std_dev, (num_samples_per_class, 3))
    
    # Combine features and labels
    X = np.vstack([class_0, class_1, class_2])
    y = np.hstack([
        np.zeros(num_samples_per_class, dtype=int),
        np.ones(num_samples_per_class, dtype=int),
        np.full(num_samples_per_class, 2, dtype=int)
    ])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


if __name__ == "__main__":
    # Test data generation
    X, y = generate_3d_classification_data(num_samples_per_class=100)
    print(f"Generated data shape: X={X.shape}, y={y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"\nFirst 5 samples:\n{X[:5]}")
    print(f"First 5 labels: {y[:5]}")

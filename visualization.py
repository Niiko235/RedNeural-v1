"""
Visualization module for neural network results.
Plots training curves, decision boundaries, and accuracy comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_training_curves(networks_dict, title="Training Loss Comparison"):
    """
    Plot training loss curves for multiple networks.
    
    Args:
        networks_dict: Dictionary with activation names as keys and network objects as values
                      e.g., {'sigmoid': net_sigmoid, 'tanh': net_tanh}
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    for activation_name, network in networks_dict.items():
        history = network.training_history
        plt.plot(history['epoch'], history['loss'], label=activation_name.capitalize(), linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt


def plot_accuracy_comparison(networks_dict, X_test, y_test_onehot, title="Accuracy Comparison"):
    """
    Plot training and test accuracy for multiple networks.
    
    Args:
        networks_dict: Dictionary with activation names as keys and network objects as values
        X_test: Test features
        y_test_onehot: One-hot encoded test labels
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    activations = list(networks_dict.keys())
    train_accs = []
    test_accs = []
    
    for activation_name, network in networks_dict.items():
        train_acc = network.training_history['train_accuracy'][-1]
        test_acc = network.accuracy(X_test, y_test_onehot)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    x_pos = np.arange(len(activations))
    width = 0.35
    
    plt.bar(x_pos - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    plt.bar(x_pos + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    
    plt.xlabel('Activation Function', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [a.capitalize() for a in activations])
    plt.ylim([0, 1.05])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
        plt.text(i - width/2, train_acc + 0.02, f'{train_acc:.3f}', ha='center', fontsize=10)
        plt.text(i + width/2, test_acc + 0.02, f'{test_acc:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    return plt


def plot_decision_boundary_2d(network, X_test, y_test, title="Decision Boundary"):
    """
    Plot 2D decision boundary (using first 2 features or PCA).
    
    Args:
        network: Trained neural network
        X_test: Test features
        y_test: True class labels (not one-hot)
        title: Title for the plot
    """
    # Use first 2 features
    X_2d = X_test[:, :2]
    
    # Create mesh grid
    min_x, max_x = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    min_y, max_y = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 200),
                         np.linspace(min_y, max_y, 200))
    
    # Prepare mesh points (add zero for z dimension)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_3d = np.column_stack([mesh_points, np.zeros(mesh_points.shape[0])])
    
    # Normalize using same stats (if available)
    # For simplicity, we'll predict on the mesh as-is
    Z = network.predict(mesh_points_3d)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, levels=2, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.3)
    plt.contour(xx, yy, Z, levels=2, colors='black', linewidths=1, alpha=0.4)
    
    # Plot data points
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for class_idx in range(3):
        class_points = X_2d[y_test == class_idx]
        plt.scatter(class_points[:, 0], class_points[:, 1], 
                   c=colors[class_idx], label=f'Class {class_idx}', 
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Feature 1 (x)', fontsize=12)
    plt.ylabel('Feature 2 (y)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    return plt


def plot_confusion_matrix(network, X_test, y_test, title="Confusion Matrix"):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        network: Trained neural network
        X_test: Test features
        y_test: True class labels (not one-hot)
        title: Title for the plot
    """
    predictions = network.predict(X_test)
    
    # Build confusion matrix
    num_classes = 3
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_test, predictions):
        cm[true_label, pred_label] += 1
    
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    im = plt.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add labels and text
    plt.xticks(range(num_classes), [f'Pred {i}' for i in range(num_classes)])
    plt.yticks(range(num_classes), [f'True {i}' for i in range(num_classes)])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = plt.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=14, fontweight='bold')
    
    plt.colorbar(im, label='Count')
    plt.tight_layout()
    
    return plt


def plot_all_results(networks_dict, X_train, y_train, X_test, y_test, y_test_onehot, activation_name):
    """
    Create a comprehensive figure with multiple subplots.
    
    Args:
        networks_dict: Dictionary with network objects
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels (not one-hot)
        y_test_onehot: Test labels (one-hot encoded)
        activation_name: Name of activation to display
    """
    network = networks_dict[activation_name]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Neural Network Results - {activation_name.upper()} Activation', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Training loss
    history = network.training_history
    axes[0, 0].plot(history['epoch'], history['loss'], linewidth=2, color='#FF6B6B')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training accuracy
    axes[0, 1].plot(history['epoch'], history['train_accuracy'], linewidth=2, color='#4ECDC4')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Decision boundary (2D projection)
    X_2d = X_test[:, :2]
    min_x, max_x = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    min_y, max_y = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 100),
                         np.linspace(min_y, max_y, 100))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_3d = np.column_stack([mesh_points, np.zeros(mesh_points.shape[0])])
    
    Z = network.predict(mesh_points_3d)
    Z = Z.reshape(xx.shape)
    
    axes[1, 0].contourf(xx, yy, Z, levels=2, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.3)
    axes[1, 0].contour(xx, yy, Z, levels=2, colors='black', linewidths=1, alpha=0.4)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for class_idx in range(3):
        class_points = X_2d[y_test == class_idx]
        axes[1, 0].scatter(class_points[:, 0], class_points[:, 1], 
                          c=colors[class_idx], label=f'Class {class_idx}',
                          s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    axes[1, 0].set_xlabel('Feature 1 (x)', fontsize=11)
    axes[1, 0].set_ylabel('Feature 2 (y)', fontsize=11)
    axes[1, 0].set_title('Decision Boundary (2D Projection)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    
    # Plot 4: Confusion matrix
    predictions = network.predict(X_test)
    cm = np.zeros((3, 3), dtype=int)
    for true_label, pred_label in zip(y_test, predictions):
        cm[true_label, pred_label] += 1
    
    im = axes[1, 1].imshow(cm, cmap='Blues', aspect='auto')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels([f'Pred {i}' for i in range(3)])
    axes[1, 1].set_yticklabels([f'True {i}' for i in range(3)])
    axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
    axes[1, 1].set_ylabel('True Label', fontsize=11)
    axes[1, 1].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            text = axes[1, 1].text(j, i, cm[i, j], ha="center", va="center",
                                  color="white" if cm[i, j] > cm.max() / 2 else "black",
                                  fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 1], label='Count')
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully.")

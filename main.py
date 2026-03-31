"""
Main orchestration script for training and evaluating neural networks.
Trains networks with both sigmoid and tanh activation functions and compares results.
"""

import numpy as np
from data_generation import generate_3d_classification_data
from utils import one_hot_encode, normalize_data, train_test_split
from neural_network import NeuralNetwork
from visualization import (
    plot_training_curves,
    plot_accuracy_comparison,
    plot_decision_boundary_2d,
    plot_confusion_matrix,
    plot_all_results
)
import matplotlib.pyplot as plt


def main():
    """
    Main training and evaluation pipeline.
    """
    print("=" * 70)
    print("Neural Network from Scratch - Sigmoid vs Tanh Comparison")
    print("=" * 70)
    
    # ========================================================================
    # Step 1: Generate and prepare data
    # ========================================================================
    print("\n[1/6] Generating synthetic 3D classification data...")
    X, y = generate_3d_classification_data(num_samples_per_class=100, random_state=42)
    print(f"  Generated: {X.shape[0]} samples with {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Normalize data
    print("\n[2/6] Normalizing features (zero mean, unit variance)...")
    X_normalized, mean, std = normalize_data(X)
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    
    # Train-test split
    print("\n[3/6] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_ratio=0.2, random_state=42
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    
    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train, num_classes=3)
    y_test_onehot = one_hot_encode(y_test, num_classes=3)
    
    # ========================================================================
    # Step 2: Train networks with both activation functions
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING PHASE: Comparing Sigmoid vs Tanh")
    print("=" * 70)
    
    networks = {}
    activations = ['sigmoid', 'tanh']
    
    for activation in activations:
        print(f"\n[4/6] Training network with {activation.upper()} activation...")
        print("-" * 70)
        
        # Create network
        network = NeuralNetwork(
            input_size=3,
            hidden_size=16,
            output_size=3,
            learning_rate=0.01,
            activation=activation
        )
        
        print(f"Network architecture: 3 -> 16 ({activation}) -> 3")
        
        # Train
        network.train(
            X_train, y_train_onehot,
            epochs=300,
            batch_size=None,  # Full batch gradient descent
            verbose=True
        )
        
        networks[activation] = network
        
        # Evaluate
        train_acc = network.accuracy(X_train, y_train_onehot)
        test_acc = network.accuracy(X_test, y_test_onehot)
        final_loss = network.training_history['loss'][-1]
        
        print(f"\nFinal Results ({activation}):")
        print(f"  Final Loss:       {final_loss:.6f}")
        print(f"  Train Accuracy:   {train_acc:.4f} ({int(train_acc*len(y_train))}/{len(y_train)})")
        print(f"  Test Accuracy:    {test_acc:.4f} ({int(test_acc*len(y_test))}/{len(y_test)})")
    
    # ========================================================================
    # Step 3: Compare results
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Sigmoid vs Tanh")
    print("=" * 70)
    
    print("\nActivation Function Comparison:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Sigmoid':<20} {'Tanh':<20}")
    print("-" * 70)
    
    for activation in activations:
        network = networks[activation]
        train_acc = network.accuracy(X_train, y_train_onehot)
        test_acc = network.accuracy(X_test, y_test_onehot)
        final_loss = network.training_history['loss'][-1]
        
        if activation == 'sigmoid':
            print(f"{'Train Accuracy':<30} {train_acc:<20.4f}", end='')
        else:
            print(f"{train_acc:<20.4f}")
        
        if activation == 'sigmoid':
            print(f"{'Test Accuracy':<30} {test_acc:<20.4f}", end='')
        else:
            print(f"{test_acc:<20.4f}")
        
        if activation == 'sigmoid':
            print(f"{'Final Loss':<30} {final_loss:<20.6f}", end='')
        else:
            print(f"{final_loss:<20.6f}")
    
    # ========================================================================
    # Step 4: Visualize results
    # ========================================================================
    print("\n[5/6] Generating visualizations...")
    
    # Plot 1: Training loss comparison
    loss_fig = plot_training_curves(networks, title="Training Loss Comparison: Sigmoid vs Tanh")
    loss_fig.savefig('c:\\Programming\\RedNeural-v1\\results_loss_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: results_loss_comparison.png")
    
    # Plot 2: Accuracy comparison
    acc_fig = plot_accuracy_comparison(
        networks, X_test, y_test_onehot,
        title="Accuracy Comparison: Sigmoid vs Tanh"
    )
    acc_fig.savefig('c:\\Programming\\RedNeural-v1\\results_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: results_accuracy_comparison.png")
    
    # Plot 3 & 4: Detailed results for each activation
    for activation in activations:
        fig = plot_all_results(
            networks, X_train, y_train, X_test, y_test, y_test_onehot,
            activation
        )
        filename = f'c:\\Programming\\RedNeural-v1\\results_detailed_{activation}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: results_detailed_{activation}.png")
    
    # ========================================================================
    # Summary and insights
    # ========================================================================
    print("\n[6/6] Training complete!")
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    sigmoid_train_acc = networks['sigmoid'].accuracy(X_train, y_train_onehot)
    sigmoid_test_acc = networks['sigmoid'].accuracy(X_test, y_test_onehot)
    sigmoid_loss = networks['sigmoid'].training_history['loss'][-1]
    
    tanh_train_acc = networks['tanh'].accuracy(X_train, y_train_onehot)
    tanh_test_acc = networks['tanh'].accuracy(X_test, y_test_onehot)
    tanh_loss = networks['tanh'].training_history['loss'][-1]
    
    print(f"\n1. Convergence Speed:")
    print(f"   - Sigmoid: Loss converged to {sigmoid_loss:.6f}")
    print(f"   - Tanh:    Loss converged to {tanh_loss:.6f}")
    print(f"   → Tanh typically converges faster due to symmetric activation [-1, 1]")
    
    print(f"\n2. Final Performance:")
    print(f"   - Sigmoid: Train={sigmoid_train_acc:.4f}, Test={sigmoid_test_acc:.4f}")
    print(f"   - Tanh:    Train={tanh_train_acc:.4f}, Test={tanh_test_acc:.4f}")
    
    better_activation = 'tanh' if tanh_test_acc > sigmoid_test_acc else 'sigmoid'
    print(f"   → {better_activation.upper()} achieved better test accuracy")
    
    print(f"\n3. Activation Functions Explained:")
    print(f"   - Sigmoid:  σ(z) = 1/(1+e^-z), outputs ∈ [0,1]")
    print(f"              Derivative: σ(z)·(1-σ(z))")
    print(f"   - Tanh:     tanh(z) = (e^z - e^-z)/(e^z + e^-z), outputs ∈ [-1,1]")
    print(f"              Derivative: 1 - tanh(z)²")
    
    print(f"\n4. Why Tanh Often Performs Better:")
    print(f"   - Centered around 0 (symmetric)")
    print(f"   - Prevents vanishing gradient problem better")
    print(f"   - Faster convergence in practice")
    
    print(f"\n5. Results Saved:")
    print(f"   - results_loss_comparison.png")
    print(f"   - results_accuracy_comparison.png")
    print(f"   - results_detailed_sigmoid.png")
    print(f"   - results_detailed_tanh.png")
    
    print("\n" + "=" * 70)
    print("Visualization figures displayed. Press any key to close...")
    
    plt.show()


if __name__ == "__main__":
    main()

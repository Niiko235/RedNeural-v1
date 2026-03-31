"""
Activation functions and their derivatives for neural network hidden layer.
Implements sigmoid and tanh with proper derivative computation for backpropagation.
"""

import numpy as np


# ============================================================================
# Sigmoid Activation Function
# ============================================================================

def sigmoid(Z):
    """
    Sigmoid activation function.
    
    Args:
        Z: Pre-activation (logits), can be scalar or array
    
    Returns:
        A: Sigmoid(Z) = 1 / (1 + exp(-Z)), values in (0, 1)
    """
    # Clip to prevent overflow in exp
    Z_clipped = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z_clipped))


def sigmoid_derivative(Z):
    """
    Derivative of sigmoid with respect to Z.
    
    Args:
        Z: Pre-activation (logits)
    
    Returns:
        dA/dZ: Derivative = sigmoid(Z) * (1 - sigmoid(Z))
    
    Note: For efficiency in backprop, if you already have A = sigmoid(Z),
          you can compute derivative as A * (1 - A) instead of calling this.
    """
    sig = sigmoid(Z)
    return sig * (1 - sig)


# ============================================================================
# Tanh (Hyperbolic Tangent) Activation Function
# ============================================================================

def tanh(Z):
    """
    Tanh (hyperbolic tangent) activation function.
    
    Args:
        Z: Pre-activation (logits), can be scalar or array
    
    Returns:
        A: Tanh(Z), values in (-1, 1)
    """
    return np.tanh(Z)


def tanh_derivative(Z):
    """
    Derivative of tanh with respect to Z.
    
    Args:
        Z: Pre-activation (logits)
    
    Returns:
        dA/dZ: Derivative = 1 - tanh(Z)^2
    
    Note: For efficiency in backprop, if you already have A = tanh(Z),
          you can compute derivative as 1 - A^2 instead of calling this.
    """
    tanh_z = np.tanh(Z)
    return 1 - tanh_z ** 2


# ============================================================================
# Softmax Activation Function (for output layer)
# ============================================================================

def softmax(Z):
    """
    Softmax activation function for multi-class output layer.
    Converts logits to probabilities across classes.
    
    Args:
        Z: Pre-activation matrix, shape (m, num_classes)
    
    Returns:
        A: Softmax(Z), shape (m, num_classes), each row sums to 1
    """
    # Subtract max per row for numerical stability
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# ============================================================================
# Utility function to get activation and derivative by name
# ============================================================================

def get_activation(activation_name):
    """
    Get activation function and its derivative by name.
    
    Args:
        activation_name: 'sigmoid' or 'tanh'
    
    Returns:
        (activation_func, derivative_func): Tuple of activation and derivative functions
    """
    activation_name = activation_name.lower()
    
    if activation_name == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif activation_name == 'tanh':
        return tanh, tanh_derivative
    else:
        raise ValueError(f"Unknown activation: {activation_name}. Use 'sigmoid' or 'tanh'.")


if __name__ == "__main__":
    # Test activation functions
    Z = np.array([0, 1, 2, -1, -2])
    
    print("Activation Functions Test")
    print("=" * 60)
    
    print(f"\nInput Z: {Z}")
    
    # Sigmoid
    sig = sigmoid(Z)
    sig_der = sigmoid_derivative(Z)
    print(f"\nSigmoid(Z): {sig}")
    print(f"Sigmoid derivative: {sig_der}")
    print(f"Sigmoid properties: min={sig.min():.4f}, max={sig.max():.4f}")
    
    # Tanh
    th = tanh(Z)
    th_der = tanh_derivative(Z)
    print(f"\nTanh(Z): {th}")
    print(f"Tanh derivative: {th_der}")
    print(f"Tanh properties: min={th.min():.4f}, max={th.max():.4f}")
    
    # Softmax (multi-class output)
    Z_softmax = np.array([[1, 2, 3], [0, 0, 0], [-1, 0, 1]])
    soft = softmax(Z_softmax)
    print(f"\nSoftmax test (each row should sum to 1):")
    print(soft)
    print(f"Row sums: {soft.sum(axis=1)}")

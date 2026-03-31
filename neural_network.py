"""
Neural network class for 3-class classification.
Implements forward propagation, backward propagation, and training loop.
"""

import numpy as np
from activations import get_activation, softmax


class NeuralNetwork:
    """
    2-layer neural network for multi-class classification.
    Architecture: Input -> Hidden (sigmoid/tanh) -> Output (softmax)
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, activation='tanh'):
        """
        Initialize network with random weights and biases.
        
        Args:
            input_size: Number of input features (e.g., 3 for x, y, z)
            hidden_size: Number of hidden layer units
            output_size: Number of output classes (e.g., 3)
            learning_rate: Learning rate for gradient descent
            activation: 'sigmoid' or 'tanh'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # Get activation function and its derivative
        self.activation_func, self.activation_derivative = get_activation(activation)
        
        # Initialize weights with Xavier initialization
        # Xavier init: std = sqrt(1 / fan_in)
        std_input_to_hidden = np.sqrt(1.0 / input_size)
        std_hidden_to_output = np.sqrt(1.0 / hidden_size)
        
        self.W1 = np.random.randn(input_size, hidden_size) * std_input_to_hidden
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * std_hidden_to_output
        self.b2 = np.zeros((1, output_size))
        
        # Store parameters for debugging
        self.training_history = {
            'loss': [],
            'train_accuracy': [],
            'epoch': []
        }
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input features, shape (m, input_size)
        
        Returns:
            A2: Output probabilities, shape (m, output_size)
            cache: (Z1, A1, Z2) for use in backward pass
        """
        # Hidden layer
        Z1 = np.dot(X, self.W1) + self.b1  # (m, hidden_size)
        A1 = self.activation_func(Z1)       # Apply sigmoid or tanh
        
        # Output layer
        Z2 = np.dot(A1, self.W2) + self.b2  # (m, output_size)
        A2 = softmax(Z2)                    # Apply softmax
        
        cache = (Z1, A1, Z2)
        return A2, cache
    
    def backward(self, X, y_true, cache):
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input features, shape (m, input_size)
            y_true: One-hot encoded labels, shape (m, output_size)
            cache: (Z1, A1, Z2) from forward pass
        
        Returns:
            gradients: Dictionary with dW1, db1, dW2, db2
        """
        Z1, A1, Z2 = cache
        m = X.shape[0]
        
        # === Output layer gradient ===
        # For softmax + cross-entropy, dZ2 = A2 - y_true
        A2 = softmax(Z2)
        dZ2 = A2 - y_true  # (m, output_size)
        
        # === Backpropagate to hidden layer ===
        dA1 = np.dot(dZ2, self.W2.T)  # (m, hidden_size)
        
        # Apply activation derivative
        dA1_dZ1 = self.activation_derivative(Z1)  # (m, hidden_size)
        dZ1 = dA1 * dA1_dZ1  # Element-wise multiply (m, hidden_size)
        
        # === Compute weight and bias gradients ===
        dW1 = (1 / m) * np.dot(X.T, dZ1)      # (input_size, hidden_size)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)  # (1, hidden_size)
        
        dW2 = (1 / m) * np.dot(A1.T, dZ2)     # (hidden_size, output_size)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # (1, output_size)
        
        gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return gradients
    
    def update_weights(self, gradients):
        """
        Update weights and biases using gradient descent.
        
        Args:
            gradients: Dictionary with dW1, db1, dW2, db2
        """
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss.
        
        Args:
            y_true: One-hot encoded true labels, shape (m, output_size)
            y_pred: Predicted probabilities, shape (m, output_size)
        
        Returns:
            loss: Average cross-entropy loss over all samples
        """
        m = y_true.shape[0]
        
        # Avoid log(0) by clipping predictions
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Cross-entropy: -sum(y_true * log(y_pred))
        cross_entropy = -np.sum(y_true * np.log(y_pred_clipped)) / m
        
        return cross_entropy
    
    def train(self, X_train, y_train_onehot, epochs=100, batch_size=None, verbose=True):
        """
        Train the network on training data.
        
        Args:
            X_train: Training features, shape (m, input_size)
            y_train_onehot: One-hot encoded training labels, shape (m, output_size)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent. If None, use full batch.
            verbose: Print loss every epoch if True
        """
        m = X_train.shape[0]
        
        if batch_size is None:
            batch_size = m
        
        for epoch in range(epochs):
            # Shuffle data at start of epoch
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            num_batches = (m + batch_size - 1) // batch_size
            
            # Mini-batch gradient descent
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, m)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred, cache = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss
                
                # Backward pass
                gradients = self.backward(X_batch, y_batch, cache)
                
                # Update weights
                self.update_weights(gradients)
            
            # Average loss over batches
            epoch_loss /= num_batches
            
            # Compute training accuracy
            train_acc = self.accuracy(X_train, y_train_onehot)
            
            # Store history
            self.training_history['loss'].append(epoch_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['epoch'].append(epoch)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f} | Accuracy: {train_acc:.4f}")
    
    def predict(self, X):
        """
        Make predictions on data.
        
        Args:
            X: Input features, shape (m, input_size)
        
        Returns:
            predictions: Predicted class labels, shape (m,) with values 0 to output_size-1
        """
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X, y_true_onehot):
        """
        Compute classification accuracy.
        
        Args:
            X: Input features, shape (m, input_size)
            y_true_onehot: One-hot encoded true labels, shape (m, output_size)
        
        Returns:
            accuracy: Fraction of correct predictions
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y_true_onehot, axis=1)
        return np.mean(predictions == true_labels)


if __name__ == "__main__":
    # Quick test
    print("Testing NeuralNetwork class...")
    
    # Create a small test network
    net = NeuralNetwork(input_size=3, hidden_size=8, output_size=3, learning_rate=0.01, activation='tanh')
    print(f"Network created: {net.input_size} -> {net.hidden_size} -> {net.output_size}")
    
    # Test forward pass
    X_test = np.random.randn(5, 3)
    y_pred, cache = net.forward(X_test)
    print(f"Forward pass: X shape {X_test.shape} -> Output shape {y_pred.shape}")
    print(f"Output probabilities (first sample): {y_pred[0]}")
    print(f"Sum of probabilities (should be 1): {y_pred[0].sum():.6f}")

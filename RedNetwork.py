import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        # Initialization of weights and biases
        # The function is in format: np.random.randn(rows, columns) * standard_deviation
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def _relu(self, Z):
        return np.maximum(0, Z)
    
    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def _softmax(self, Z):
        # For numerical stability, subtract row-wise max
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        # Store intermediate values for backpropagation
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self._relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self._softmax(self.Z2)
        return self.A2
    
    def compute_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        # Cross-entropy loss, adding epsilon to avoid log(0)
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
        return loss
    
    def backward(self, X, Y_true):
        m = Y_true.shape[0]

        # ------------- Output Layer -------------
        dZ2 = self.A2 - Y_true                      # (m, output_size)
        dW2 = self.A1.T @ dZ2                       # (hidden_size, output_size)    
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m # (1, output_size)

        # ------------- Hidden Layer -------------
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._relu_derivative(self.Z1)   # (m, hidden_size)
        dW1 = X.T @ dZ1                             # (input_size, hidden_size)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m # (1, hidden_size)

        # ------------- Update Weights and Biases -------------

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, Y, epochs, verbose=True):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            Y_pred = self.forward(X)
            # Compute loss
            loss = self.compute_loss(Y, Y_pred)
            losses.append(loss)
            # Backward pass
            self.backward(X, Y)
            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
        return losses
    
    def predict(self, X):
        Y_pred = self.forward(X)
        return np.argmax(Y_pred, axis=1)

# Example usage
np.random.seed(42)
n_samples = 300


# Crear los centros en campos positivos
centers = [[1, 1, 1], 
           [-1, -1, 1], 
           [1, -1, -1]]


X_list = []
y_list = []

for i, center in enumerate(centers):
    X_cluster = np.random.randn(n_samples // 3, 3) + center
    X_list.append(X_cluster)
    y_list.append(np.full(n_samples // 3, i))

X = np.vstack(X_list)
y = np.hstack(y_list)

# Shuffle and split the dataset
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
Y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

# Create and train the neural network
nn = NeuralNetwork(input_size=3, hidden_size=5, output_size=3, learning_rate=0.01)
losses = nn.train(X_train, Y_train, epochs=500)

# Evaluate the model
y_pred = nn.predict(X_test)
y_true = np.argmax(Y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f'Accuracy: {accuracy:.4f}')

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Curve')
plt.show()
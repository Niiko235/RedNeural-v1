# Architecture & Module Documentation

Detailed breakdown of each module, what it does, and how to modify it.

---

## 📋 Quick Module Reference

| Module               | Purpose                 | Key Functions                                                | LOC  |
| -------------------- | ----------------------- | ------------------------------------------------------------ | ---- |
| `main.py`            | Orchestration           | `main()`                                                     | ~180 |
| `data_generation.py` | Synthetic data creation | `generate_3d_classification_data()`                          | ~50  |
| `utils.py`           | Data preprocessing      | `one_hot_encode()`, `normalize_data()`, `train_test_split()` | ~90  |
| `activations.py`     | Activation functions    | `sigmoid()`, `tanh()`, `softmax()`, `get_activation()`       | ~120 |
| `neural_network.py`  | Core network logic      | `NeuralNetwork` class with forward/backward/train            | ~250 |
| `visualization.py`   | Plotting results        | Multiple plot functions, `plot_all_results()`                | ~280 |

---

## 🔧 Module Deep Dives

### 1. `data_generation.py` — Creating Training Data

**Purpose**: Generate synthetic 3D data with 3 well-separated clusters

**Main Function**:

```python
generate_3d_classification_data(num_samples_per_class=100, random_state=42)
```

**How It Works**:

1. **Define 3 cluster centers** (class centroids):
   ```python
   center_class_0 = np.array([-2.0, -2.0, -2.0])  # Lower-left-front
   center_class_1 = np.array([2.0, 2.0, 2.0])     # Upper-right-back
   center_class_2 = np.array([2.0, -2.0, 0.0])    # Upper-left-middle
   ```

2. **Generate noise around each center** using Gaussian distribution:
   ```python
   class_0 = np.random.normal(center_class_0, std_dev=0.8, size=(100, 3))
   class_1 = np.random.normal(center_class_1, std_dev=0.8, size=(100, 3))
   class_2 = np.random.normal(center_class_2, std_dev=0.8, size=(100, 3))
   ```

3. **Combine and shuffle**:
   ```python
   X = vstack([class_0, class_1, class_2])  # Shape: (300, 3)
   y = [0,0,...,0, 1,1,...,1, 2,2,...,2]    # Shape: (300,)
   # Shuffle indices
   ```

**Returns**:

- `X`: Shape (300, 3) — features [x, y, z] for each sample
- `y`: Shape (300,) — class labels [0, 1, 2]

**Customization**:

To create **different data distribution**:

```python
# More/fewer samples per class
X, y = generate_3d_classification_data(num_samples_per_class=200)

# Different cluster centers
center_class_0 = np.array([0, 0, 0])       # Around origin
center_class_1 = np.array([5, 5, 5])       # Far away
center_class_2 = np.array([5, -5, 0])      # Different configuration

# More noise (harder classification task)
std_dev = 2.0  # Instead of 0.8

# Less noise (easier classification task)
std_dev = 0.3  # Very tight clusters
```

To **load real data** instead:

```python
# Replace generate_3d_classification_data() in main.py with:
import pandas as pd
data = pd.read_csv('your_data.csv')
X = data[['x', 'y', 'z']].values  # Shape: (m, 3)
y = data['class'].values           # Shape: (m,) with values 0,1,2
```

---

### 2. `utils.py` — Data Preprocessing

**Purpose**: Prepare raw data for training (encode, normalize, split)

#### Function: `one_hot_encode(y, num_classes)`

**What it does**: Converts class labels to one-hot vectors

**Example**:

```python
y = np.array([0, 1, 2, 0, 1])
one_hot = one_hot_encode(y, num_classes=3)
# Result:
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1],
#  [1, 0, 0],
#  [0, 1, 0]]
```

**Why necessary**:

- Cross-entropy loss formula requires this format
- Softmax output is also multi-hot

**Implementation**:

```python
def one_hot_encode(y, num_classes):
    m = len(y)
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1  # Set 1 at correct position
    return one_hot
```

**Advanced**: Create from pandas:

```python
import pandas as pd
one_hot = pd.get_dummies(y, drop_first=False).values
```

---

#### Function: `normalize_data(X)`

**What it does**: Standardize features to zero mean and unit variance

**Example**:

```python
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
X_norm, mean, std = normalize_data(X)
# mean: [4.0, 5.0, 6.0]
# std:  [2.45, 2.45, 2.45]
# X_norm: [[-1.22, -1.22, -1.22],
#          [ 0.0,   0.0,   0.0],
#          [ 1.22,  1.22,  1.22]]
```

**Why necessary**:

- Features on different scales influence learning differently
- Gradient descent converges faster with normalized data
- Weight initialization assumes normalized input

**Formula**:

```
X_normalized = (X - mean) / std
```

**Implementation**:

```python
def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    X_normalized = (X - mean) / std
    return X_normalized, mean, std
```

**Denormalize (convert predictions back to original scale)**:

```python
X_original = X_normalized * std + mean
```

---

#### Function: `train_test_split(X, y, test_ratio, random_state)`

**What it does**: Split data into training and testing sets

**Example**:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, random_state=42)
# 80% train (240 samples), 20% test (60 samples)
```

**Why necessary**:

- **Training set**: Learn patterns (optimize weights)
- **Test set**: Evaluate generalization (unseen data)
- Prevents overfitting assessment

**Implementation**:

```python
def train_test_split(X, y, test_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    m = len(X)
    test_size = int(m * test_ratio)
    indices = np.random.permutation(m)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```

**Advanced stratified split** (ensures class distribution):

```python
from sklearn.model_selection import train_test_split as sk_split
X_train, X_test, y_train, y_test = sk_split(X, y, test_size=0.2, stratify=y)
```

---

### 3. `activations.py` — Activation Functions

**Purpose**: Implement sigmoid, tanh, and softmax with derivatives

#### Sigmoid

```python
def sigmoid(Z):
    Z_clipped = np.clip(Z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-Z_clipped))

def sigmoid_derivative(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)
```

**Usage in forward pass**:

```python
Z1 = X @ W1 + b1
A1 = sigmoid(Z1)  # Hidden layer output
```

**Usage in backward pass**:

```python
dA1_dZ1 = sigmoid_derivative(Z1)  # Activation derivative
dZ1 = dA1 * dA1_dZ1               # Apply to incoming gradient
```

**Why clip to [-500, 500]?**

- `exp(x)` overflows for large x (max float ~308)
- Sigmoid values are already 0 or 1 outside this range
- No numerical loss clipping this way

---

#### Tanh

```python
def tanh(Z):
    return np.tanh(Z)  # Built-in NumPy function

def tanh_derivative(Z):
    tanh_z = np.tanh(Z)
    return 1 - tanh_z ** 2
```

**Alternative using cached value** (more efficient):

```python
# During forward pass: cache A1 = tanh(Z1)
# During backward pass:
dA1_dZ1 = 1 - A1 ** 2  # Use cached value instead of recomputing
```

---

#### Softmax

```python
def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)  # Numerical stability
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
```

**Why subtract max?**

- Prevents overflow when computing exp
- Doesn't change result (exp(x-c) / sum(exp(x-c)) = exp(x) / sum(exp(x)))
- Improves numerical stability

**Verification** (each row sums to 1):

```python
Z = np.array([[1,2,3], [0,0,0]])
output = softmax(Z)
print(output.sum(axis=1))  # Should be [1.0, 1.0]
```

---

#### Helper Function: `get_activation(activation_name)`

**Purpose**: Return activation function and derivative by name

```python
def get_activation(activation_name):
    if activation_name == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif activation_name == 'tanh':
        return tanh, tanh_derivative
    else:
        raise ValueError(f"Unknown activation: {activation_name}")
```

**Usage**:

```python
activation_func, activation_derivative = get_activation('tanh')
A1 = activation_func(Z1)           # Forward
dZ1 = dA1 * activation_derivative(Z1)  # Backward
```

**To add a new activation** (e.g., ReLU):

```python
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

# Add to get_activation:
elif activation_name == 'relu':
    return relu, relu_derivative
```

---

### 4. `neural_network.py` — Core Network Class

**Purpose**: Implement forward pass, backward pass, and training loop

#### Class Initialization

```python
def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, activation='tanh'):
    self.input_size = 3
    self.hidden_size = 16
    self.output_size = 3
    self.learning_rate = 0.01
    self.activation_name = 'tanh'
    
    # Get activation functions
    self.activation_func, self.activation_derivative = get_activation('tanh')
    
    # Initialize weights with Xavier initialization
    std_in_to_hidden = np.sqrt(1.0 / input_size)      # sqrt(1/3)
    std_hidden_to_out = np.sqrt(1.0 / hidden_size)    # sqrt(1/16)
    
    self.W1 = np.random.randn(3, 16) * std_in_to_hidden
    self.b1 = np.zeros((1, 16))
    self.W2 = np.random.randn(16, 3) * std_hidden_to_out
    self.b2 = np.zeros((1, 3))
```

**Why these dimensions?**

- `W1`: (input_size, hidden_size) = (3, 16)
  - Each input feature connects to all 16 hidden units
- `b1`: (1, hidden_size) = (1, 16)
  - One bias per hidden unit, broadcast over samples
- `W2`: (hidden_size, output_size) = (16, 3)
  - Each hidden unit connects to 3 output classes
- `b2`: (1, output_size) = (1, 3)
  - One bias per output class

---

#### Forward Pass

```python
def forward(self, X):  # X shape: (m, 3)
    # Hidden layer
    Z1 = np.dot(X, self.W1) + self.b1           # (m, 3) @ (3, 16) = (m, 16)
    A1 = self.activation_func(Z1)                # (m, 16)
    
    # Output layer
    Z2 = np.dot(A1, self.W2) + self.b2          # (m, 16) @ (16, 3) = (m, 3)
    A2 = softmax(Z2)                             # (m, 3)
    
    cache = (Z1, A1, Z2)
    return A2, cache
```

**Data flow**:

```
Input (m, 3)
    ↓
Multiply by W1 (3, 16)  →  Z1 (m, 16)
    ↓
Add b1 (1, 16)  →  Z1 + b1 (m, 16)
    ↓
Apply sigmoid/tanh  →  A1 (m, 16)
    ↓
Multiply by W2 (16, 3)  →  Z2 (m, 3)
    ↓
Add b2 (1, 3)  →  Z2 + b2 (m, 3)
    ↓
Apply softmax  →  A2 (m, 3) [probabilities]
```

**Cache for backward pass**:

```python
cache = (Z1, A1, Z2)  # Store intermediate values
```

Why cache? Reusing avoids recomputation in backward pass.

---

#### Backward Pass

```python
def backward(self, X, y_true, cache):  # y_true shape: (m, 3) one-hot
    Z1, A1, Z2 = cache
    m = X.shape[0]
    
    # === Output layer ===
    A2 = softmax(Z2)
    dZ2 = A2 - y_true              # (m, 3) - (m, 3) = (m, 3)
    
    # === Backpropagate to hidden ===
    dA1 = np.dot(dZ2, self.W2.T)   # (m, 3) @ (3, 16) = (m, 16)
    dA1_dZ1 = self.activation_derivative(Z1)  # (m, 16)
    dZ1 = dA1 * dA1_dZ1            # Element-wise (m, 16)
    
    # === Compute weight gradients ===
    dW1 = (1/m) * np.dot(X.T, dZ1)     # (3, m) @ (m, 16) = (3, 16)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)  # (1, 16)
    
    dW2 = (1/m) * np.dot(A1.T, dZ2)    # (16, m) @ (m, 3) = (16, 3)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)  # (1, 3)
    
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
```

**Gradient flow**:

```
Loss
  ↓
dZ2 = ∂Loss/∂Z2 = A2 - y_true
  ↓
dA1 = dZ2 @ W2ᵀ  (backprop to hidden)
  ↓
dZ1 = dA1 * σ'(Z1)  (apply activation derivative)
  ↓
dW1 = Xᵀ @ dZ1 / m
db1 = sum(dZ1) / m
dW2 = A1ᵀ @ dZ2 / m
db2 = sum(dZ2) / m
```

**Why element-wise multiplication?**

```python
dZ1 = dA1 * dA1_dZ1
```

Chain rule: `∂Loss/∂Z1 = (∂Loss/∂A1) · (∂A1/∂Z1)`

---

#### Weight Update

```python
def update_weights(self, gradients):
    learning_rate = self.learning_rate
    self.W1 -= learning_rate * gradients['dW1']
    self.b1 -= learning_rate * gradients['db1']
    self.W2 -= learning_rate * gradients['dW2']
    self.b2 -= learning_rate * gradients['db2']
```

**Intuition**:

- Subtract gradient (direction of steepest descent)
- Multiply by learning_rate to control step size
- Larger LR = bigger jumps (risk overshooting)
- Smaller LR = smaller jumps (more epochs to converge)

**Typical values**:

- 0.1: Too large, oscillates
- 0.01: Good for this problem ✓
- 0.001: Conservative, slower convergence

---

#### Loss Computation

```python
def compute_loss(self, y_true, y_pred):
    m = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)  # Avoid log(0)
    cross_entropy = -np.sum(y_true * np.log(y_pred_clipped)) / m
    return cross_entropy
```

**Formula breakdown**:

```
For each sample i:
    Loss_i = -Σⱼ(y_true[i,j] * log(y_pred[i,j]))
    
    Since y_true is one-hot, only one j has 1, rest are 0:
    Loss_i = -log(y_pred[i, true_class])

Final loss = average over all samples
```

**Why clip predictions?**

- `log(0) = -∞` (undefined)
- `log(1e-7) ≈ -16.1` (manageable)
- Very small change to result but prevents NaN

---

#### Training Loop

```python
def train(self, X_train, y_train_onehot, epochs=100, batch_size=None, verbose=True):
    m = X_train.shape[0]
    if batch_size is None:
        batch_size = m  # Full batch
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]
        
        epoch_loss = 0
        num_batches = (m + batch_size - 1) // batch_size
        
        # Mini-batches
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, m)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward → Backward → Update
            y_pred, cache = self.forward(X_batch)
            loss = self.compute_loss(y_batch, y_pred)
            gradients = self.backward(X_batch, y_batch, cache)
            self.update_weights(gradients)
            
            epoch_loss += loss
        
        # Average loss over batches
        epoch_loss /= num_batches
        
        # Compute accuracy
        train_acc = self.accuracy(X_train, y_train_onehot)
        
        # Store history
        self.training_history['loss'].append(epoch_loss)
        self.training_history['train_accuracy'].append(train_acc)
        
        if verbose and (epoch+1) % (epochs//10) == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f} | Acc: {train_acc:.4f}")
```

**Key loop components**:

1. **Shuffle**: Randomize order each epoch (helps generalization)
2. **Mini-batches**: Process batch_size samples at a time
3. **Forward/Backward/Update**: Gradient descent step
4. **History**: Track loss and accuracy for visualization

---

#### Prediction

```python
def predict(self, X):  # X shape: (m, 3)
    y_pred, _ = self.forward(X)  # Get softmax outputs
    return np.argmax(y_pred, axis=1)  # Return class with highest probability
```

**Example**:

```python
y_pred_probs = [0.1, 0.7, 0.2]  # Probabilities for 3 classes
np.argmax(y_pred_probs)  # Returns 1 (class with 0.7)
```

---

#### Accuracy

```python
def accuracy(self, X, y_true_onehot):
    predictions = self.predict(X)
    true_labels = np.argmax(y_true_onehot, axis=1)
    return np.mean(predictions == true_labels)
```

**Example**:

```python
predictions = [0, 1, 2, 0, 0]
true_labels = [0, 1, 1, 0, 2]
matches = [True, True, False, True, False]
accuracy = 3/5 = 0.6 (60%)
```

---

### 5. `visualization.py` — Plotting Results

**Purpose**: Create informative plots for analysis

#### Key Functions

| Function                      | Plots                            |
| ----------------------------- | -------------------------------- |
| `plot_training_curves()`      | Loss vs epochs (multiple models) |
| `plot_accuracy_comparison()`  | Train/test accuracy bars         |
| `plot_decision_boundary_2d()` | 2D decision regions              |
| `plot_confusion_matrix()`     | Classification matrix            |
| `plot_all_results()`          | 4-panel comprehensive view       |

**Example: Custom visualization**

```python
from visualization import plot_training_curves
import matplotlib.pyplot as plt

# After training networks
networks = {'sigmoid': net_sigmoid, 'tanh': net_tanh}
plot_training_curves(networks)
plt.show()
```

---

### 6. `main.py` — Orchestration

**Purpose**: Tie everything together (data → train → evaluate → visualize)

**Flow**:

```
1. Generate data (300 samples, 3 classes)
2. Normalize features
3. Split 80/20 train/test
4. Create & train sigmoid network (300 epochs)
5. Create & train tanh network (300 epochs)
6. Compare results (accuracy, loss)
7. Generate 4 comparison plots
8. Print summary with insights
```

**Key section** (training loop):

```python
for activation in ['sigmoid', 'tanh']:
    network = NeuralNetwork(3, 16, 3, 0.01, activation=activation)
    network.train(X_train, y_train_onehot, epochs=300, verbose=True)
    networks[activation] = network
```

---

## 🔧 Common Customizations

### Change Hidden Layer Size

**File**: `main.py`, line ~120

```python
# Before
network = NeuralNetwork(input_size=3, hidden_size=16, ...)

# After (try 8, 32, 64)
network = NeuralNetwork(input_size=3, hidden_size=32, ...)
```

**Effect**:

- Smaller (8): Faster training, less capacity
- Larger (64): Slower training, more capacity (risk overfitting on 300 samples)

---

### Change Number of Epochs

**File**: `main.py`, line ~130

```python
# Before
network.train(..., epochs=300, ...)

# After
network.train(..., epochs=500, ...)
```

**Effect**:

- More epochs = more training time but better convergence (usually)
- Less epochs = faster but may not fully learn

---

### Change Learning Rate

**File**: `main.py`, line ~124

```python
# Before
NeuralNetwork(..., learning_rate=0.01, ...)

# After
NeuralNetwork(..., learning_rate=0.001, ...)  # Slower
NeuralNetwork(..., learning_rate=0.1, ...)    # Faster (risk instability)
```

---

### Train Only Tanh (Faster Testing)

**File**: `main.py`, line ~108

```python
# Before
activations = ['sigmoid', 'tanh']

# After
activations = ['tanh']
```

---

### Use Different Data

**File**: `main.py`, line ~77

```python
# Before
X, y = generate_3d_classification_data(num_samples_per_class=100)

# After (more samples, fewer samples, custom data, etc.)
X, y = generate_3d_classification_data(num_samples_per_class=500)

# Or load from file
import pandas as pd
df = pd.read_csv('your_data.csv')
X = df[['x', 'y', 'z']].values
y = df['class'].values
```

---

## 📝 Code Quality Tips

### Add Comments for Clarity

```python
# Before
A1 = sigmoid(Z1)
dZ1 = dA1 * sigmoid_derivative(Z1)

# After
A1 = sigmoid(Z1)  # Apply sigmoid activation to hidden layer
dZ1_contribution = sigmoid_derivative(Z1)  # Gradient of sigmoid
dZ1 = dA1 * dZ1_contribution  # Chain rule: ∂L/∂Z1 = ∂L/∂A1 * ∂A1/∂Z1
```

### Use Type Hints

```python
# Before
def forward(self, X):
    return A2, cache

# After
def forward(self, X: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """Forward propagation through network."""
    return A2, cache
```

### Vectorize Operations

```python
# Slow (loop version)
for i in range(m):
    Z1[i] = np.dot(W1, X[i]) + b1

# Fast (vectorized)
Z1 = np.dot(X, W1) + b1  # m samples at once
```

---

## 🧪 Testing Individual Modules

### Test data_generation.py

```python
from data_generation import generate_3d_classification_data
X, y = generate_3d_classification_data(10)
print(f"X shape: {X.shape}")  # Expected: (30, 3)
print(f"y unique: {np.unique(y)}")  # Expected: [0, 1, 2]
```

### Test utils.py

```python
from utils import one_hot_encode, normalize_data
y = np.array([0, 1, 2])
one_hot = one_hot_encode(y, 3)
print(one_hot.shape)  # Expected: (3, 3)
print(one_hot[0])    # Expected: [1, 0, 0]
```

### Test activations.py

```python
from activations import sigmoid, tanh, softmax
z = np.array([0, 1, -1])
print(sigmoid(z))      # Should be [0.5, 0.73, 0.27]
print(tanh(z))         # Should be [0, 0.76, -0.76]

z2d = np.array([[1,2,3]])
print(softmax(z2d).sum())  # Should be 1.0
```

---

**End of Architecture Guide**

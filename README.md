# Neural Network from Scratch - 3-Class Classification

A complete implementation of a 2-layer neural network built entirely with
**NumPy** and basic matrix operations. This project demonstrates how neural
networks work at the mathematical level, comparing **Sigmoid** and **Tanh**
activation functions on a 3D classification task.

**Perfect for learning:** No black-box libraries (TensorFlow, PyTorch) — just
pure NumPy to understand the fundamentals!

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Results & Interpretation](#results--interpretation)
- [Key Concepts](#key-concepts)
- [Extension Ideas](#extension-ideas)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

### What Does It Do?

This project trains a neural network to classify 3D data points into **3
distinct classes**. Each data point has three features: `x`, `y`, and `z`
coordinates. The network learns to draw decision boundaries that separate the
three classes.

### Why This Project?

- **Learning**: Understand neural networks from first principles
- **No Magic**: Every calculation is visible and understandable
- **Comparison**: See how different activation functions (sigmoid vs tanh)
  affect learning
- **Practical**: Real forward/backward propagation, not toy examples

### Key Results

| Activation | Train Acc  | Test Acc   | Final Loss |
| ---------- | ---------- | ---------- | ---------- |
| Sigmoid    | 96.25%     | 96.67%     | 0.777      |
| **Tanh**   | **99.17%** | **98.33%** | **0.170**  |

**Insight**: Tanh converges **4.6× faster** than sigmoid due to its symmetric
output range [-1, 1].

---

## 🧮 Mathematical Foundation

### 1. Neural Network Architecture

```
Input Layer (3 features)
    ↓
Hidden Layer (16 units, sigmoid or tanh activation)
    ↓
Output Layer (3 classes, softmax activation)
```

### 2. Forward Propagation

Given input `X` (shape: m × 3), the network computes:

```
Z₁ = X @ W₁ + b₁                    (pre-activation in hidden layer)
A₁ = activation(Z₁)                 (hidden layer output, m × 16)
Z₂ = A₁ @ W₂ + b₂                   (pre-activation in output layer)
A₂ = softmax(Z₂)                    (output probabilities, m × 3)
```

**Matrix Dimensions:**

- `W₁`: 3 × 16 (input → hidden)
- `b₁`: 1 × 16 (bias for hidden layer)
- `W₂`: 16 × 3 (hidden → output)
- `b₂`: 1 × 3 (bias for output layer)

### 3. Activation Functions

#### Sigmoid

```
σ(z) = 1 / (1 + e^(-z))
Output range: (0, 1)
Derivative: σ(z) · (1 - σ(z))
```

**Characteristics**:

- Squashes values to (0, 1)
- Outputs NOT centered (mean ≈ 0.5)
- Slower convergence in deep networks
- Historical importance but less preferred now

#### Tanh (Hyperbolic Tangent)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
Output range: (-1, 1)
Derivative: 1 - tanh²(z)
```

**Characteristics**:

- Squashes values to (-1, 1) — **symmetric around 0**
- Better gradient flow during backpropagation
- Faster convergence in practice
- **Recommended** for most hidden layers

### 4. Softmax (Output Layer)

For multi-class classification, the output layer uses softmax:

```
softmax(z_i) = e^(z_i) / Σⱼ e^(z_j)
```

Converts logits to probability distribution (sums to 1).

### 5. Cross-Entropy Loss

Measures how well predicted probabilities match true labels:

```
Loss = -Σ(y_true · log(y_pred)) / m
```

Where:

- `y_true`: One-hot encoded labels [1, 0, 0], [0, 1, 0], or [0, 0, 1]
- `y_pred`: Predicted probabilities from softmax
- `m`: Number of samples

### 6. Backpropagation (Gradient Computation)

Given cached values from forward pass, compute gradients:

**Output layer gradient:**

```
dZ₂ = A₂ - y_true    (derivative of softmax + cross-entropy)
```

**Backpropagation to hidden layer:**

```
dA₁ = dZ₂ @ W₂ᵀ
dZ₁ = dA₁ ⊙ activation'(Z₁)    (⊙ = element-wise multiply)
```

**Weight and bias gradients:**

```
dW₁ = (1/m) · Xᵀ @ dZ₁
db₁ = (1/m) · sum(dZ₁, axis=0)
dW₂ = (1/m) · A₁ᵀ @ dZ₂
db₂ = (1/m) · sum(dZ₂, axis=0)
```

### 7. Gradient Descent (Parameter Update)

```
W₁ := W₁ - learning_rate · dW₁
b₁ := b₁ - learning_rate · db₁
W₂ := W₂ - learning_rate · dW₂
b₂ := b₂ - learning_rate · db₂
```

---

## 🏗️ Architecture

### 2-Layer Network Design

```
┌─────────────────────────────────────────────────────────┐
│                   NEURAL NETWORK                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input (3D features)                                    │
│      ↓                                                   │
│  [X @ W₁ + b₁] → [Sigmoid/Tanh]  ← Hidden Layer        │
│      ↓                                                   │
│  [A₁ @ W₂ + b₂] → [Softmax]  ← Output (3 classes)      │
│      ↓                                                   │
│  Class Probabilities                                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture?

1. **Input Layer (3 units)**: One per feature (x, y, z)
2. **Hidden Layer (16 units)**:
   - Enough capacity to learn non-linear boundaries
   - Xavier initialization: `std = sqrt(1/3) ≈ 0.577`
3. **Output Layer (3 units)**:
   - One per class (softmax gives probabilities)
   - Xavier initialization: `std = sqrt(1/16) ≈ 0.25`

---

## 📁 Project Structure

```
RedNeural-v1/
├── README.md                          ← This file
├── ARCHITECTURE.md                    ← Detailed architecture explanation
├── MATH_REFERENCE.md                  ← Math formulas and derivations
├── USAGE_GUIDE.md                     ← How to use and modify code
│
├── main.py                            ← Entry point: orchestrates training and evaluation
├── data_generation.py                 ← Creates synthetic 3D classification data
├── utils.py                           ← Data preprocessing (normalize, one-hot, split)
├── activations.py                     ← Sigmoid, tanh, softmax implementations
├── neural_network.py                  ← Core NeuralNetwork class with forward/backward
├── visualization.py                   ← Plotting functions for results
│
├── results_loss_comparison.png         ← Sigmoid vs tanh loss curves
├── results_accuracy_comparison.png    ← Sigmoid vs tanh accuracy bars
├── results_detailed_sigmoid.png       ← 4-panel sigmoid analysis
├── results_detailed_tanh.png          ← 4-panel tanh analysis
│
└── .venv/                             ← Python virtual environment
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Step 1: Clone or Navigate to Project

```bash
cd c:\Programming\RedNeural-v1
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell/CMD)**: `.\.venv\Scripts\activate`
- **Linux/Mac**: `source .venv/bin/activate`

### Step 3: Install Dependencies

```bash
pip install numpy matplotlib scikit-learn
```

**What each package does:**

- **NumPy**: Matrix operations and numerical computing
- **Matplotlib**: Visualization and plotting
- **Scikit-learn**: Used only for PCA in visualization (optional, can be
  removed)

### Verify Installation

```bash
python -c "import numpy; print(f'NumPy {numpy.__version__} ✓')"
python -c "import matplotlib; print(f'Matplotlib ✓')"
```

---

## ⚡ Quick Start

### Run Everything (Data → Train → Evaluate → Visualize)

```bash
python main.py
```

**What happens:**

1. Generates 300 synthetic 3D samples (100 per class)
2. Normalizes features to zero mean, unit variance
3. Splits into 80% train (240 samples), 20% test (60 samples)
4. Trains two networks: one with sigmoid, one with tanh activation
5. Each trains for 300 epochs with full-batch gradient descent
6. Prints loss and accuracy every 30 epochs
7. Generates 4 comparison visualizations
8. Displays results summary

**Expected Output:**

```
======================================================================
Neural Network from Scratch - Sigmoid vs Tanh Comparison
======================================================================

[1/6] Generating synthetic 3D classification data...
  Generated: 300 samples with 3 features, 3 classes

[2/6] Normalizing features...
  Completed

[3/6] Splitting data (80% train, 20% test)...
  Training set: 240 samples
  Test set:     60 samples

[4/6] Training network with SIGMOID activation...
Epoch 30/300 | Loss: 1.131243 | Accuracy: 0.3833
...
Epoch 300/300 | Loss: 0.777054 | Accuracy: 0.9625

Final Results (sigmoid):
  Final Loss:       0.777054
  Train Accuracy:   0.9625
  Test Accuracy:    0.9667

[4/6] Training network with TANH activation...
Epoch 30/300 | Loss: 0.805830 | Accuracy: 0.7375
...
Epoch 300/300 | Loss: 0.170038 | Accuracy: 0.9917

Final Results (tanh):
  Final Loss:       0.170038
  Train Accuracy:   0.9917
  Test Accuracy:    0.9833
```

---

## 📖 Detailed Usage

### Option 1: Use Only Tanh (Recommended)

Edit `main.py`, line with `activations = ['sigmoid', 'tanh']`:

```python
activations = ['tanh']  # Train only tanh
```

Shorter runtime, less output.

### Option 2: Train Custom Network Directly

```python
import numpy as np
from data_generation import generate_3d_classification_data
from utils import normalize_data, train_test_split, one_hot_encode
from neural_network import NeuralNetwork

# Generate data
X, y = generate_3d_classification_data(num_samples_per_class=150)

# Normalize
X_norm, _, _ = normalize_data(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_ratio=0.2)
y_train_onehot = one_hot_encode(y_train, num_classes=3)
y_test_onehot = one_hot_encode(y_test, num_classes=3)

# Create network
network = NeuralNetwork(
    input_size=3,
    hidden_size=32,  # Try larger hidden layer
    output_size=3,
    learning_rate=0.01,
    activation='tanh'
)

# Train
network.train(X_train, y_train_onehot, epochs=500, verbose=True)

# Evaluate
train_acc = network.accuracy(X_train, y_train_onehot)
test_acc = network.accuracy(X_test, y_test_onehot)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Predict on new data
predictions = network.predict(X_test)
```

### Option 3: Load Your Own Data

```python
import numpy as np
from utils import normalize_data, one_hot_encode

# Load your CSV or data structure
# X should be shape (num_samples, 3) with x, y, z values
# y should be shape (num_samples,) with class labels 0, 1, 2

X = np.loadtxt('your_data.csv', delimiter=',')[:, :3]
y = np.loadtxt('your_labels.csv', delimiter=',', dtype=int)

# Normalize
X_norm, _, _ = normalize_data(X)
y_onehot = one_hot_encode(y, num_classes=3)

# Train network
# ... (see Option 2 above)
```

---

## 📊 Results & Interpretation

### What the Visualizations Show

#### 1. **results_loss_comparison.png** — Training Loss Curves

```
Loss (lower is better)
    ▲
    │     ╱╲ Sigmoid (slower convergence)
    │    ╱  
    │   ╱
    │  ╱     ╱ Tanh (faster convergence)
    │ ╱    ╱
    │╱__╱______________________
              Epochs →
```

**What to look for:**

- Both curves should decrease monotonically (or mostly)
- Tanh reaches lower final loss faster
- If curve is flat or increasing, learning rate may need adjustment

#### 2. **results_accuracy_comparison.png** — Train vs Test Accuracy

**Bar chart showing:**

- Train accuracy (usually higher)
- Test accuracy (more realistic, generalization)
- Ideally, both should be >95% on this task

**Interpretation:**

- If train >> test: **overfitting** → reduce hidden layer size or add
  regularization
- If both low: **underfitting** → increase hidden layer size or epochs
- If similar and high: **good generalization** ✓

#### 3. **results_detailed_sigmoid.png** & **results_detailed_tanh.png**

Four panels per activation:

| Panel        | Shows             | What to Look For                         |
| ------------ | ----------------- | ---------------------------------------- |
| Top-Left     | Training Loss     | Smooth decrease, no spikes               |
| Top-Right    | Training Accuracy | Increasing trend                         |
| Bottom-Left  | Decision Boundary | Colored regions match data clusters      |
| Bottom-Right | Confusion Matrix  | Mostly on diagonal (correct predictions) |

---

## 🧠 Key Concepts

### 1. Forward Pass vs Backward Pass

| Phase        | Direction       | Purpose             | Formula                                |
| ------------ | --------------- | ------------------- | -------------------------------------- |
| **Forward**  | Inputs → Output | Compute predictions | `A2 = softmax(ReLU(X @ W1 @ W2 + b2))` |
| **Backward** | Output → Inputs | Compute gradients   | `dW = X^T @ dZ / m`                    |

### 2. Activation Functions Compared

| Property           | Sigmoid       | Tanh            |
| ------------------ | ------------- | --------------- |
| Output Range       | (0, 1)        | (-1, 1)         |
| Centered at 0?     | No (≈0.5)     | Yes             |
| Derivative Range   | (0, 0.25)     | (0, 1)          |
| Vanishing Gradient | More severe   | Less severe     |
| Convergence        | Slower        | Faster          |
| Use Case           | Output layer* | Hidden layers** |

*Sigmoid still good for binary classification output **Tanh generally better for
hidden layers

### 3. Weight Initialization (Xavier)

Why random initialization?
- Symmetric weights → neurons compute same thing (no learning)
- Too large → activation functions saturate
- Too small → gradients vanish

**Xavier Initialization:**
```
std_dev = sqrt(1 / fan_in)
W = randn(shape) * std_dev
```

For input→hidden: `std = sqrt(1/3) ≈ 0.577`
For hidden→output: `std = sqrt(1/16) ≈ 0.25`

### 4. One-Hot Encoding

Converts class labels to vectors:
```
Class 0 → [1, 0, 0]
Class 1 → [0, 1, 0]
Class 2 → [0, 0, 1]
```

Why? Cross-entropy loss requires this format.

### 5. Data Normalization

Standardizes features to same scale:
```
X_normalized = (X - mean) / std
```

Why? 
- Prevents one feature from dominating learning
- Weights initialize better
- Gradient descent converges faster

---

## 🔧 Extension Ideas

### Easy Extensions

1. **Try Different Hyperparameters**
   ```python
   # Modify in main.py or create new script
   hidden_size = 32  # Instead of 16
   learning_rate = 0.001  # Instead of 0.01
   epochs = 1000  # Instead of 300
   ```

2. **Add More Data**
   ```python
   X, y = generate_3d_classification_data(num_samples_per_class=500)  # More samples
   ```

3. **Create Your Own Data**
   - Generate 3 custom clusters
   - Load from CSV
   - Add noise/outliers

### Moderate Extensions

4. **Add Regularization (L2 Penalty)**
   ```python
   # Add to update_weights:
   regularization_strength = 0.01
   self.W1 -= learning_rate * (gradients['dW1'] + regularization_strength * self.W1 / m)
   self.W2 -= learning_rate * (gradients['dW2'] + regularization_strength * self.W2 / m)
   ```

5. **Implement Momentum**
   ```python
   # Store velocity for each parameter
   self.v_W1 = np.zeros_like(self.W1)
   # In update:
   momentum = 0.9
   self.v_W1 = momentum * self.v_W1 - learning_rate * gradients['dW1']
   self.W1 += self.v_W1
   ```

6. **Add Batch Normalization**
   ```python
   # Normalize hidden layer outputs
   A1_mean = np.mean(A1, axis=0)
   A1_std = np.std(A1, axis=0)
   A1_normalized = (A1 - A1_mean) / (A1_std + 1e-8)
   ```

### Advanced Extensions

7. **3+ Hidden Layers**
   ```python
   # Extend forward/backward for arbitrary depth
   Z1 = X @ W1 + b1
   A1 = tanh(Z1)
   Z2 = A1 @ W2 + b2
   A2 = tanh(Z2)  # Additional hidden layer
   Z3 = A2 @ W3 + b3
   A3 = softmax(Z3)
   ```

8. **Adaptive Learning Rate (Adam Optimizer)**
   ```python
   # Track per-parameter gradient history
   m_W1 += beta1 * (dW1 - m_W1)  # First moment
   v_W1 += beta2 * (dW1^2 - v_W1)  # Second moment
   W1 -= lr * m_W1 / (sqrt(v_W1) + eps)
   ```

9. **Early Stopping**
   ```python
   # Stop if validation loss increases
   if val_loss > best_val_loss:
       patience_counter += 1
       if patience_counter > 20:
           break
   ```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'numpy'"

**Solution:**
```bash
pip install numpy matplotlib scikit-learn
```

Or with virtual environment:
```bash
.venv\Scripts\pip install numpy matplotlib scikit-learn
```

### Problem: Loss is NaN or Infinity

**Causes & Solutions:**
1. **Learning rate too high** → Reduce from 0.01 to 0.001
2. **Numerical instability in softmax** → Already handled (subtract max before exp)
3. **Data not normalized** → Check if normalize_data() was called

### Problem: Accuracy Not Improving

**Checklist:**
- [ ] Data normalized? (mean≈0, std≈1)
- [ ] Weights initialized with Xavier? (check activations.py)
- [ ] Learning rate reasonable? (try 0.001 to 0.1)
- [ ] Enough epochs? (try 500 instead of 300)
- [ ] Hidden layer large enough? (try 32 instead of 16)

### Problem: Takes Too Long to Train

**Optimizations:**
1. Reduce hidden layer size: `hidden_size = 8` (instead of 16)
2. Use smaller dataset: `num_samples_per_class=50` (instead of 100)
3. Fewer epochs: `epochs=100` (instead of 300)
4. Larger learning rate: `learning_rate=0.05` (might need adjustment)

### Problem: Test Accuracy Much Lower Than Train Accuracy

**Cause:** Overfitting (model memorized training data)

**Solutions:**
1. Add L2 regularization (see Extension #4)
2. Reduce hidden layer: `hidden_size = 8`
3. Increase dropout (probabilistically "turn off" neurons)

---

## 📚 References & Further Reading

### Concepts
- **Backpropagation**: Rumelhart et al., 1986 ("Learning representations by back-propagating errors")
- **Xavier Initialization**: Glorot & Bengio, 2010
- **Softmax & Cross-Entropy**: cs231n.stanford.edu

### Related Frameworks (for comparison after learning)
- **TensorFlow/Keras**: High-level API built on low-level operations like ours
- **PyTorch**: More dynamic, research-friendly version of TensorFlow
- **JAX**: NumPy-like but with autodiff (automatic differentiation)

---

## 👥 Contributing & Questions

- Found a bug? Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component info
- Want to extend? See Extension Ideas above
- Questions on math? Check [MATH_REFERENCE.md](MATH_REFERENCE.md)
- Need usage help? See [USAGE_GUIDE.md](USAGE_GUIDE.md)

---

## 📄 License

This project is for educational purposes. Feel free to use, modify, and share.

---

**Created**: 2026
**Purpose**: Learning neural networks from scratch with pure NumPy
**Status**: ✅ Working, ready to extend!

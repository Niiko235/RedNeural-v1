# Usage Guide & How-To

Practical guide for team members on how to use, modify, and extend the neural
network project.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [Running the Default Pipeline](#running-the-default-pipeline)
- [Modifying Hyperparameters](#modifying-hyperparameters)
- [Using Custom Data](#using-custom-data)
- [Troubleshooting](#troubleshooting)
- [Common Tasks](#common-tasks)
- [Code Snippets](#code-snippets)

---

## ⚡ Quick Start

### 1 Minute Installation

```bash
cd c:\Programming\RedNeural-v1
pip install numpy matplotlib scikit-learn
python main.py
```

**What happens**: Trains both sigmoid and tanh networks, generates 4 plots.

---

## 🔧 Installation & Setup

### Option A: Virtual Environment (Recommended)

**Why**: Isolates project dependencies from system Python.

**Steps**:

```bash
# Navigate to project
cd c:\Programming\RedNeural-v1

# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\activate

# Install packages
pip install numpy matplotlib scikit-learn

# Verify
python -c "import numpy; print('NumPy OK')"
```

**Deactivate when done**:

```bash
deactivate
```

---

### Option B: System Python

**Warning**: May conflict with other projects.

```bash
pip install --user numpy matplotlib scikit-learn
python main.py
```

---

### Troubleshooting Installation

**Problem**: `pip: command not found`

**Solution**:

```bash
python -m pip install numpy matplotlib scikit-learn
```

**Problem**: Permission denied (Linux/Mac)

**Solution**:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib scikit-learn
```

---

## ▶️ Running the Default Pipeline

### Basic Execution

```bash
python main.py
```

**Output**:

- Trains 2 networks (sigmoid + tanh)
- 300 epochs each
- ~10 minutes on typical machine
- Generates 4 PNG files (plots)
- Prints detailed results

### Run with Python Explicitly

```bash
c:\Programming\RedNeural-v1\.venv\Scripts\python.exe main.py
```

### Understanding the Output

```
[1/6] Generating synthetic 3D classification data...
  Generated: 300 samples with 3 features, 3 classes
```

→ Created training data with 3 equally-sized clusters

```
[2/6] Normalizing features (zero mean, unit variance)...
  Mean: [0.72, -0.73, 0.05]
  Std:  [2.00, 2.03, 1.81]
```

→ Standardized so all features have similar scale

```
[3/6] Splitting data (80% train, 20% test)...
  Training set: 240 samples
  Test set:     60 samples
```

→ 240 for learning, 60 for evaluation

```
[4/6] Training network with SIGMOID activation...
Epoch 30/300 | Loss: 1.131243 | Accuracy: 0.3833
Epoch 60/300 | Loss: 1.076748 | Accuracy: 0.6500
...
```

→ Loss decreasing? Good. Accuracy increasing? Good.

```
Final Results (sigmoid):
  Final Loss:       0.777054
  Train Accuracy:   0.9625 (231/240)
  Test Accuracy:    0.9667 (58/60)
```

→ 96.67% correct on unseen test data ✓

---

## ⚙️ Modifying Hyperparameters

### Hidden Layer Size

**Location**: `main.py` around line 124

```python
# Current
network = NeuralNetwork(
    input_size=3,
    hidden_size=16,        # ← Change this
    output_size=3,
    learning_rate=0.01,
    activation=activation
)

# Examples
hidden_size=8    # Simpler, faster training
hidden_size=32   # More capacity, slower training
hidden_size=64   # Large, risk overfitting on small dataset
```

**Effect on Results**:

```
Smaller (8):
  - Training: ~2x faster
  - Performance: May plateau lower
  - Risk: Underfitting (not enough capacity)

Larger (64):
  - Training: ~2x slower
  - Performance: May fit better
  - Risk: Overfitting (memorizes noise)
```

**How to test**:

```python
# Add to main.py after training loop
print(f"Final test accuracy (hidden={hidden_size}): {test_acc:.4f}")
```

---

### Learning Rate

**Location**: `main.py` around line 124

```python
network = NeuralNetwork(
    input_size=3,
    hidden_size=16,
    output_size=3,
    learning_rate=0.01,    # ← Change this
    activation=activation
)

# Examples
learning_rate=0.0001  # Slow convergence
learning_rate=0.001   # Conservative
learning_rate=0.01    # Balanced (current)
learning_rate=0.1     # Aggressive (may diverge)
learning_rate=0.5     # Too aggressive (likely diverge)
```

**Tuning guide**:

| Issue                 | Solution              |
| --------------------- | --------------------- |
| Loss not decreasing   | Increase LR (try 10×) |
| Loss oscillating/NaN  | Decrease LR (try 10÷) |
| Converging too slowly | Increase LR (try 2×)  |
| Converging well       | Leave it!             |

**Example**: If `learning_rate=0.01` doesn't work:

```python
# Try this progression
learning_rate=0.001   # 10× smaller
learning_rate=0.1     # 10× larger
learning_rate=0.05    # 5× larger
```

---

### Number of Epochs

**Location**: `main.py` around line 130

```python
network.train(
    X_train, y_train_onehot,
    epochs=300,            # ← Change this
    batch_size=None,
    verbose=True
)

# Examples
epochs=100    # Quick test, ~3 min
epochs=300    # Standard, ~10 min
epochs=1000   # Extended training, ~30 min
```

**How to decide**:

1. **Plot the loss curve** from `training_history['loss']`
2. **Look for plateau**: When loss stops decreasing
3. **Set epochs** to 20-50% beyond plateau

**Example script**:

```python
import matplotlib.pyplot as plt
network = NeuralNetwork(3, 16, 3, 0.01, 'tanh')
network.train(X_train, y_train_onehot, epochs=500, verbose=False)

plt.plot(network.training_history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('When does loss plateau?')
plt.show()
```

---

### Batch Size

**Location**: `main.py` around line 130

```python
network.train(
    X_train, y_train_onehot,
    epochs=300,
    batch_size=None,       # ← Change this (None = full batch)
    verbose=True
)

# Examples
batch_size=None    # Full batch (240 samples) — smoother curves
batch_size=32      # Mini-batch — noisier but sometimes faster
batch_size=16      # Smaller micro-batches — very noisy
```

**Full Batch** (`None`):

- Uses all 240 samples per update
- Smoother loss curve
- Deterministic (same result each time)
- Good for visualization

**Mini-Batch** (e.g., `32`):

- Uses 32 samples, then updates
- ~7-8 updates per epoch
- Noisier loss curve but good regularization
- Faster convergence sometimes

**Recommendation**: Start with `None`, experiment with `32` if underfitting.

---

## 📊 Using Custom Data

### Load from CSV File

**Assume CSV format**:

```
x, y, z, class
0.5, 1.2, -0.3, 0
1.5, 2.1, 0.8, 1
-2.0, -1.5, -2.2, 2
...
```

**Code** (replace data generation in `main.py`):

```python
import pandas as pd
import numpy as np

# Instead of: X, y = generate_3d_classification_data(...)

# Load CSV
df = pd.read_csv('c:\\path\\to\\your_data.csv')
X = df[['x', 'y', 'z']].values  # Extract 3D features
y = df['class'].values           # Extract class labels

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
print(f"Classes: {np.unique(y)}")

# Continue with rest of main.py as normal...
X_normalized, _, _ = normalize_data(X)
```

---

### Load from NumPy Array

```python
import numpy as np

# Load .npy files
X = np.load('features.npy')      # Shape: (m, 3)
y = np.load('labels.npy')        # Shape: (m,)

# Or create from list
X = np.array([[1, 2, 3], [4, 5, 6], ...])
y = np.array([0, 1, 0, ...])

# Verify shapes
assert X.shape[1] == 3, "Need 3 features (x, y, z)"
assert len(np.unique(y)) == 3, "Need 3 classes"
```

---

### Load from Python list

```python
import numpy as np

class_0_data = [[1, 2, 3], [1.1, 2.1, 2.9], ...]  # List of [x,y,z]
class_1_data = [[5, 5, 5], [5.2, 4.9, 5.1], ...]
class_2_data = [[0, 10, 0], [-0.5, 9.8, 0.2], ...]

# Convert to arrays
X = np.array(class_0_data + class_1_data + class_2_data)
y = np.array([0]*len(class_0_data) + [1]*len(class_1_data) + [2]*len(class_2_data))

# Shuffle (recommended)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
```

---

### Validate Your Data

Before training, always check:

```python
print(f"X shape: {X.shape}")  # Should be (m, 3)
print(f"y shape: {y.shape}")  # Should be (m,)
print(f"X min: {X.min()}, max: {X.max()}")  # Check ranges
print(f"y unique: {np.unique(y)}")  # Should be [0, 1, 2]
print(f"Class distribution: {np.bincount(y)}")  # Check balance

# Check for NaN/Inf
assert not np.isnan(X).any(), "X contains NaN!"
assert not np.isinf(X).any(), "X contains Inf!"
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: numpy"

**Symptom**:

```
Traceback (most recent call last):
  File "main.py", line 6, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

**Solutions** (try in order):

```bash
# 1. Install to active Python
pip install numpy matplotlib scikit-learn

# 2. Use explicit Python path
c:\Programming\RedNeural-v1\.venv\Scripts\python.exe main.py

# 3. Activate virtual environment first
.\.venv\Scripts\activate
python main.py

# 4. Use pip from Python module
python -m pip install numpy matplotlib scikit-learn
python main.py
```

---

### Problem: Loss is NaN or Infinity

**Symptom**:

```
Epoch 10/300 | Loss: nan | Accuracy: 0.2500
```

**Cause & Solutions**:

| Cause                  | Solution          | Code                                                         |
| ---------------------- | ----------------- | ------------------------------------------------------------ |
| Data not normalized    | Normalize first   | `from utils import normalize_data; X = normalize_data(X)[0]` |
| Learning rate too high | Reduce 10×        | `learning_rate=0.001`                                        |
| Bad weight init        | Check Xavier init | See `ARCHITECTURE.md`                                        |
| Numerical overflow     | Already protected | Check `activations.py` for clipping                          |

**Test script**:

```python
# Check each step
print(f"Data range: {X.min():.2f} to {X.max():.2f}")  # Should be -3 to 3
print(f"X has NaN: {np.isnan(X).any()}")              # Should be False

# Test forward pass
network = NeuralNetwork(3, 16, 3, 0.01, 'tanh')
output, _ = network.forward(X[:1])
print(f"Output: {output}")                            # Should be valid probs
print(f"Contains NaN: {np.isnan(output).any()}")     # Should be False
```

---

### Problem: Accuracy Not Improving

**Symptom**:

```
Epoch 30/300 | Loss: 0.9962 | Accuracy: 0.3333  (stuck at ~33%)
Epoch 60/300 | Loss: 0.9961 | Accuracy: 0.3333
Epoch 90/300 | Loss: 0.9960 | Accuracy: 0.3333
```

33% = random guessing on 3 classes. Network not learning!

**Checklist**:

- [ ] Data normalized? `X = normalize_data(X)[0]`
- [ ] Correct activation chosen? Try `activation='tanh'`
- [ ] Learning rate reasonable? Try `0.01` to `0.05`
- [ ] Enough hidden units? Try `hidden_size=32` (from 16)
- [ ] Enough epochs? Try `epochs=500` (from 300)

**Debug script**:

```python
network = NeuralNetwork(3, 16, 3, learning_rate=0.01, activation='tanh')
network.train(X_train, y_train_onehot, epochs=10, verbose=True)

# Plot loss curve
import matplotlib.pyplot as plt
plt.plot(network.training_history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss should decrease steadily')
plt.show()
```

If loss flat: learning rate too low or network too small.

---

### Problem: Training Very Slow

**Symptom**: Takes >30 minutes for 300 epochs

**Solutions** (fastest first):

```python
# 1. Reduce hidden layer size (2× speedup easily)
hidden_size=8  # Instead of 16

# 2. Reduce epochs (test with fewer first)
epochs=100  # Instead of 300

# 3. Reduce training data
num_samples_per_class=50  # Instead of 100

# 4. Reduce precision (advanced)
# Use float32 instead of default float64
X = X.astype(np.float32)
```

**Speed comparison**:

```
hidden_size=16: ~10 min for 300 epochs
hidden_size=8:  ~5 min for 300 epochs
hidden_size=32: ~20 min for 300 epochs
```

---

### Problem: Test Accuracy Much Lower Than Train

**Symptom**:

```
Train Accuracy: 0.9958
Test Accuracy:  0.6667  (way lower!)
```

**Cause**: Overfitting (memorized training data instead of learning patterns)

**Solutions**:

1. **Reduce hidden layer**:
   ```python
   hidden_size=8  # Instead of 16
   ```

2. **Use fewer epochs**:
   ```python
   epochs=100  # Instead of 300
   ```

3. **Add L2 regularization** (advanced):
   ```python
   # In update_weights method, modify:
   regularization = 0.001
   self.W1 -= lr * (dW1 + regularization * self.W1 / m)
   self.W2 -= lr * (dW2 + regularization * self.W2 / m)
   ```

4. **Use more training data** (if possible):
   ```python
   X, y = generate_3d_classification_data(num_samples_per_class=500)
   ```

---

## 🔨 Common Tasks

### Task 1: Try Only Tanh (Skip Sigmoid)

**Why**: Faster testing since sigmoid usually worse.

**Edit** `main.py` line ~108:

```python
# Change this:
activations = ['sigmoid', 'tanh']

# To this:
activations = ['tanh']
```

**Time saved**: ~50% (one network instead of two)

---

### Task 2: Save Trained Model

**Code**:

```python
import pickle

# After training
with open('trained_network.pkl', 'wb') as f:
    pickle.dump(network, f)

print("Model saved!")
```

**Load later**:

```python
import pickle

with open('trained_network.pkl', 'rb') as f:
    network = pickle.load(f)

# Use immediately
predictions = network.predict(new_data)
```

---

### Task 3: Train Multiple Networks with Different Hidden Sizes

**Code**:

```python
import numpy as np

hidden_sizes = [8, 16, 32, 64]
results = {}

for h_size in hidden_sizes:
    print(f"\nTraining with hidden_size={h_size}...")
    
    network = NeuralNetwork(3, h_size, 3, 0.01, 'tanh')
    network.train(X_train, y_train_onehot, epochs=300, verbose=False)
    
    test_acc = network.accuracy(X_test, y_test_onehot)
    results[h_size] = test_acc
    
    print(f"  Test Accuracy: {test_acc:.4f}")

# Print summary
print("\nHidden Size vs Test Accuracy:")
for h_size, acc in results.items():
    print(f"  {h_size}: {acc:.4f}")

# Find best
best_h = max(results, key=results.get)
print(f"\nBest: hidden_size={best_h} with accuracy={results[best_h]:.4f}")
```

---

### Task 4: Visualize Loss Curves for Multiple Learning Rates

**Code**:

```python
import matplotlib.pyplot as plt

learning_rates = [0.001, 0.01, 0.1, 0.5]
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    print(f"Training with lr={lr}...")
    network = NeuralNetwork(3, 16, 3, lr, 'tanh')
    network.train(X_train, y_train_onehot, epochs=100, verbose=False)
    
    plt.plot(network.training_history['loss'], label=f'lr={lr}')
```


plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Rate Comparison')
plt.show()
```

---

### Task 5: Evaluate on Multiple Metrics

**Code**:
```python
from sklearn.metrics import classification_report, confusion_matrix

# Get predictions
y_pred = network.predict(X_test)
y_true = np.argmax(y_test_onehot, axis=1)

# Classification report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Per-class accuracy
for class_idx in range(3):
    class_mask = y_true == class_idx
    class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
    count = np.sum(class_mask)
    print(f"Class {class_idx} ({count} samples): {class_acc:.4f}")
```

---

## 💻 Code Snippets

### Minimal Training Script

```python
import numpy as np
from data_generation import generate_3d_classification_data
from utils import normalize_data, train_test_split, one_hot_encode
from neural_network import NeuralNetwork

# Generate and prepare data
X, y = generate_3d_classification_data(100)
X_norm, _, _ = normalize_data(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y)
y_train_oh = one_hot_encode(y_train, 3)
y_test_oh = one_hot_encode(y_test, 3)

# Train
net = NeuralNetwork(3, 16, 3, 0.01, 'tanh')
net.train(X_train, y_train_oh, epochs=500, verbose=True)

# Evaluate
test_acc = net.accuracy(X_test, y_test_oh)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

### Hyperparameter Sweep Script

```python
from itertools import product

# Ranges to test
hidden_sizes = [8, 16, 32]
learning_rates = [0.001, 0.01, 0.1]
epochs_list = [100, 300, 500]

best_config = None
best_acc = 0

for h, lr, e in product(hidden_sizes, learning_rates, epochs_list):
    net = NeuralNetwork(3, h, 3, lr, 'tanh')
    net.train(X_train, y_train_oh, epochs=e, verbose=False)
    
    acc = net.accuracy(X_test, y_test_oh)
    
    if acc > best_acc:
        best_acc = acc
        best_config = (h, lr, e)
    
    print(f"h={h:2d}, lr={lr:.3f}, e={e:3d} → acc={acc:.4f}")

print(f"\nBest config: hidden_size={best_config[0]}, lr={best_config[1]}, epochs={best_config[2]}")
print(f"Best accuracy: {best_acc:.4f}")
```

---

### Visualization Script

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curve
axes[0, 0].plot(net.training_history['loss'])
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')

# Accuracy curve
axes[0, 1].plot(net.training_history['train_accuracy'])
axes[0, 1].set_title('Training Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')

# Both on same plot
axes[1, 0].plot(net.training_history['loss'], label='Loss')
axes[1, 0].plot([a*10 for a in net.training_history['train_accuracy']], label='Accuracy (×10)')
axes[1, 0].set_title('Loss vs Accuracy')
axes[1, 0].legend()

# Histogram of final loss
axes[1, 1].hist(net.training_history['loss'], bins=20)
axes[1, 1].set_title('Distribution of Loss Values')
axes[1, 1].set_xlabel('Loss')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

---

**End of Usage Guide**

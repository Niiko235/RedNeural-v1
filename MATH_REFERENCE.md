# Mathematical Reference & Detailed Derivations

Complete mathematical formulation of the neural network with all formulas and
derivations.

---

## 📐 Table of Contents

- [Notation & Conventions](#notation--conventions)
- [Forward Propagation](#forward-propagation)
- [Loss Function](#loss-function)
- [Backward Propagation (Backpropagation)](#backward-propagation-backpropagation)
- [Gradient Descent](#gradient-descent)
- [Activation Functions & Derivatives](#activation-functions--derivatives)
- [Worked Example](#worked-example)

---

## 📋 Notation & Conventions

### Matrix Notation

| Symbol | Meaning                | Shape   | Example                       |
| ------ | ---------------------- | ------- | ----------------------------- |
| **X**  | Input features         | (m, 3)  | 240 samples × 3 features      |
| **W₁** | Hidden layer weights   | (3, 16) | 3 inputs → 16 hidden units    |
| **b₁** | Hidden layer bias      | (1, 16) | Broadcasting over 240 samples |
| **Z₁** | Hidden pre-activation  | (m, 16) | Before activation function    |
| **A₁** | Hidden activation      | (m, 16) | After sigmoid/tanh            |
| **W₂** | Output layer weights   | (16, 3) | 16 hidden → 3 classes         |
| **b₂** | Output layer bias      | (1, 3)  | Broadcasting over 240 samples |
| **Z₂** | Output pre-activation  | (m, 3)  | Before softmax                |
| **A₂** | Output (probabilities) | (m, 3)  | After softmax, sums to 1      |
| **Y**  | True labels (one-hot)  | (m, 3)  | e.g., [1,0,0], [0,1,0], ...   |
| **m**  | Number of samples      | scalar  | 240 (train) or 60 (test)      |

### Operations

| Symbol | Meaning                       | Example                                   |
| ------ | ----------------------------- | ----------------------------------------- |
| **@**  | Matrix multiplication         | `X @ W = (m,3) @ (3,16) = (m,16)`         |
| **⊙**  | Element-wise multiplication   | `A ⊙ B` — multiply corresponding elements |
| **ᵀ**  | Matrix transpose              | `Xᵀ` flips rows/columns                   |
| **∂**  | Partial derivative            | `∂L/∂W` — how loss changes with weight    |
| **∇**  | Gradient (vector of partials) | `∇W L` = all partial derivatives          |

---

## 🔄 Forward Propagation

### Layer 1: Hidden Layer

**Input**: X (m, 3)

**Computation**:

```
Z₁ = X @ W₁ + b₁
```

**Dimensions**:

```
(m, 3) @ (3, 16) + (1, 16) = (m, 16)
         └─→ (m, 16)
```

**Bias Broadcasting**:

```
When adding (m, 16) + (1, 16), the (1, 16) is repeated m times:
[[b₁₁, b₁₂, ..., b₁₁₆],
 [b₁₁, b₁₂, ..., b₁₆],
 ...  (m times)
 [b₁₁, b₁₂, ..., b₁₆]]
```

**Example (small numbers)**:

```
X = [[x₁, x₂, x₃],
     [x₄, x₅, x₆]]  (2 samples, 3 features)

W₁ = [[w₁₁, w₁₂],    (3 features → 2 hidden units)
      [w₂₁, w₂₂],
      [w₃₁, w₃₂]]

X @ W₁ = [[x₁·w₁₁ + x₂·w₂₁ + x₃·w₃₁,  x₁·w₁₂ + x₂·w₂₂ + x₃·w₃₂],
          [x₄·w₁₁ + x₅·w₂₁ + x₆·w₃₁,  x₄·w₁₂ + x₅·w₂₂ + x₆·w₃₂]]
```

**Activation**:

```
A₁ = σ(Z₁)  or  A₁ = tanh(Z₁)
```

Where σ is sigmoid or tanh (element-wise application).

---

### Layer 2: Output Layer

**Input**: A₁ (m, 16)

**Computation**:

```
Z₂ = A₁ @ W₂ + b₂
```

**Dimensions**:

```
(m, 16) @ (16, 3) + (1, 3) = (m, 3)
```

**Softmax**:

```
A₂ = softmax(Z₂)
```

**Detailed softmax formula** (for each sample i):

```
A₂[i, j] = exp(Z₂[i, j]) / Σₖ exp(Z₂[i, k])
```

Where:

- j ∈ {0, 1, 2} (output classes)
- k sums over all 3 classes
- Result: probability distribution per sample

**Numerical stability** (what code actually does):

```
max_per_row = max(Z₂[i, :]) for each i
Z₂_shifted = Z₂ - max_per_row  (subtract max per row)
A₂[i, j] = exp(Z₂_shifted[i, j]) / Σₖ exp(Z₂_shifted[i, k])
```

**Why subtract max?**

- Prevents `exp(large number)` overflow
- Mathematically identical: `exp(x - c) / Σ exp(x - c) = exp(x) / Σ exp(x)`
- Numerical stability without loss of accuracy

---

## 🎯 Loss Function

### Cross-Entropy Loss

**Formula** (for multi-class):

```
L = -1/m · Σᵢ₌₁ᵐ Σⱼ₌₁³ Yᵢⱼ · log(A₂ᵢⱼ)
```

Where:

- m = number of samples (240)
- Yᵢⱼ = 1 if sample i is class j, else 0 (one-hot)
- A₂ᵢⱼ = predicted probability sample i is class j
- log = natural logarithm

### Simplified for One-Hot

Since Y is one-hot (each row has exactly one 1):

```
L = -1/m · Σᵢ₌₁ᵐ log(A₂ᵢ,true_classᵢ)
```

**Example**:

```
Sample 1: Y = [1, 0, 0], A₂ = [0.7, 0.2, 0.1]
Contribution: -log(0.7) ≈ 0.357

Sample 2: Y = [0, 1, 0], A₂ = [0.1, 0.8, 0.1]
Contribution: -log(0.8) ≈ 0.223

Average loss = (0.357 + 0.223) / 2 ≈ 0.290
```

### Numerical Stability

**Problem**: `log(0) = -∞`

**Solution** (in code):

```python
A₂_clipped = np.clip(A₂, 1e-7, 1-1e-7)
loss = -np.mean(Y * np.log(A₂_clipped))
```

Clipping to [1e-7, 1-1e-7] ensures:

- log(1e-7) ≈ -16.1 (manageable)
- No NaN values
- Minimal change to actual loss (~0.0000001%)

---

## 🔙 Backward Propagation (Backpropagation)

### Overview

**Goal**: Compute gradients `∂L/∂W` and `∂L/∂b` for each parameter.

**Method**: Chain rule applied backwards through the network.

### Step 1: Output Layer Gradient

**Derivative of softmax + cross-entropy** (key insight):

```
∂L/∂Z₂ = A₂ - Y
```

**Derivation** (simplified):

```
L = -Σ Y · log(A₂)

∂A₂/∂Z₂ = ∂softmax/∂Z₂  (Jacobian)
         = diag(A₂) - A₂·A₂ᵀ  (diagonal - outer product)

∂L/∂Z₂ = ∂L/∂A₂ · ∂A₂/∂Z₂
       = (-Y/A₂) · [diag(A₂) - A₂·A₂ᵀ]
       = A₂ - Y  (after simplification)
```

This is a beautiful result: the gradient is just difference between prediction
and truth!

**Result** (shape):

```
dZ₂ = A₂ - Y  (m, 3)
```

---

### Step 2: Backpropagate to Hidden Layer

**Gradient w.r.t. hidden activation**:

```
∂L/∂A₁ = ∂L/∂Z₂ · ∂Z₂/∂A₁
       = dZ₂ @ W₂ᵀ
```

**Dimensions**:

```
(m, 3) @ (3, 16) = (m, 16)
```

**Details**:

```
Z₂ = A₁ @ W₂ + b₂

∂Z₂ᵢⱼ/∂A₁ᵢₖ = W₂ₖⱼ

Therefore:
∂L/∂A₁ = ∂L/∂Z₂ @ W₂ᵀ
```

---

### Step 3: Apply Activation Derivative

**Gradient w.r.t. hidden pre-activation**:

```
∂L/∂Z₁ = ∂L/∂A₁ · ∂A₁/∂Z₁  (element-wise multiply)
```

#### For Sigmoid:

```
∂A₁/∂Z₁ = A₁ · (1 - A₁)  where A₁ = sigmoid(Z₁)

dZ₁ = dA₁ ⊙ [A₁ · (1 - A₁)]
```

#### For Tanh:

```
∂A₁/∂Z₁ = 1 - A₁²  where A₁ = tanh(Z₁)

dZ₁ = dA₁ ⊙ (1 - A₁²)
```

**Why element-wise?**

```
Each Z₁ᵢⱼ → A₁ᵢⱼ → activation derivative ∂A₁ᵢⱼ/∂Z₁ᵢⱼ
Applied independently to each element.
```

---

### Step 4: Compute Weight Gradients

**Hidden layer to output weights**:

```
∂L/∂W₂ = 1/m · A₁ᵀ @ dZ₂
```

**Derivation**:

```
Z₂ = A₁ @ W₂ + b₂

∂Z₂ᵢⱼ/∂W₂ₖⱼ = A₁ᵢₖ

∂L/∂W₂ₖⱼ = Σᵢ (∂L/∂Z₂ᵢⱼ · ∂Z₂ᵢⱼ/∂W₂ₖⱼ)
         = Σᵢ (dZ₂ᵢⱼ · A₁ᵢₖ)
         = (A₁ᵀ @ dZ₂)ₖⱼ

Average over m samples: divide by m
```

**Dimensions**:

```
(16, m) @ (m, 3) / m = (16, 3)
```

---

**Input layer to hidden weights**:

```
∂L/∂W₁ = 1/m · Xᵀ @ dZ₁
```

**Dimensions**:

```
(3, m) @ (m, 16) / m = (3, 16)
```

---

### Step 5: Compute Bias Gradients

**Output layer bias**:

```
∂L/∂b₂ = 1/m · Σᵢ dZ₂ᵢ  (sum over samples, average)
```

**Dimensions**:

```
Sum of (m, 3) by rows → (1, 3)
```

---

**Hidden layer bias**:

```
∂L/∂b₁ = 1/m · Σᵢ dZ₁ᵢ  (sum over samples, average)
```

**Dimensions**:

```
Sum of (m, 16) by rows → (1, 16)
```

---

## 📉 Gradient Descent

### Parameter Update Rule

**General form**:

```
W := W - α · ∂L/∂W
b := b - α · ∂L/∂b
```

Where α (alpha) is the learning rate.

### Applied to Our Network

**Output layer**:

```
W₂ := W₂ - α · dW₂
b₂ := b₂ - α · db₂
```

**Hidden layer**:

```
W₁ := W₁ - α · dW₁
b₁ := b₁ - α · db₁
```

### Intuition

**What's happening**:

1. Compute loss L (how wrong predictions are)
2. Compute ∂L/∂W (in which direction does W need to move to reduce loss?)
3. Move W in opposite direction of gradient (steepest descent)

**Visual**:

```
Loss surface
    ▲
    │     ╱╲     Current position
    │    ╱  ╲       ●
    │   ╱    ╲     ╱
    │  ╱      ╲  ╱   Negative gradient
    │ ╱        ●←──── Points this way
    │____________→ W
           
Each step: move along negative gradient to lower loss
```

### Learning Rate Effect

| Learning Rate | Behavior     | Issues                 |
| ------------- | ------------ | ---------------------- |
| α = 0.1       | Large steps  | May overshoot, diverge |
| α = 0.01      | Medium steps | **Sweet spot** ✓       |
| α = 0.001     | Small steps  | Slow convergence       |
| α = 0.00001   | Tiny steps   | Very slow              |

---

## 🔧 Activation Functions & Derivatives

### Sigmoid

**Function**:

```
σ(z) = 1 / (1 + e^(-z))
```

**Range**: (0, 1)

**Derivative** (derivation):

```
Let u = 1 + e^(-z)
σ(z) = 1/u = u^(-1)

∂σ/∂z = -1 · u^(-2) · (-e^(-z))
      = e^(-z) / (1 + e^(-z))²
      = [1/(1 + e^(-z))] · [(1 + e^(-z) - 1)/(1 + e^(-z))]
      = σ(z) · (1 - σ(z))
```

**So**: ∂σ/∂z = σ(z) · (1 - σ(z))

**Efficient computation** (using cached A₁):

```python
if A₁ already computed from forward pass:
    dZ1 = dA1 * (A1 * (1 - A1))  # Instead of recomputing sigmoid
```

---

### Tanh

**Function**:

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
         = sinh(z) / cosh(z)
```

**Range**: (-1, 1)

**Derivative** (derivation):

```
∂tanh/∂z = 1 - tanh²(z)
```

**Proof**:

```
Let y = tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

∂y/∂z = [2e^z · (e^z + e^(-z)) - (e^z - e^(-z)) · 2e^(-z)] / (e^z + e^(-z))²
```

      = 2[e^(2z) + 1 - 1 - e^(-2z)] / (e^z + e^(-z))²
      = 4 / (e^z + e^(-z))²

Also: (e^z + e^(-z))² = (e^z - e^(-z))² + 4
So: ∂y/∂z = 1 - [(e^z - e^(-z)) / (e^z + e^(-z))]²
          = 1 - tanh²(z)
```

**Efficient computation** (using cached A₁):
```python
if A₁ already computed from forward pass:
    dZ1 = dA1 * (1 - A1**2)  # Much faster than recomputing tanh
```

---

### Softmax

**Function** (element-wise for sample i):
```
softmax(z)ⱼ = e^(zⱼ) / Σₖ e^(zₖ)
```

**Properties**:
- Output sums to 1 (probability distribution)
- All outputs positive
- Differentiable everywhere

**Derivative** (for cross-entropy loss):
```
∂L/∂z = softmax(z) - y
```

(This is what we use in code — the beautiful result mentioned earlier)

---

## 📝 Worked Example

### Setting Dimensions and Initialization

**Given**:
- 2 training samples, 3 features, 3 classes, 2 hidden units (for simplicity)

**Network**:
```
Input (2, 3) → Hidden (2, 2, activation) → Output (2, 3)
```

**Weights** (arbitrary small values):
```
W₁ = [[0.1, 0.2],     (3 × 2)
      [0.3, 0.4],
      [0.5, 0.6]]

b₁ = [0.01, 0.02]     (1 × 2)

W₂ = [[0.7, 0.8, 0.9],    (2 × 3)
      [0.1, 0.2, 0.3]]

b₂ = [0.01, 0.02, 0.03]   (1 × 3)
```

**Input**:
```
X = [[1, 2, 3],       (2 × 3)
     [0.5, 1, 1.5]]

Y = [[1, 0, 0],       (2 × 3, one-hot)
     [0, 0, 1]]
```

---

### Forward Pass

**Hidden pre-activation**:
```
Z₁ = X @ W₁ + b₁

Z₁[0] = [1, 2, 3] @ [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] + [0.01, 0.02]
      = [1×0.1 + 2×0.3 + 3×0.5, 1×0.2 + 2×0.4 + 3×0.6] + [0.01, 0.02]
      = [0.1 + 0.6 + 1.5, 0.2 + 0.8 + 1.8] + [0.01, 0.02]
      = [2.2, 2.8] + [0.01, 0.02]
      = [2.21, 2.82]

Z₁[1] = [0.5, 1, 1.5] @ W₁ + [0.01, 0.02]
      = [0.5×0.1 + 1×0.3 + 1.5×0.5, 0.5×0.2 + 1×0.4 + 1.5×0.6] + [0.01, 0.02]
      = [0.05 + 0.3 + 0.75, 0.1 + 0.4 + 0.9] + [0.01, 0.02]
      = [1.1, 1.4] + [0.01, 0.02]
      = [1.11, 1.42]

Z₁ = [[2.21, 2.82],
      [1.11, 1.42]]
```

**Hidden activation** (using sigmoid for simplicity):
```
σ(z) = 1/(1 + e^(-z))

A₁[0,0] = 1/(1 + e^(-2.21)) ≈ 1/(1 + 0.0109) ≈ 0.989
A₁[0,1] = 1/(1 + e^(-2.82)) ≈ 1/(1 + 0.0061) ≈ 0.994
A₁[1,0] = 1/(1 + e^(-1.11)) ≈ 1/(1 + 0.330) ≈ 0.752
A₁[1,1] = 1/(1 + e^(-1.42)) ≈ 1/(1 + 0.242) ≈ 0.805

A₁ ≈ [[0.989, 0.994],
      [0.752, 0.805]]
```

...continuing with output layer would give full example, but this shows the pattern!

---

**End of Mathematical Reference**

# How Causal TGAN Generates Synthetic Data from SCMs

## The Core Question

**How does Causal TGAN draw inference from Structural Causal Models (SCMs) while generating new synthetic data?**

---

## Part 1: SCM Formulation Recap

### What is an SCM?

A Structural Causal Model is defined as a triplet: **`<X, F, U>`**

```
X_i = f_i(PA_i, U_i)
```

Where:
- **X_i**: The i-th observable variable (e.g., `protocol_type`, `service`, `src_bytes`)
- **PA_i**: Parent variables of X_i in the causal graph
- **f_i**: The causal mechanism (a function)
- **U_i**: Exogenous variable (random noise, independent across variables)

### Traditional SCM Example

In traditional causal inference:

```python
# Example: Simple SCM for network traffic

# U1, U2, U3 are random noise (exogenous variables)
protocol = f1(U1)                           # Root node
service = f2(protocol, U2)                  # Depends on protocol
duration = f3(protocol, service, U3)        # Depends on both

# Where f1, f2, f3 are specified functions
```

### Key Insight

In traditional SCMs, the functions `f_i` are **given** or **assumed** (e.g., linear functions, polynomials).

In **Causal TGAN**, the functions `f_i` are **neural networks learned from data**.

---

## Part 2: How Causal TGAN Learns the SCM

### Step 1: Replace Functions with Neural Networks

Instead of specifying `f_i`, Causal TGAN **learns** it:

```
X_i = NN_i(PA_i, U_i)
```

Where `NN_i` is a neural network (called the "causal mechanism") for variable X_i.

### Step 2: Neural Network Architecture for Each Variable

From the code (`model/module/generator.py:6-42`):

```python
class base_continuous_generator(nn.Module):
    def __init__(self, parent_dim, z_dim, feature_dim):
        # Input: concatenated [parent_values, noise]
        # Output: generated value for this variable

        self.model = nn.Sequential(
            Linear(parent_dim + z_dim, 64),
            BatchNorm1d + LeakyReLU,
            Linear(64, 128),
            BatchNorm1d + LeakyReLU,
            Linear(128, 128),
            BatchNorm1d + LeakyReLU,
            Linear(128, feature_dim),
            Activation(tanh or gumbel_softmax)  # Depends on variable type
        )

    def forward(self, noise, parents):
        # Concatenate parent values and noise
        x = concat([parents, noise])
        # Pass through neural network
        x = self.model(x)
        return x  # Generated value
```

**What this means:**
- Each variable gets its own neural network
- Input = [values of parent variables, random noise]
- Output = generated value for this variable
- The network learns the causal mechanism from data

### Step 3: Training Process

During training (GAN framework):

```python
for epoch in epochs:
    for batch in real_data:
        # 1. Generate fake samples using current causal mechanisms
        fake_data = causal_generator.sample(batch_size)

        # 2. Discriminator evaluates real vs fake
        D_real = discriminator(real_data)
        D_fake = discriminator(fake_data)

        # 3. Update discriminator to distinguish real from fake
        D_loss = D_fake - D_real + gradient_penalty

        # 4. Update all causal mechanisms (neural networks) to fool discriminator
        G_loss = -D_fake

        # This trains each NN_i to learn the true causal mechanism f_i
```

**Key Point:** The neural networks learn to approximate the true causal mechanisms that generated the real data.

---

## Part 3: How Generation Works (The Answer to Your Question)

### The Autoregressive Generation Process

From `model/module/generator.py:156-173`:

```python
def sample(self, batch_size):
    """
    Generate synthetic samples by following the causal graph
    """
    # Initialize empty tensor for generated samples
    fake_sample = torch.zeros((batch_size, total_dim))

    # Generate variables in TOPOLOGICAL ORDER (parents before children)
    for idx in range(len(nodes)):
        current_node = nodes[idx]

        # 1. Sample random noise U_i for this variable
        exogenous_var = torch.randn(batch_size, z_dim)

        # 2. Get parent values (already generated in previous iterations)
        parents_name = current_node.parents
        parents_val = fake_sample[:, parent_indices]

        # 3. Generate this variable using its learned causal mechanism
        # X_i = NN_i(PA_i, U_i)
        generated_value = current_node.causal_mechanism(exogenous_var, parents_val)

        # 4. Store generated value
        fake_sample[:, current_node_indices] = generated_value

    return fake_sample
```

### Step-by-Step Example: Generating One Sample

Let's generate a single network traffic record with this causal graph:

```python
causal_graph = [
    ['protocol_type', []],                    # Root
    ['service', ['protocol_type']],           # Child of protocol
    ['logged_in', ['service']],               # Child of service
    ['root_shell', ['logged_in']],            # Child of logged_in
    ['label', ['protocol_type', 'logged_in', 'root_shell']]  # Multiple parents
]
```

**Generation Process:**

#### Iteration 1: Generate `protocol_type` (Root Node)

```python
# No parents, only noise
U1 = sample_noise()  # e.g., [-0.34, 1.22]

# Use learned causal mechanism NN1
protocol_type = NN1(U1)  # e.g., [0.1, 0.2, 0.7] → TCP (after softmax)

# Store in fake_sample
fake_sample['protocol_type'] = 'TCP'
```

#### Iteration 2: Generate `service` (Depends on protocol_type)

```python
# Get parent value (already generated)
parent_value = fake_sample['protocol_type']  # 'TCP' → [1, 0, 0] one-hot

# Sample new noise
U2 = sample_noise()  # e.g., [0.56, -0.89]

# Use learned causal mechanism NN2
service = NN2(concat([parent_value, U2]))  # Input: [1, 0, 0, 0.56, -0.89]
                                           # Output: e.g., 'http' (softmax over services)

# Store in fake_sample
fake_sample['service'] = 'http'
```

#### Iteration 3: Generate `logged_in` (Depends on service)

```python
parent_value = fake_sample['service']  # 'http' → [...] one-hot

U3 = sample_noise()

logged_in = NN3(concat([parent_value, U3]))  # Output: 0 or 1

fake_sample['logged_in'] = 1
```

#### Iteration 4: Generate `root_shell` (Depends on logged_in)

```python
parent_value = fake_sample['logged_in']  # 1

U4 = sample_noise()

root_shell = NN4(concat([parent_value, U4]))  # Output: 0 or 1

fake_sample['root_shell'] = 0
```

#### Iteration 5: Generate `label` (Depends on protocol, logged_in, root_shell)

```python
# Multiple parents - concatenate all
parent_values = concat([
    fake_sample['protocol_type'],  # [1, 0, 0]
    fake_sample['logged_in'],      # 1
    fake_sample['root_shell']      # 0
])

U5 = sample_noise()

label = NN5(concat([parent_values, U5]))  # Output: 'normal' or 'attack'

fake_sample['label'] = 'normal'
```

**Final Generated Sample:**
```python
{
    'protocol_type': 'TCP',
    'service': 'http',
    'logged_in': 1,
    'root_shell': 0,
    'label': 'normal'
}
```

---

## Part 4: Why This Works (The Causal Inference)

### The Power of Topological Ordering

The key to generating from SCMs is **topological ordering**:

1. **Parents are always generated before children**
2. This respects the causal flow of information
3. Each variable is conditioned on its true causes (parents)

From `model/module/generator.py:175-206`:

```python
def node_order(self):
    """
    Topology sorting: Order nodes from root to leaf
    """
    ordered_nodes = []

    # Start with root nodes (no parents)
    for node in graph:
        if node.parents == []:
            ordered_nodes.append(node)

    # Add nodes whose parents are all already in ordered_nodes
    while len(graph) > 0:
        for node in graph:
            if all(parent in ordered_nodes for parent in node.parents):
                ordered_nodes.append(node)
                graph.remove(node)

    return ordered_nodes
```

### Visual Example

```
Causal Graph:
A → B → D
    ↓
    C → E

Generation Order (Topological):
1. A (no parents)
2. B (parent A already generated)
3. C (parent B already generated)
4. D (parent B already generated)
5. E (parents C and D already generated)
```

### Why Not Just Use a Single Generator?

**Traditional GAN (like CTGAN):**
```python
# Single generator for all variables
noise = sample_noise(dim=128)
fake_sample = single_generator(noise)
# All variables generated simultaneously
# No explicit causal structure
```

**Causal TGAN:**
```python
# Separate generator for each variable
for each variable in topological_order:
    noise = sample_noise(dim=2)  # Smaller noise dimension per variable
    parent_values = already_generated_parents
    value = variable_generator(concat([parent_values, noise]))
# Variables generated sequentially
# Respects causal dependencies
```

**Advantages:**
1. **Preserves causal relationships**: B is truly caused by A (not just correlated)
2. **Interpretable**: Each generator is a learned causal mechanism
3. **Controllable**: Can intervene on variables (set values and generate downstream)
4. **Better generalization**: Learns true data-generating process

---

## Part 5: Mathematical Formulation

### Joint Distribution Factorization

**Traditional joint distribution:**
```
P(X1, X2, ..., Xn) = complex multivariate distribution
```

**Causal factorization (using causal graph):**
```
P(X1, X2, ..., Xn) = ∏ P(Xi | PA_i)
                     i=1 to n
```

Where `PA_i` are the parents of `Xi`.

### How Causal TGAN Learns This

Each neural network `NN_i` learns the conditional distribution `P(Xi | PA_i)`:

```
P(Xi | PA_i) ≈ NN_i(PA_i, Ui)
```

Where:
- `NN_i` outputs parameters of a distribution (or directly generates a sample)
- `Ui` introduces randomness (diversity in generation)
- Training via GAN ensures `NN_i` matches the true conditional distribution

### Generation is Sampling from the Joint Distribution

```python
# Sample from P(X1, X2, ..., Xn)

# Using causal factorization:
X1 ~ NN1(U1)                    # Sample from P(X1)
X2 ~ NN2(X1, U2)                # Sample from P(X2 | X1)
X3 ~ NN3(X1, X2, U3)            # Sample from P(X3 | X1, X2)
...

# Result: (X1, X2, X3, ...) ~ P(X1, X2, ..., Xn)
```

This is exactly what the `sample()` function does!

---

## Part 6: Concrete Code Walkthrough

Let's trace through the actual code for generating 3 samples:

### Setup

```python
# From model/causalTGAN.py:196-202
def sample(self, batch_size):
    if self.condGAN is None:
        # Pure causal generation
        return self.causal_controller.sample(batch_size)
    else:
        # Hybrid: causal + conditional GAN
        condvec = self.causal_controller.sample(batch_size)
        return self.condGAN.sample(batch_size, condvec)
```

### Pure Causal Generation

```python
# From model/module/generator.py:156-173
def sample(self, batch_size=3):  # Generate 3 samples

    # Step 1: Initialize empty tensor
    # Shape: (3 samples, total_feature_dim)
    fake_sample = torch.zeros((3, 150))  # Assume 150 total dimensions

    # Step 2: Loop through nodes in topological order
    for idx in range(5):  # 5 variables in graph

        current_node = self.nodes[idx]

        # Step 3: Sample exogenous noise for this variable
        # Shape: (3, 2) - 3 samples, 2 noise dimensions
        U = torch.randn(3, 2)
        # U = [[-0.5, 0.3],
        #      [ 1.2, -0.8],
        #      [ 0.1, 0.9]]

        # Step 4: Get parent values from already-generated part
        parents_name = current_node.parents  # e.g., ['protocol_type', 'service']
        parents_idx = [10, 11, 12, 25, 26, 27, ...]  # Column indices

        if parents_idx != []:
            parents_val = fake_sample[:, parents_idx]
            # Shape: (3, parent_dim)
        else:
            parents_val = None  # Root node

        # Step 5: Generate value using causal mechanism (neural network)
        generated_val = current_node.cal_val(U, parents_val)
        # This calls: NN(concat([parents_val, U]))
        # Output shape: (3, feature_dim)

        # Step 6: Store in the fake_sample tensor
        val_position = [50, 51, 52, 53]  # Positions for this variable
        fake_sample[:, val_position] = generated_val

    # Step 7: Return complete samples
    return fake_sample
    # Shape: (3, 150) - 3 complete samples with all 150 features
```

### What Happens Inside cal_val()

```python
# From model/module/generator.py:105-113
def cal_val(self, noises, parents):
    """
    Generate value for this node
    """
    # Input:
    #   noises: (batch_size, z_dim) e.g., (3, 2)
    #   parents: (batch_size, parent_dim) e.g., (3, 15) or None

    # Causal mechanism is a neural network
    self.val = self.causal_mechanism(noises, parents)

    return self.val
```

### What Happens Inside causal_mechanism.forward()

```python
# From model/module/generator.py:29-41
def forward(self, noise, parents):
    # Step 1: Concatenate parents and noise
    if parents is not None:
        x = torch.cat([parents, noise], dim=-1)
        # Shape: (3, parent_dim + z_dim) e.g., (3, 17)
    else:
        x = noise
        # Shape: (3, z_dim) e.g., (3, 2)

    # Step 2: Pass through neural network
    x = self.model(x)  # MLP with BatchNorm and LeakyReLU
    # Shape: (3, feature_dim) e.g., (3, 4) for one-hot encoded variable

    # Step 3: Apply activation
    if self.feature_dim == 1:
        x = torch.tanh(x)  # Continuous variable → [-1, 1]
    else:
        # Discrete variable → probability distribution
        x_normalized = torch.tanh(x[:, 0])  # First component
        x_categorical = gumbel_softmax(x[:, 1:])  # Rest is one-hot
        x = torch.cat([x_normalized, x_categorical], dim=1)

    return x
    # Shape: (3, feature_dim)
```

---

## Part 7: Key Takeaways

### How Causal TGAN Uses SCMs for Generation

1. **Training Phase:**
   - Learn neural network `NN_i` for each variable `Xi`
   - Each `NN_i` approximates the causal mechanism `f_i`
   - Training via GAN ensures realistic samples

2. **Generation Phase:**
   - Follow topological order (parents before children)
   - For each variable `Xi`:
     - Sample random noise `Ui`
     - Get parent values `PA_i` (already generated)
     - Generate `Xi = NN_i(PA_i, Ui)`
   - Return complete sample

3. **Why It's "Causal":**
   - Each variable is generated by its true causes (parents)
   - Noise `Ui` adds diversity (different samples each time)
   - Structure encodes domain knowledge (attack logic)

### Comparison with Other Methods

| Method | Generation Process |
|--------|-------------------|
| **CTGAN** | Single generator: `all_variables = NN(noise)` |
| **Causal TGAN** | Sequential: `Xi = NN_i(parents, noise_i)` |
| **SMOTE** | Interpolation: `new = old1 + α*(old2 - old1)` |

### The "Inference" Part

**What inference happens?**

The term "inference" in SCMs usually means:
1. **Observational inference**: Given some variables, predict others
2. **Interventional inference**: Set a variable, see downstream effects
3. **Counterfactual inference**: "What if X had been different?"

**In Causal TGAN generation:**
- We're doing **ancestral sampling** from the learned SCM
- Each step is a forward pass: cause → effect
- No backward inference (that would be different)

**But you CAN do inference with trained Causal TGAN:**

```python
# Conditional generation (intervention)
protocol_type = 'TCP'  # Fix protocol
service = NN2(protocol_type, U2)  # Generate service given protocol
logged_in = NN3(service, U3)      # Generate logged_in given service
# ... continue

# This lets you generate "TCP attacks" specifically
```

---

## Part 8: Visualization

### Data Flow Diagram

```
┌─────────────────────────────────────────────┐
│  Causal TGAN Generation Process             │
└─────────────────────────────────────────────┘

Input: batch_size = 1000 (want 1000 samples)

┌──────────────────┐
│ Topological Sort │
│ Causal Graph     │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Iteration 1: Generate Root Nodes            │
├─────────────────────────────────────────────┤
│ U1 ~ N(0,1)          [1000 x 2]            │
│ X1 = NN1(U1)         [1000 x 3]            │
│ Store X1 in output                          │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Iteration 2: Generate Children of X1        │
├─────────────────────────────────────────────┤
│ U2 ~ N(0,1)          [1000 x 2]            │
│ PA2 = X1             [1000 x 3]            │
│ X2 = NN2([PA2, U2])  [1000 x 70]           │
│ Store X2 in output                          │
└────────┬────────────────────────────────────┘
         │
         ▼
         ...
         │
         ▼
┌─────────────────────────────────────────────┐
│ Iteration N: Generate Leaf Nodes            │
├─────────────────────────────────────────────┤
│ UN ~ N(0,1)          [1000 x 2]            │
│ PAN = [X1, X3, X5]   [1000 x 15]           │
│ XN = NNN([PAN, UN])  [1000 x 5]            │
│ Store XN in output                          │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Output: Complete Samples [1000 x 150]       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Inverse Transform to Original Space         │
│ (Denormalize, decode one-hot, etc.)        │
└─────────────────────────────────────────────┘
         │
         ▼
    CSV Output
```

### Network Attack Example Flow

```
Causal Graph for Network Attack:

    ┌──────────────┐
    │ protocol_type│ (U1)
    └───────┬──────┘
            │
            ▼
        ┌───────┐
        │service│ (U2)
        └───┬───┘
            │
            ▼
      ┌──────────┐
      │ logged_in│ (U3)
      └─────┬────┘
            │
            ▼
      ┌──────────┐
      │root_shell│ (U4)
      └─────┬────┘
            │
            ▼
        ┌─────┐
        │label│ (U5, also depends on protocol_type)
        └─────┘

Generation:
1. U1 = [0.5, -0.3] → NN1 → protocol = 'TCP'
2. U2 = [-0.8, 1.2] + 'TCP' → NN2 → service = 'http'
3. U3 = [0.1, 0.9] + 'http' → NN3 → logged_in = 1
4. U4 = [-1.5, 0.2] + logged_in=1 → NN4 → root_shell = 0
5. U5 = [0.7, -0.4] + 'TCP' + logged_in=1 + root_shell=0
   → NN5 → label = 'normal'

Result: One complete synthetic network record
```

---

## Conclusion

**How Causal TGAN draws inference from SCMs during generation:**

1. **Learns causal mechanisms** (neural networks) during training
2. **Generates samples autoregressively** following causal order
3. **Each variable is generated by its causes** (parents + noise)
4. **Respects the causal structure** encoded in the graph

This is fundamentally different from traditional GANs that generate all variables at once, and from SMOTE that just interpolates. Causal TGAN explicitly models the data-generating process through learned SCMs.

The "inference" is the forward propagation through the causal graph: causes → effects, using the learned mechanisms, with randomness from exogenous variables to create diversity.

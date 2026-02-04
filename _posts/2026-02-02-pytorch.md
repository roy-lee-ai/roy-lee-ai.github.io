---
title: "Pytorch note"
date: 2025-02-02
categories: [Python, PyTorch]
tags: [python, pyTorch]     # TAG names should always be lowercase
math: true
---

## Before begin
### In PyTorch, they call all 1D, 2D, 3D a tensor.

---

## 1. Tensor Creation

### From Python List
You can create a tensor directly from a standard Python list.

```python
import torch
import numpy as np

list_data = [[10, 20], [30, 40]]

# Create tensor from list
tensor1 = torch.Tensor(list_data)

print(tensor1)
# tensor([[10., 20.],
#         [30., 40.]])

# Check attributes
print(f"shape: {tensor1.shape}")   # Shape of the tensor
print(f"dtype: {tensor1.dtype}")   # Data type (default is float32)
print(f"device: {tensor1.device}") # Device location (cpu/cuda)

```

{: .prompt-tip }

> **Using GPU**: If `torch.cuda.is_available()` returns `True`, you can move the tensor to the GPU memory using `.to("cuda")`.

### From NumPy

When converting a NumPy array to a tensor, use `torch.from_numpy()`.

```python
numpy_data = np.array(list_data)
tensor2 = torch.from_numpy(numpy_data)

print(f"dtype: {tensor2.dtype}") 
# Result: torch.int64 (Inherits the integer type from NumPy)

```

{: .prompt-warning }

> **Watch out for Data Types**: Tensors created from NumPy integer arrays default to `int64`. Since Deep Learning models typically require floating-point numbers, it is essential to cast the type using `.float()`.

```python
# Cast to float during creation
tensor2 = torch.from_numpy(numpy_data).float()

```

### Random Data Generation

These functions are frequently used for initializing weights in neural networks.

| Function | Description | Distribution |
| --- | --- | --- |
| `torch.rand(m, n)` | Generates values between 0 and 1 | Uniform Distribution |
| `torch.randn(m, n)` | Generates values with Mean=0, Var=1 | Normal (Gaussian) Distribution |

---

## 2. Tensor Operations

### Indexing & Slicing

Used to access specific elements or extract sub-tensors.

**Base Data (`tensor6`, `tensor7`)**

| `tensor6` | 1 | 2 | 3 |
| --- | --- | --- | --- |
| **Row 1** | 4 | 5 | 6 |

| `tensor7` | 7 | 8 | 9 |
| --- | --- | --- | --- |
| **Row 1** | 10 | 11 | 12 |

**Slicing Examples**

```python
# 1. All data in the first row
print(tensor6[0]) 
# [1., 2., 3.]

# 2. All rows, from the second column to the end
print(tensor6[:, 1:]) 
# [[2., 3.],
#  [5., 6.]]

# 3. Intersection: Rows 0~1 and Columns 0~1
print(tensor7[0:2, 0:-1])
# [[ 7.,  8.],
#  [10., 11.]]

```

### Multiplication: Element-wise vs Matrix Multiplication

It is important to distinguish between simple element-wise multiplication and matrix multiplication (dot product).

1. **`mul` (Element-wise Product)**: Multiplies elements at the same position. (Same as `*` operator).
2. **`matmul` (Matrix Multiplication)**: Performs matrix multiplication. (Same as `@` operator).

```python
# Element-wise product
tensor8 = tensor6.mul(tensor7) 
# Result shape: (2, 3) remains the same

# Matrix Multiplication (Error Case)
# tensor6(2x3) @ tensor7(2x3) -> Shape Mismatch Error!
# The columns of the first matrix (3) must match the rows of the second (2).

```

{: .prompt-info }

> **Reshape (`view`)**: To perform matrix multiplication, we need to change the shape of `tensor7` to `(3, 2)`.

```python
# Perform matrix multiplication after reshaping
# (2, 3) @ (3, 2) -> (2, 2) result
tensor9 = tensor6.matmul(tensor7.view(3, 2))

```

---

## 3. Tensor Concatenation

You can join tensors using `torch.cat`. The `dim` parameter determines the direction of concatenation.

### `dim=0` (Vertical Concatenation)

Stacks tensors vertically (along the rows). The number of columns remains the same.

```python
torch.cat([tensor6, tensor7], dim=0)

```

| Result | Col 0 | Col 1 | Col 2 |
| --- | --- | --- | --- |
| **t6** | 1 | 2 | 3 |
| **t6** | 4 | 5 | 6 |
| **t7** | 7 | 8 | 9 |
| **t7** | 10 | 11 | 12 |

### `dim=1` (Horizontal Concatenation)

Stacks tensors horizontally (along the columns). The number of rows remains the same.

```python
torch.cat([tensor6, tensor7], dim=1)

```

| Result | t6 | t6 | t6 | t7 | t7 | t7 |
| --- | --- | --- | --- | --- | --- | --- |
| **Row 0** | 1 | 2 | 3 | 7 | 8 | 9 |
| **Row 1** | 4 | 5 | 6 | 10 | 11 | 12 |


---


## 1. Overview of Neural Network Structure

A Neural Network for deep learning is composed of various **layers** that perform data operations to calculate predictions.

* **PyTorch API Hierarchy**:
    * **High-Level**: `torch.nn` (Layers, Loss functions), `torch.optim` (Optimizers).
    * **Low-Level**: PyTorch Engine, Hardware (CPU/GPU/MPS).

The overall training workflow follows a cycle: **Data Definition $\rightarrow$ Model Construction $\rightarrow$ Feed Forward $\rightarrow$ Loss Calculation $\rightarrow$ Optimization**.

---

## 2. Step-by-Step Implementation

### Step 1: Data Definition
Since the basic data type in PyTorch is a Tensor, all data must first be converted into Tensors. For more efficient handling (mini-batching, shuffling), we use `TensorDataset` and `DataLoader`.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. Create Raw Tensors (Reshaping to (N, 1) is common for regression)
x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6, 1)

# 2. Create Dataset and DataLoader
dataset = TensorDataset(x_train, y_train)

# DataLoader handles batch_size and shuffling automatically
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

```

### Step 2: Model Construction (`nn.Module`)

A model is generally defined as a class that inherits from `nn.Module`.

* `__init__`: Define the layers (e.g., `nn.Linear`, `nn.ReLU`, `nn.Sequential`).
* `forward`: Define how data passes through the layers (Feed Forward).

```python
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # Define data flow
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Instantiate the model
model = MyNeuralNetwork()

```

### Step 3: Loss Function & Optimizer

To train the model, we need to calculate the error (Loss) and update the parameters to minimize it.

| Component | Description | Examples |
| --- | --- | --- |
| **Loss Function** | Calculates the difference between prediction and ground truth. | `nn.MSELoss` (Regression), `nn.CrossEntropyLoss` (Classification) |
| **Optimizer** | Updates model parameters (weights/biases). | `torch.optim.SGD`, `ADAM`, `RMSProp` |

```python
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

```

### Step 4: Training Loop

The training loop iterates through the dataset multiple times (epochs). In each iteration, the model makes a prediction, calculates the loss, and backpropagates the error to update weights.

{: .prompt-danger }

> **Crucial Steps in the Loop**:
> 1. `optimizer.zero_grad()`: Reset gradients from the previous step.
> 2. `loss.backward()`: Calculate gradients (Backpropagation).
> 3. `optimizer.step()`: Update parameters using the gradients.
> 
> 

---

## 3. Full Example: Linear Regression

Here is a complete example combining all the steps to learn a simple linear relationship.

```python
import torch
import torch.nn as nn

# 1. Data
x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6, 1)

# 2. Model (Simple Linear Layer: Input 1 -> Output 1)
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 1) # One input, one output
        )
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = MyNeuralNetwork()

# 3. Loss & Optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# 4. Training Loop
nums_epoch = 2000

for epoch in range(nums_epoch + 1):
    # Prediction
    prediction = model(x_train)
    
    # Calculate Loss
    loss = loss_function(prediction, y_train)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"epoch = {epoch}, current loss = {loss.item()}")

# 5. Testing
# Predict for a new input (e.g., 3.0 should yield approx 5.0)
x_test = torch.Tensor([-3.1, 3.0, 1.2]).view(3, 1)
pred = model(x_test)
print(pred)

```



## The implementation of a **Multi-Variable Linear Regression** model using PyTorch.

## 1. Problem Statement

We aim to find the optimal weights ($w_1, w_2, w_3$) and bias ($b$) for the following relationship:

$$y = w_1x_1 + w_2x_2 + w_3x_3 + b$$

* **Ground Truth (Target)**: In this example, the data is generated based on:
    * Weights: $[2, -3, 2]$
    * Bias: $0$
* **Goal**: The model should start with random weights and learn these exact values through training.

---

## 2. Data Preparation

First, we load the raw data from a CSV file using NumPy and convert it into PyTorch Tensors.

### Loading and Slicing Data
The dataset `LEC06_TrainData.csv` contains 4 columns: the first three are inputs ($x$), and the last one is the label ($y$).

```python
import torch
import numpy as np

# 1. Load data from CSV using NumPy
# The file has 4 columns: x1, x2, x3, y
loaded_data = np.loadtxt('./LEC06_TrainData.csv', delimiter=',')

# 2. Slicing
# x_train_np: All rows, first 3 columns (Index 0 to -1)
x_train_np = loaded_data[:, 0:-1]
# y_train_np: All rows, last column (Index -1)
y_train_np = loaded_data[:, [-1]]

print(f"X shape: {x_train_np.shape}") # (N, 3)
print(f"Y shape: {y_train_np.shape}") # (N, 1)

```

### Converting to Tensors

PyTorch requires data to be in Tensor format for training.

```python
# 3. Convert NumPy arrays to PyTorch Tensors
x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)

```

---

## 3. Model Construction

We define a custom model class inheriting from `nn.Module`. Since we have **3 input variables** and **1 output variable**, we use `nn.Linear(3, 1)`.

```python
import torch.nn as nn

class MyLinearRegressionModel(nn.Module):
    def __init__(self, input_nodes):
        super().__init__()
        # Linear Layer: Accepts 'input_nodes' inputs, produces 1 output
        self.linear_stack = nn.Sequential(
            nn.Linear(input_nodes, 1)
        )
        
    def forward(self, x):
        # Feed-forward pass
        prediction = self.linear_stack(x)
        return prediction

# Instantiate the model with 3 input nodes
model = MyLinearRegressionModel(3)

```

{: .prompt-info }

> **Parameter Initialization**: Initially, the model's weights and bias are set to random values. We can verify this by checking `model.parameters()`.

---

## 4. Training Setup

We define the Loss Function and the Optimizer.

* **Loss Function**: `nn.MSELoss()` (Mean Squared Error), which is standard for regression problems.
* **Optimizer**: `torch.optim.SGD` (Stochastic Gradient Descent) with a learning rate (`lr`) of `1e-2`.

```python
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

```

---

## 5. Training Loop

The training process involves repeating the **Feed Forward  Loss Calculation  Backpropagation** cycle for a specified number of epochs (2000 in this example).

```python
nums_epoch = 2000
loss_list = []

for epoch in range(nums_epoch + 1):
    # 1. Prediction (Feed Forward)
    # Note: We pass the input data 'x_train' to the model instance directly
    prediction = model(x_train)
    
    # 2. Calculate Loss
    loss = loss_function(prediction, y_train)
    loss_list.append(loss.item())
    
    # 3. Backpropagation & Optimization
    optimizer.zero_grad()  # Reset gradients
    loss.backward()        # Calculate gradients
    optimizer.step()       # Update weights
    
    # Log progress every 100 epochs
    if epoch % 100 == 0:
        print(f"epoch = {epoch}, current loss = {loss.item()}")

```

**Result:**
The loss starts high (e.g., ~30.4) and rapidly decreases to nearly zero (), indicating successful convergence.

---

## 6. Evaluation & Visualization

### Verifying Learned Parameters

After training, we check if the model learned the correct weights  and bias .

```python
for name, child in model.named_children():
    for param in child.parameters():
        print(name, param)

# Expected Output:
# Weight: tensor([[ 2.0000, -3.0000,  2.0000]])
# Bias: tensor([something very close to 0])

```

### Visualizing Loss Trend

We can plot the loss history to visualize the learning process.

```python
import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(loss_list, label='train loss')
plt.legend(loc='best')
plt.show()

```

### Testing with New Data

Finally, we test the model with unseen data to verify its generalization capability.

```python
# New test data (User defined)
# Formula: y = 2*x1 - 3*x2 + 2*x3
x_test = torch.Tensor([
    [5, 5, 0],   # Expect: 10 - 15 + 0 = -5
    [2, 3, 1],   # Expect: 4 - 9 + 2 = -3
    [-1, 0, -1], # Expect: -2 - 0 - 2 = -4
    [10, 5, 2],  # Expect: 20 - 15 + 4 = 9
    [4, -1, -2]  # Expect: 8 + 3 - 4 = 7
])

# Get predictions
pred = model(x_test)

print("Predictions:\n", pred)

```

--- 

## The implementation of **Logistic Regression** using PyTorch.

* **Dataset**: Kaggle Pima Indians Diabetes Dataset
* **Goal**: Predict whether a patient has diabetes (1) or not (0) based on health metrics.

---

## 1. Data Preparation

We load the dataset from a CSV file. The dataset contains **8 input features** (e.g., glucose, blood pressure) and **1 label** (diabetes outcome).

### Loading & Slicing

```python
import numpy as np
import torch

# 1. Load Data
# The dataset has 9 columns in total (8 inputs + 1 label)
loaded_data = np.loadtxt('./content/diabetes.csv', delimiter=',')

# 2. Slicing
# X (Inputs): All rows, columns 0 to 7 (First 8 columns)
x_train_np = loaded_data[:, 0:-1]
# Y (Labels): All rows, the last column (Index -1)
y_train_np = loaded_data[:, [-1]]

print(f"loaded_data.shape = {loaded_data.shape}")
print(f"x_train_np.shape = {x_train_np.shape}")
print(f"y_train_np.shape = {y_train_np.shape}")

# Output:
# loaded_data.shape = (759, 9)
# x_train_np.shape = (759, 8)
# y_train_np.shape = (759, 1)

```

### Tensor Conversion

Since the model requires PyTorch Tensors, we convert the NumPy arrays.

```python
import torch
from torch import nn

# Convert NumPy arrays to PyTorch Tensors
x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)

```

---

## 2. Model Construction

For binary classification, the model architecture changes slightly from Linear Regression. We must pass the output of the Linear layer through a **Sigmoid** activation function to squash the value between 0 and 1.

### Class Definition

```python
class MyLogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logistic_stack = nn.Sequential(
            # Input: 8 features -> Output: 1 value
            nn.Linear(8, 1), 
            # Sigmoid transforms the output to a probability (0.0 ~ 1.0)
            nn.Sigmoid()     
        )
        
    def forward(self, data):
        prediction = self.logistic_stack(data)
        return prediction

# Instantiate the model
model = MyLogisticRegressionModel()

```

{: .prompt-info }

> **Why Sigmoid?**
> The `nn.Linear` layer produces values ranging from  to . The `nn.Sigmoid()` function maps these values to the range , which represents the **probability** of the class being 1 (True).

---

## 3. Loss Function & Optimizer

### Binary Cross Entropy (BCE)

For binary classification, we use `nn.BCELoss` instead of MSE.

* **Formula**: $ \text{loss} = - \frac{1}{N} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})] $

```python
# Loss Function: Binary Cross Entropy
loss_function = nn.BCELoss()

# Optimizer: Stochastic Gradient Descent (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

```

---

## 4. Training Loop

The training loop includes an additional step to calculate **accuracy**. Since the model outputs a probability (0.0~1.0), we apply a threshold (usually 0.5) to determine the final class (0 or 1).

```python
nums_epoch = 5000
train_loss_list = []
train_accuracy_list = []

for epoch in range(nums_epoch + 1):
    # 1. Feed Forward
    outputs = model(x_train)
    
    # 2. Calculate Loss
    loss = loss_function(outputs, y_train)
    train_loss_list.append(loss.item())
    
    # 3. Calculate Accuracy
    # If output > 0.5, predict 1 (True), else 0 (False)
    prediction = outputs > 0.5 
    
    # Compare prediction with ground truth (y_train)
    correct = (prediction.float() == y_train) 
    accuracy = correct.sum().item() / len(correct)
    train_accuracy_list.append(accuracy)
    
    # 4. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"epoch = {epoch}, loss = {loss.item():.4f}, accuracy = {accuracy:.4f}")

```

---

## 5. Result Visualization

We visualize the training process to ensure the loss is decreasing and accuracy is increasing.

```python
import matplotlib.pyplot as plt

# 1. Loss Trend
plt.subplot(1, 2, 1)
plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(train_loss_list, label='train loss')
plt.legend(loc='best')

# 2. Accuracy Trend
plt.subplot(1, 2, 2)
plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.plot(train_accuracy_list, label='train accuracy')
plt.legend(loc='best')

plt.show()

```

### Analysis

* **Loss**: Rapidly decreases initially and stabilizes around 0.47.
* **Accuracy**: Increases sharply and saturates around 77% (0.77).

---

## 6. Important Considerations

The current implementation is a basic example and has two major limitations that need to be addressed in real-world projects:

### 1. Overfitting Check

* **Current State**: We are calculating loss and accuracy **only on training data**.
* **Risk**: The model might be "memorizing" the training data (Overfitting).
* **Solution**: We must split the data into **Training / Validation / Test** sets. The model should be evaluated on the Validation/Test set to verify its true performance.

### 2. Batch Size

* **Current State**: We are feeding the entire dataset (759 rows) into the model at once (Batch Size = Total Data).
* **Risk**: This is computationally expensive and slow for large datasets. It can also lead to poor convergence.
* **Solution**: Use **Mini-batches** (e.g., 32, 64) to update weights more frequently and efficiently.

---



## Batching
When training Deep Learning models, we rarely feed the entire dataset into the model at once. Instead, we need a pipeline to:
1.  **Store/Load** input features and labels.
2.  **Split** the data into small groups called **Batches**.
3.  **Shuffle** the data to prevent the model from learning the order of data.

PyTorch provides two essential classes to handle this efficiently:
* **`Dataset`**: Defines **how** to get a single data sample (and its label) and the total size of the data.
* **`DataLoader`**: Wraps the `Dataset` to provide **batching**, **shuffling**, and parallel loading.

---

## 1. Defining a Custom Dataset

To create a custom dataset, you must create a class that inherits from `torch.utils.data.Dataset` and implement three "magic methods": `__init__`, `__getitem__`, and `__len__`.

### Code Implementation

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Prepare Raw Data
# Total 6 samples. Reshaped to (6, 1)
x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6, 1)

# 2. Define Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, x_train, y_train):
        """
        Initialize the dataset. 
        Store the data tensors passed as arguments.
        """
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        """
        Return a single sample (x, y) at the given 'index'.
        PyTorch uses this to fetch data one by one to assemble a batch.
        """
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.x_train.shape[0]  # Returns 6

```

---

## 2. Using DataLoader

Once the `Dataset` is defined, we pass it to the `DataLoader`. The `DataLoader` automatically handles the complex logic of batching and shuffling.

### Instantiation

```python
# Create an instance of the CustomDataset
dataset = CustomDataset(x_train, y_train)

# Create the DataLoader
# batch_size=3: The loader will group 3 samples into one batch.
# shuffle=True: The data order will be randomized every epoch.
train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)

```

{: .prompt-tip }

> **Calculation**:
> * Total Data Size: 6
> * Batch Size: 3
> * Total Iterations per Epoch:  batches.
> 
> 

---

## 3. Training Loop Integration

In the training loop, we iterate over the `DataLoader`. In each iteration, the loader returns a **mini-batch** of data (features and labels) instead of a single item.

```python
# Assume model, loss_function, and optimizer are defined elsewhere

# Run for 2 Epochs
for epoch in range(2):
    
    # Iterate through the DataLoader
    # enumerate() gives us the batch index (idx) and the data (batch_data)
    for idx, batch_data in enumerate(train_loader):
        
        # Unpack the batch (x and y)
        # Since batch_size=3, x_train_batch will have shape (3, 1)
        x_train_batch, y_train_batch = batch_data
        
        # --- Standard Training Steps ---
        # 1. Prediction
        output_batch = model(x_train_batch)
        
        # 2. Loss Calculation
        loss = loss_function(output_batch, y_train_batch)
        
        # 3. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Logging ---
        print('=========================================')
        print(f'epoch = {epoch+1}, batch_idx = {idx+1}')
        print(f'Batch Size: x={len(x_train_batch)}, y={len(y_train_batch)}')
        print('=========================================')


```

### Output Flow

Since `shuffle=True`, the specific values in each batch might vary, but the structure will be:

**Epoch 1** (Total 2 Batches)

* **Batch 1**: Contains 3 random samples.
* **Batch 2**: Contains the remaining 3 samples.

**Epoch 2** (Total 2 Batches)

* **Batch 1**: Contains 3 random samples (shuffled differently).
* **Batch 2**: Contains the remaining 3 samples.

```text
# Example Output Log
epoch = 1 , batch_idx = 1 . 3 3 3  <-- 1st Batch (Size 3)
epoch = 1 , batch_idx = 2 . 3 3 3  <-- 2nd Batch (Size 3)
epoch = 2 , batch_idx = 1 . 3 3 3
epoch = 2 , batch_idx = 2 . 3 3 3

```


## Deep Learning Architecture

Unlike machine learning such as Linear or Logistic Regression, a **Deep Learning** architecture consists of three main components:
1.  **Input Layer**: Receives the raw data.
2.  **Hidden Layer(s)**: One or more layers between input and output where feature extraction and calculation happen.
3.  **Output Layer**: Produces the final prediction.

The data flows through these layers (**Feed Forward**), the error is calculated, and weights are updated backwards (**Backpropagation**).

---

## 1. Data Preparation

We use a simple dataset representing **Study Hours (x)** vs. **Pass/Fail (y)**.
* **Rule**: If $x \le 12$, Fail (0). If $x \ge 14$, Pass (1).

```python
import torch

# 1. Training Data
# Input: Study hours
x_train = torch.Tensor([2, 4, 6, 8, 10, 
                        12, 14, 16, 18, 20]).view(10, 1)

# Label: 0 (Fail) or 1 (Pass)
y_train = torch.Tensor([0, 0, 0, 0, 0, 
                        0, 1, 1, 1, 1]).view(10, 1)

print(f"x_train shape: {x_train.shape}") # torch.Size([10, 1])
print(f"y_train shape: {y_train.shape}") # torch.Size([10, 1])

```

---

## 2. Model Architecture (Hidden Layer)

This is the core difference. We use `nn.Sequential` to stack multiple layers.
We map the **1 Input Node** to **8 Hidden Nodes**, and then to **1 Output Node**.

### Code Implementation

```python
import torch.nn as nn

class MyDeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the layer stack
        self.deeplearning_stack = nn.Sequential(
            # Hidden Layer: Accepts 1 input, generates 8 outputs (features)
            nn.Linear(1, 8),
            
            # Output Layer: Accepts 8 inputs (from hidden layer), generates 1 final output
            nn.Linear(8, 1),
            
            # Activation: Sigmoid for binary classification (0~1 probability)
            nn.Sigmoid()
        )
        
    def forward(self, data):
        prediction = self.deeplearning_stack(data)
        return prediction

# Instantiate the model
model = MyDeepLearningModel()

```

### Parameter Inspection

Since we added a hidden layer, the number of parameters (weights/biases) increases significantly compared to simple regression.

* **Input  Hidden**:  weights +  biases.
* **Hidden  Output**:  weights +  bias.

```python
# Check parameters
for name, child in model.named_children():
    for param in child.parameters():
        print(name, param)
        
# Example Output structure:
# Parameter containing: [[...]] (8 values for hidden weights)
# Parameter containing: [...]   (8 values for hidden bias)
# ...

```

---

## 3. Training Loop

The training process is identical to Logistic Regression. We use **Binary Cross Entropy (BCELoss)** and **SGD**.

```python
# 1. Settings
loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

nums_epoch = 5000

# 2. Training Loop
for epoch in range(nums_epoch + 1):
    
    # Feed Forward
    outputs = model(x_train)
    
    # Loss Calculation
    loss = loss_function(outputs, y_train)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"epoch = {epoch}, current loss = {loss.item()}")

# Loss decreases from ~0.72 to ~0.015

```

---

## 4. Evaluation & Inference

When using the model for inference (prediction), it is good practice to switch the model to evaluation mode using `model.eval()`.

### Logical Thresholding

The model outputs a probability (float). We convert this to a binary result (0 or 1).

* If `prediction > 0.5`  **1 (True/Pass)**
* If `prediction <= 0.5`  **0 (False/Fail)**

```python
# Switch to Evaluation Mode
model.eval()

# Test Data: Includes values not in training set (e.g., 0.5, 3.0, 13.0, 31.0)
test_data = torch.Tensor([0.5, 3.0, 3.5, 11.0, 13.0, 31.0]).view(6, 1)

# 1. Get raw probability predictions
pred = model(test_data)

# 2. Convert to Binary Class (0.0 or 1.0)
logical_value = (pred > 0.5).float()

print("Raw Predictions:\n", pred)
print("Logical Classes:\n", logical_value)

```

### Result Analysis

| Input () | Prediction (Prob) | Result (Class) | Note |
| --- | --- | --- | --- |
| **0.5 ~ 11.0** |  | **0** | Correctly predicts Fail () |
| **13.0** |  | **1** | Boundary case. Probability > 50% |
| **31.0** |  | **1** | Correctly predicts Pass () |

```

{: .prompt-info }

> **Why `model.eval()`?**
> Although not strictly necessary for this simple model, `model.eval()` is crucial in complex models using Dropout or Batch Normalization. It tells PyTorch to disable training-specific behaviors during testing.

```

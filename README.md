# Micrograd & ManniGrad

This project is inspired by [Andrej Karpathy's micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0).

## Structure

- **tutorial-follow-along/**: Contains code and notebooks that closely follow Karpathy's original micrograd implementation and explanations.
- **ManniGrad/**: My own take on micrograd, with several enhancements and new features.

## Features in ManniGrad

- **Additional Activation Functions**: Implemented extra activation functions such as ReLU, sigmoid, and more (see `ManniGrad/engine.py` and `ManniGrad/nn.py`).
- **Custom Optimizer Class**: Created an optimizer class that supports both SGD and Adam optimizers, making it easy to switch between them for training.

## Getting Started

1. To explore the original micrograd, see the `tutorial-follow-along` folder.
2. To use the enhanced ManniGrad version, check the `ManniGrad` folder and the example scripts in `example/`.

## ManniGrad: Detailed Usage & Setup

### Installation & Setup

Use pip to install [ManniGrad](https://pypi.org/project/mannigrad)
```
pip install mannigrad
```

### Using ManniGrad

You can use ManniGrad by importing its modules in your own scripts or by running the provided example.

#### Example: Training a Neural Network

```python
from mannigrad.engine import Value
from mannigrad.nn import MLP, SGD, Adam
from mannigrad.utils import mse_loss

# Define a simple MLP: 2 inputs, two hidden layers (4 neurons each), 1 output
mlp = MLP(2, [4, 4, 1], activations=[Value.relu, Value.sigmoid, Value.tanh])

# Training data
X = [[Value(x1), Value(x2)] for x1, x2 in [[0,0],[0,1],[1,0],[1,1]]]
Y = [Value(y) for y in [0,1,1,0]]

# Choose optimizer: SGD or Adam
optimizer = Adam(mlp.parameters(), lr=0.01)  # or SGD(...)

for epoch in range(500):
    y_preds = [mlp(x) for x in X]
    loss = mse_loss(y_preds, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss = {loss.data:.6f}")
```

#### Activation Functions
- `Value.tanh`, `Value.relu`, `Value.sigmoid`, etc. are available for use in your networks.

#### Optimizers
- `SGD`: Standard stochastic gradient descent.
- `Adam`: Adaptive Moment Estimation optimizer.
- Both can be used via the `nn` module and accept any model parameters.

### File Overview
- `ManniGrad/engine.py`: Core autograd engine and Value class (with custom activations).
- `ManniGrad/nn.py`: Neural network layers, MLP, and optimizer classes (SGD, Adam).
- `ManniGrad/utils.py`: Utility functions (e.g., loss functions).
- `example/train_model.py`: Example script demonstrating training with ManniGrad.

### Running the Example
From the project root, run:
```sh
python example/train_model.py
```


**Credit:**
- Original concept and tutorial by [Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Extensions and new features by yours truly.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ManniGrad.engine import Value
from ManniGrad.nn import MLP, SGD
from ManniGrad.utils import mse_loss

# 2 inputs → [4, 4, 1] network (2 hidden layers with 4 neurons each)
mlp = MLP(2, [4, 4, 1], activations=[Value.tanh, Value.tanh, Value.tanh])

# train data
raw_X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
raw_Y = [0, 1, 1, 0]


X = [[Value(x1), Value(x2)] for x1, x2 in raw_X]
Y = [Value(y) for y in raw_Y]

opt = SGD(mlp.parameters(), lr=0.1)

# training loop
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    y_preds = [mlp(x) for x in X]
    loss = mse_loss(y_preds, Y)
    
    # Backward pass
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # Logging
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss = {loss.data:.6f}")

# preds
print("\nPredictions after training:")
for x, y in zip(raw_X, y_preds):
    print(f"Input {x} → Predicted {y.data:.4f}")



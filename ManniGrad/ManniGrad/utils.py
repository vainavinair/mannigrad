def mse_loss(pred, target):
    """Mean Squared Error loss for scalar predictions."""
    return (pred - target) ** 2
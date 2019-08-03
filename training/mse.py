import numpy as np

"""
    Mean Squared Error Calculation files
"""


def mse_loss(valid, pred):
    return ((valid - pred) ** 2).mean()


y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred))

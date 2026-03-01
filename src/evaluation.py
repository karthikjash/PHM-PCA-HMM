import numpy as np

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    Measures average squared prediction error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    Measures average absolute prediction error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred))
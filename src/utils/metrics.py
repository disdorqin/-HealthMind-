import numpy as np

def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate direction accuracy (up/down prediction).
    Accuracy = P(sign(y_t - y_{t-1}) == sign(ŷ_t - y_{t-1}))
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
        
    # We compare change from previous true value
    # For y_true[i], the previous value is y_true[i-1]
    # For y_pred[i], the baseline is also y_true[i-1] (we predict the move)
    
    true_diff = y_true[1:] - y_true[:-1]
    pred_diff = y_pred[1:] - y_true[:-1]
    
    # Handle zero diffs?
    # sign(0) = 0.
    
    match = (np.sign(true_diff) == np.sign(pred_diff))
    return float(np.mean(match))

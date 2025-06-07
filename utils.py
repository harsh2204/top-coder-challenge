import json
import numpy as np
from tinygrad.tensor import Tensor

def load_data(file_path):
    """Load and preprocess data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    X = []
    y = []
    
    for case in data:
        inputs = case['input']
        X.append([
            inputs['trip_duration_days'],
            inputs['miles_traveled'],
            inputs['total_receipts_amount']
        ])
        if 'expected_output' in case:
            y.append(case['expected_output'])
    
    X = np.array(X, dtype=np.float32)
    if y:
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    return X_normalized, y, (X_mean, X_std)

def prepare_batch(X, y=None, batch_size=32):
    """Prepare data batch for training."""
    indices = np.random.permutation(len(X))[:batch_size]
    X_batch = Tensor(X[indices])
    if y is not None:
        y_batch = Tensor(y[indices])
        return X_batch, y_batch
    return X_batch

def calculate_accuracy(predictions, targets, threshold=1.0):
    """Calculate accuracy within threshold."""
    differences = np.abs(predictions - targets)
    return np.mean(differences <= threshold) * 100

def evaluate_model(model, X, y, stats=None):
    """Evaluate model performance."""
    with Tensor.train(False):
        predictions = model(Tensor(X)).numpy()
        
    mae = np.mean(np.abs(predictions - y))
    accuracy_1 = calculate_accuracy(predictions, y, threshold=1.0)
    accuracy_01 = calculate_accuracy(predictions, y, threshold=0.01)
    
    return {
        'mae': mae,
        'accuracy_1': accuracy_1,
        'accuracy_01': accuracy_01,
        'predictions': predictions
    } 

import json
import numpy as np
from model import ReimbursementModel
from utils import evaluate_model
from tinygrad.tensor import Tensor

def load_and_normalize_data(file_path, stats_path):
    # Load normalization stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])
    
    # Load data
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
    
    # Normalize using saved stats
    X_normalized = (X - mean) / std
    
    return X_normalized, y

def main():
    # Load model
    model = ReimbursementModel()
    model.load('best_model.npy')
    
    # Evaluate on public cases
    print("Evaluating on public cases...")
    X_public, y_public = load_and_normalize_data('public_cases.json', 'normalization_stats.json')
    public_metrics = evaluate_model(model, X_public, y_public)
    
    print(f"Public Cases Results:")
    print(f"MAE: {public_metrics['mae']:.4f}")
    print(f"Accuracy (±$1.00): {public_metrics['accuracy_1']:.2f}%")
    print(f"Accuracy (±$0.01): {public_metrics['accuracy_01']:.2f}%")
    
    # Evaluate on private cases
    print("\nEvaluating on private cases...")
    X_private, _ = load_and_normalize_data('private_cases.json', 'normalization_stats.json')
    
    # Generate predictions for private cases
    with Tensor.no_grad():
        private_predictions = model(Tensor(X_private)).numpy()
    
    # Save private case predictions
    with open('private_results.txt', 'w') as f:
        for pred in private_predictions:
            f.write(f"{pred[0]:.2f}\n")
    
    print("Private case predictions saved to private_results.txt")

if __name__ == "__main__":
    main() 

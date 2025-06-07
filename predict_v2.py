import numpy as np
from tinygrad.tensor import Tensor
from model_v2 import ReimbursementModelV2
import json
import sys
import os

def main():
    # Force CPU execution
    os.environ["CPU"] = "1"
    
    if len(sys.argv) != 4:
        print("Usage: python predict_v2.py <trip_duration> <miles> <receipts>")
        sys.exit(1)

    trip_duration = float(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])

    # Load normalization stats
    with open('normalization_stats_v2.json', 'r') as f:
        stats = json.load(f)
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])

    # Load model
    model = ReimbursementModelV2()
    model.load('best_model_v2.npy')

    # Prepare input
    input_data = np.array([trip_duration, miles, receipts], dtype=np.float32).reshape(1, -1)
    input_normalized = (input_data - mean) / std
    input_tensor = Tensor(input_normalized)
    
    # Make prediction
    Tensor.training = False  # Disable training mode
    prediction = model(input_tensor).numpy()[0, 0]

    # Print prediction with additional insights
    print(f"{prediction:.2f}")
if __name__ == "__main__":
    main() 

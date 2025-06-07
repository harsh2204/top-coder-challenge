import numpy as np
from tinygrad.tensor import Tensor
from model import ReimbursementModel
import json
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python predict.py <trip_duration> <miles> <receipts>")
        sys.exit(1)

    trip_duration = int(sys.argv[1])
    miles = int(sys.argv[2])
    receipts = int(sys.argv[3])

    # Load normalization stats
    with open('normalization_stats.json', 'r') as f:
        stats = json.load(f)
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])

    # Load model
    model = ReimbursementModel()
    model.load('best_model.npy')

    # Prepare input
    input_data = np.array([trip_duration, miles, receipts], dtype=np.float32).reshape(1, -1)
    input_normalized = (input_data - mean) / std
    input_tensor = Tensor(input_normalized)

    # Make prediction
    with Tensor.no_grad():
        prediction = model(input_tensor).numpy()[0, 0]

    # Print only the prediction rounded to 2 decimal places
    print(f"{prediction:.2f}")


if __name__ == "__main__":
    main()

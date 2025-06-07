import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from model import ReimbursementModel
from utils import load_data, prepare_batch, evaluate_model
from tqdm import tqdm
import json
import os
from tinygrad.nn.state import get_parameters

def train(model, X_train, y_train, X_val, y_val, epochs=1000, batch_size=32, learning_rate=1e-3):
    optimizer = Adam([layer.weight for layer in [model.l1, model.l2, model.l3, model.l4]] + 
                    [layer.bias for layer in [model.l1, model.l2, model.l3, model.l4]], 
                    lr=learning_rate)
    best_val_mae = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training
        model_outputs = []
        with Tensor.train():
            for _ in range(len(X_train) // batch_size):
                X_batch, y_batch = prepare_batch(X_train, y_train, batch_size)
                
                # Forward pass
                out = model(X_batch)
                loss = ((out - y_batch) ** 2).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                model_outputs.append(loss.numpy())
        
        # Validation
        val_metrics = evaluate_model(model, X_val, y_val)
        
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            model.save('best_model.npy')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"Training Loss: {np.mean(model_outputs):.4f}")
            print(f"Validation MAE: {val_metrics['mae']:.4f}")
            print(f"Validation Accuracy (±$1.00): {val_metrics['accuracy_1']:.2f}%")
            print(f"Validation Accuracy (±$0.01): {val_metrics['accuracy_01']:.2f}%")
            print("-" * 50)
        
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

def main():
    # Load and preprocess data
    X, y, stats = load_data('public_cases.json')
    
    # Split data into train and validation sets
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Initialize and train model
    model = ReimbursementModel()
    train(model, X_train, y_train, X_val, y_val)
    
    # Save normalization stats
    with open('normalization_stats.json', 'w') as f:
        json.dump({
            'mean': stats[0].tolist(),
            'std': stats[1].tolist()
        }, f)
    
    # Final evaluation
    model.load('best_model.npy')
    final_metrics = evaluate_model(model, X_val, y_val)
    print("\nFinal Evaluation Results:")
    print(f"MAE: {final_metrics['mae']:.4f}")
    print(f"Accuracy (±$1.00): {final_metrics['accuracy_1']:.2f}%")
    print(f"Accuracy (±$0.01): {final_metrics['accuracy_01']:.2f}%")

if __name__ == "__main__":
    main() 

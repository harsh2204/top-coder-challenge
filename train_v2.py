import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from model_v2 import ReimbursementModelV2
from utils import load_data, prepare_batch, evaluate_model
from tqdm import tqdm
import json
import os
from tinygrad.nn.state import get_parameters

def train(model, X_train, y_train, X_val, y_val, epochs=1500, batch_size=64, learning_rate=5e-4):
    best_val_mae = float('inf')
    is_continuation = False
    
    # Try to load previous best model if it exists
    if os.path.exists('best_model_v2.npy'):
        try:
            model.load('best_model_v2.npy')
            print("Loaded previous best model, continuing training...")
            # Evaluate to get the best MAE
            val_metrics = evaluate_model(model, X_val, y_val)
            best_val_mae = val_metrics['mae']
            print(f"Previous best MAE: {best_val_mae:.4f}")
            is_continuation = True
        except:
            print("Could not load previous model, starting fresh...")
    
    # Use lower learning rate for continuation training and larger model
    if is_continuation:
        learning_rate = learning_rate * 0.2  # Reduce by 5x for fine-tuning
        print(f"Using reduced learning rate for continuation: {learning_rate}")
    
    # Create optimizer AFTER loading model to ensure it references the correct parameters
    optimizer = Adam(get_parameters(model), lr=learning_rate)
    
    patience = 300  # Increased patience for larger model
    patience_counter = 0
    
    print(f"Training Large Model V2:")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Max epochs: {epochs}")
    print(f"- Patience: {patience}")
    
    for epoch in tqdm(range(epochs), desc="Training V2 Model (Large)"):
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
            model.save('best_model_v2.npy')
            patience_counter = 0
            print(f"\nNew best MAE: {best_val_mae:.4f}")
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:  # Less frequent logging for longer training
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
    np.random.seed(42)  # Same seed as V1 for fair comparison
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Initialize and train model
    model = ReimbursementModelV2()
    train(model, X_train, y_train, X_val, y_val)
    
    # Save normalization stats
    with open('normalization_stats_v2.json', 'w') as f:
        json.dump({
            'mean': stats[0].tolist(),
            'std': stats[1].tolist()
        }, f)
    
    # Final evaluation
    model.load('best_model_v2.npy')
    final_metrics = evaluate_model(model, X_val, y_val)
    
    print("\nFinal Results:")
    print(f"MAE: {final_metrics['mae']:.4f}")
    print(f"Accuracy (±$1.00): {final_metrics['accuracy_1']:.2f}%")
    print(f"Accuracy (±$0.01): {final_metrics['accuracy_01']:.2f}%")

if __name__ == "__main__":
    main() 

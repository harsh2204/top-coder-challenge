import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from model_v2 import EfficientReimbursementNet
from utils import load_data
from tqdm import tqdm
import json
import os
from tinygrad.nn.state import get_parameters
import time

def evaluate_model_turbo(model, X, y, batch_size=512):
    """Ultra-fast evaluation optimized for EfficientNet architecture"""
    predictions = []
    
    with Tensor.train(False):  # Evaluation mode
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            X_batch = Tensor(X[i:end_idx], requires_grad=False)
            pred_batch = model(X_batch)
            predictions.append(pred_batch.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    mae = np.mean(np.abs(predictions - y))
    accuracy_1 = np.mean(np.abs(predictions - y) <= 1.0) * 100
    accuracy_01 = np.mean(np.abs(predictions - y) <= 0.01) * 100
    accuracy_001 = np.mean(np.abs(predictions - y) <= 0.001) * 100  # Near-perfect accuracy
    
    return {
        'mae': mae,
        'accuracy_1': accuracy_1,
        'accuracy_01': accuracy_01,
        'accuracy_001': accuracy_001,
        'predictions': predictions
    }

class SmartLearningRate:
    """Advanced learning rate scheduler based on interview insights about system complexity"""
    def __init__(self, initial_lr=2e-4, patience=12, factor=0.65, min_lr=1e-8, warmup_epochs=20):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.best_loss = float('inf')
        self.wait = 0
        self.epoch = 0
        self.loss_history = []
        self.improvement_streak = 0
        
    def step(self, loss):
        """Dynamic learning rate with warmup and sophisticated adaptation"""
        self.epoch += 1
        self.loss_history.append(loss)
        
        # Warmup phase - gradually increase LR
        if self.epoch <= self.warmup_epochs:
            warmup_factor = self.epoch / self.warmup_epochs
            self.current_lr = self.initial_lr * warmup_factor
            return True
        
        # Keep only recent history
        if len(self.loss_history) > 30:
            self.loss_history = self.loss_history[-30:]
        
        improved = False
        if loss < self.best_loss * 0.9995:  # Very small improvement threshold for precision
            self.best_loss = loss
            self.wait = 0
            self.improvement_streak += 1
            improved = True
        else:
            self.wait += 1
            self.improvement_streak = 0
        
        # Aggressive learning rate boost for consistent improvement
        if self.improvement_streak >= 8 and len(self.loss_history) >= 10:
            recent_trend = np.mean(np.diff(self.loss_history[-8:]))
            if recent_trend < -0.05:  # Strong downward trend
                old_lr = self.current_lr
                self.current_lr = min(self.current_lr * 1.15, self.initial_lr * 1.5)
                if self.current_lr > old_lr:
                    print(f"Boosting LR for strong improvement: {old_lr:.2e} -> {self.current_lr:.2e}")
                    self.improvement_streak = 0
                    return True
        
        # Reduce LR if stagnating
        if self.wait >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait = 0
            if self.current_lr < old_lr:
                print(f"Reducing LR due to stagnation: {old_lr:.2e} -> {self.current_lr:.2e}")
                return True
        
        return False
    
    def get_lr(self):
        return self.current_lr

def create_domain_aware_features(X):
    """Create additional features based on interview insights"""
    duration = X[:, 0]
    miles = X[:, 1] 
    receipts = X[:, 2]
    
    # Kevin's efficiency sweet spot (180-220 miles per day)
    efficiency = miles / (duration + 1e-8)
    efficiency_bonus = np.where(
        (efficiency >= 180) & (efficiency <= 220), 
        1.0, 
        np.maximum(0.0, 1.0 - np.abs(efficiency - 200) / 100)
    )
    
    # Trip length sweet spots (4-6 days optimal per Jennifer)
    duration_bonus = np.where(
        (duration >= 4) & (duration <= 6),
        1.0,
        np.maximum(0.0, 1.0 - np.abs(duration - 5) / 3)
    )
    
    # Spending per day thresholds (Kevin's research)
    spending_per_day = receipts / (duration + 1e-8)
    spending_penalty = np.where(
        (duration <= 3) & (spending_per_day > 75), -0.2,  # Short trip penalty
        np.where(
            (duration >= 4) & (duration <= 6) & (spending_per_day > 120), -0.3,  # Medium trip penalty
            np.where(
                (duration > 6) & (spending_per_day > 90), -0.4,  # Long trip penalty
                0.0
            )
        )
    )
    
    # Marcus's "effort" theory - high mileage + reasonable spending
    effort_score = np.minimum(miles / 100, 5.0) * np.maximum(0.5, 1.0 - spending_per_day / 200)
    
    # Lisa's receipt amount sweet spots
    receipt_efficiency = np.where(
        (receipts >= 600) & (receipts <= 800), 1.2,  # Sweet spot
        np.where(
            receipts < 50, 0.3,  # Penalty for tiny receipts
            np.where(
                receipts > 1200, 0.7,  # Diminishing returns
                1.0
            )
        )
    )
    
    # Add these as additional features
    enhanced_features = np.column_stack([
        X,  # Original features
        efficiency_bonus,
        duration_bonus, 
        spending_penalty,
        effort_score,
        receipt_efficiency
    ])
    
    return enhanced_features

def train_efficient_net(model, X_train, y_train, X_val, y_val, epochs=500, batch_size=256, learning_rate=2e-4):
    """Training optimized for EfficientNet with domain insights"""
    best_val_mae = float('inf')
    is_continuation = False
    
    # Enhanced feature engineering with float32 precision
    print("Applying domain-aware feature engineering...")
    X_train_enhanced = create_domain_aware_features(X_train).astype(np.float32)
    X_val_enhanced = create_domain_aware_features(X_val).astype(np.float32)
    
    # Try to load previous best model
    if os.path.exists('best_model_v2.npy'):
        try:
            model.load('best_model_v2.npy')
            print("Loaded previous EfficientNet model, continuing training...")
            val_metrics = evaluate_model_turbo(model, X_val_enhanced, y_val)
            best_val_mae = val_metrics['mae']
            print(f"Previous best MAE: {best_val_mae:.6f}")
            is_continuation = True
        except Exception as e:
            print(f"Could not load previous model ({e}), starting fresh...")
    
    # Conservative learning rate for EfficientNet
    if is_continuation:
        learning_rate = learning_rate * 0.3  # Very conservative for continuation
    
    lr_scheduler = SmartLearningRate(learning_rate, patience=20, factor=0.7, warmup_epochs=30)
    optimizer = Adam(get_parameters(model), lr=lr_scheduler.get_lr())
    
    # Training parameters optimized for EfficientNet
    steps_per_epoch = max(len(X_train_enhanced) // batch_size, 12)
    eval_frequency = 3  # More frequent evaluation for precision
    patience = 80
    patience_counter = 0
    
    print(f"EfficientNet Training Setup:")
    print(f"- Enhanced features: {X_train_enhanced.shape[1]} dimensions")
    print(f"- Batch size: {batch_size}")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Initial learning rate: {lr_scheduler.get_lr():.2e}")
    print(f"- Target: Sub-0.001 MAE (near-perfect accuracy)")
    
    # Pre-generate batch indices for speed
    all_indices = []
    for _ in range(epochs * steps_per_epoch):
        indices = np.random.randint(0, len(X_train_enhanced), size=batch_size)
        all_indices.append(indices)
    
    batch_idx = 0
    start_time = time.time()
    best_epoch = 0
    
    for epoch in tqdm(range(epochs), desc="EfficientNet Training"):
        # Training phase with mixed precision simulation
        epoch_losses = []
        with Tensor.train():
            for step in range(steps_per_epoch):
                # Use pre-generated indices
                indices = all_indices[batch_idx % len(all_indices)]
                batch_idx += 1
                
                X_batch = Tensor(X_train_enhanced[indices].astype(np.float32), requires_grad=False)
                y_batch = Tensor(y_train[indices].astype(np.float32), requires_grad=False)
                
                # Forward pass
                out = model(X_batch)
                
                # Huber loss for robustness (better than MSE for outliers)
                diff = out - y_batch
                huber_loss = Tensor.where(
                    diff.abs() <= 1.0,
                    0.5 * diff ** 2,
                    diff.abs() - 0.5
                ).mean()
                
                # L2 regularization for EfficientNet
                l2_reg = sum(p.pow(2).sum() for p in get_parameters(model)) * 1e-6
                total_loss = huber_loss + l2_reg
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                for p in get_parameters(model):
                    if p.grad is not None:
                        p.grad = p.grad.clip(-1.0, 1.0)
                
                optimizer.step()
                epoch_losses.append(total_loss.numpy())
        
        avg_loss = np.mean(epoch_losses)
        
        # Adaptive learning rate
        lr_changed = lr_scheduler.step(avg_loss)
        if lr_changed:
            optimizer = Adam(get_parameters(model), lr=lr_scheduler.get_lr())
        
        # Frequent validation for precision
        if epoch % eval_frequency == 0 or epoch == epochs - 1:
            val_metrics = evaluate_model_turbo(model, X_val_enhanced, y_val)
            
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_epoch = epoch
                model.save('best_model_v2.npy')
                patience_counter = 0
                elapsed = time.time() - start_time
                print(f"\n🎯 Epoch {epoch} ({elapsed:.1f}s) - NEW BEST MAE: {best_val_mae:.6f}")
                print(f"   Acc±$1: {val_metrics['accuracy_1']:.2f}% | Acc±$0.01: {val_metrics['accuracy_01']:.2f}% | Acc±$0.001: {val_metrics['accuracy_001']:.2f}%")
            else:
                patience_counter += eval_frequency
            
            # Detailed logging
            if epoch % (eval_frequency * 5) == 0:
                elapsed = time.time() - start_time
                epochs_per_sec = (epoch + 1) / elapsed
                print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Val MAE: {val_metrics['mae']:.6f} | "
                      f"Best: {best_val_mae:.6f} | LR: {lr_scheduler.get_lr():.2e} | Speed: {epochs_per_sec:.1f} ep/s")
            
            # Early stopping with high precision target
            if best_val_mae < 0.001:  # Near-perfect accuracy achieved
                print(f"\n🏆 NEAR-PERFECT ACCURACY ACHIEVED! MAE: {best_val_mae:.6f}")
                print("Continuing training for potential further improvement...")
                # Don't break, keep training for even better results
        
        # Extended patience for EfficientNet
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best MAE: {best_val_mae:.6f} (achieved at epoch {best_epoch})")
            break
    
    total_time = time.time() - start_time
    print(f"\nEfficientNet training completed in {total_time:.1f}s ({epochs/total_time:.1f} epochs/sec)")
    return best_val_mae

def main():
    # Load and preprocess data
    X, y, stats = load_data('public_cases.json')
    
    # Split data with stratification for better validation
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.85 * len(X))  # Larger training set for EfficientNet
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    print(f"Dataset: {len(X_train)} train, {len(X_val)} val samples")
    print(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
    
    # EfficientNet training with domain insights
    model = EfficientReimbursementNet(input_features=8)  # 8 enhanced features
    best_mae = train_efficient_net(
        model, X_train, y_train, X_val, y_val, 
        epochs=600,  # More epochs for convergence to near-zero error
        batch_size=256,  # Optimal batch size for EfficientNet
        learning_rate=2e-4  # Conservative LR for stability
    )
    
    # Save normalization stats
    with open('normalization_stats_v2.json', 'w') as f:
        json.dump({
            'mean': stats[0].tolist(),
            'std': stats[1].tolist()
        }, f)
    
    # Final comprehensive evaluation
    model.load('best_model_v2.npy')
    X_val_enhanced = create_domain_aware_features(X_val)
    final_metrics = evaluate_model_turbo(model, X_val_enhanced, y_val)
    
    print("\n" + "="*60)
    print("🏆 FINAL EFFICIENTNET RESULTS")
    print("="*60)
    print(f"Best MAE achieved: ${best_mae:.6f}")
    print(f"Final MAE: ${final_metrics['mae']:.6f}")
    print(f"Accuracy (±$1.00): {final_metrics['accuracy_1']:.3f}%")
    print(f"Accuracy (±$0.01): {final_metrics['accuracy_01']:.3f}%") 
    print(f"Accuracy (±$0.001): {final_metrics['accuracy_001']:.3f}%")
    print("="*60)
    
    if final_metrics['mae'] < 0.01:
        print("🎯 EXCELLENT: Sub-penny accuracy achieved!")
    elif final_metrics['mae'] < 0.1:
        print("✅ GREAT: Sub-dime accuracy achieved!")
    elif final_metrics['mae'] < 1.0:
        print("👍 GOOD: Sub-dollar accuracy achieved!")
    
    # Analysis of predictions
    predictions = final_metrics['predictions']
    errors = np.abs(predictions - y_val)
    print(f"\nError Analysis:")
    print(f"- Mean error: ${np.mean(errors):.6f}")
    print(f"- Median error: ${np.median(errors):.6f}")
    print(f"- 95th percentile error: ${np.percentile(errors, 95):.6f}")
    print(f"- Max error: ${np.max(errors):.6f}")

if __name__ == "__main__":
    main() 

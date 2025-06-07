import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear

class SqueezeExcitation:
    """Squeeze-and-Excitation block for feature recalibration"""
    def __init__(self, channels, reduction=4):
        self.fc1 = Linear(channels, max(1, channels // reduction))
        self.fc2 = Linear(max(1, channels // reduction), channels)
    
    def __call__(self, x):
        # Global average pooling equivalent for 1D features
        squeeze = x.mean(axis=0, keepdim=True)  # [1, channels]
        excitation = self.fc2(self.fc1(squeeze).relu()).sigmoid()
        return x * excitation

class MBConvBlock:
    """Mobile Inverted Bottleneck Block adapted for tabular data"""
    def __init__(self, in_channels, out_channels, expand_ratio=4, se_ratio=0.25):
        self.expand_ratio = expand_ratio
        self.use_residual = in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand = Linear(in_channels, expanded_channels) if expand_ratio != 1 else None
        
        # Depthwise equivalent - feature mixing
        self.depthwise = Linear(expanded_channels, expanded_channels)
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            self.se = SqueezeExcitation(expanded_channels, int(1/se_ratio))
        else:
            self.se = None
        
        # Projection phase
        self.project = Linear(expanded_channels, out_channels)
        
        # Dropout for regularization
        self.dropout_rate = 0.1
    
    def __call__(self, x):
        identity = x
        
        # Expansion
        if self.expand is not None:
            x = self.expand(x).swish()
        
        # Depthwise (feature mixing)
        x = self.depthwise(x).swish()
        
        # Squeeze-and-Excitation
        if self.se is not None:
            x = self.se(x)
        
        # Projection
        x = self.project(x)
        
        # Residual connection with dropout
        if self.use_residual:
            # Simple dropout simulation
            if Tensor.training:
                dropout_mask = Tensor.uniform(*x.shape) > self.dropout_rate
                x = x * dropout_mask
            x = x + identity
        
        return x

class EfficientReimbursementNet:
    def __init__(self, input_features=8):
        # Accept the enhanced feature set from training script
        self.feature_dim = input_features
        
        # Stem - initial feature processing
        self.stem = Linear(self.feature_dim, 64)
        
        # EfficientNet-inspired blocks with progressive channel scaling
        self.blocks = [
            MBConvBlock(64, 96, expand_ratio=1, se_ratio=0.25),   # Stage 1
            MBConvBlock(96, 96, expand_ratio=6, se_ratio=0.25),   # Stage 1 repeat
            
            MBConvBlock(96, 144, expand_ratio=6, se_ratio=0.25),  # Stage 2
            MBConvBlock(144, 144, expand_ratio=6, se_ratio=0.25), # Stage 2 repeat
            MBConvBlock(144, 144, expand_ratio=6, se_ratio=0.25), # Stage 2 repeat
            
            MBConvBlock(144, 192, expand_ratio=6, se_ratio=0.25), # Stage 3
            MBConvBlock(192, 192, expand_ratio=6, se_ratio=0.25), # Stage 3 repeat
            MBConvBlock(192, 192, expand_ratio=6, se_ratio=0.25), # Stage 3 repeat
            
            MBConvBlock(192, 256, expand_ratio=6, se_ratio=0.25), # Stage 4
            MBConvBlock(256, 256, expand_ratio=6, se_ratio=0.25), # Stage 4 repeat
        ]
        
        # Head - final prediction layers
        self.head_conv = Linear(256, 512)
        self.head_dropout = 0.2
        self.classifier = Linear(512, 1)
    
    def __call__(self, x):
        # Input x should already be enhanced features from training script
        # Ensure float32 precision for Metal compatibility
        if not isinstance(x, Tensor):
            x = Tensor(x.astype(np.float32))
        
        # Stem
        x = self.stem(x).swish()
        
        # EfficientNet blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head_conv(x).swish()
        
        # Dropout simulation for head
        if Tensor.training and self.head_dropout > 0:
            dropout_mask = Tensor.uniform(*x.shape) > self.head_dropout
            x = x * dropout_mask
        
        # Final prediction
        x = self.classifier(x)
        
        return x
    
    def save(self, path):
        """Save all model parameters"""
        params = []
        
        # Stem
        params.extend([self.stem.weight.numpy(), self.stem.bias.numpy()])
        
        # Blocks
        for block in self.blocks:
            if block.expand is not None:
                params.extend([block.expand.weight.numpy(), block.expand.bias.numpy()])
            else:
                params.extend([None, None])  # Placeholder for consistency
            
            params.extend([block.depthwise.weight.numpy(), block.depthwise.bias.numpy()])
            
            if block.se is not None:
                params.extend([
                    block.se.fc1.weight.numpy(), block.se.fc1.bias.numpy(),
                    block.se.fc2.weight.numpy(), block.se.fc2.bias.numpy()
                ])
            else:
                params.extend([None, None, None, None])
            
            params.extend([block.project.weight.numpy(), block.project.bias.numpy()])
        
        # Head
        params.extend([
            self.head_conv.weight.numpy(), self.head_conv.bias.numpy(),
            self.classifier.weight.numpy(), self.classifier.bias.numpy()
        ])
        
        np.save(path, np.array(params, dtype=object), allow_pickle=True)
    
    def load(self, path):
        """Load all model parameters"""
        params = np.load(path, allow_pickle=True)
        i = 0
        
        # Stem
        self.stem.weight.assign(Tensor(params[i].astype(np.float32)))
        self.stem.bias.assign(Tensor(params[i+1].astype(np.float32)))
        i += 2
        
        # Blocks
        for block in self.blocks:
            if block.expand is not None and params[i] is not None:
                block.expand.weight.assign(Tensor(params[i].astype(np.float32)))
                block.expand.bias.assign(Tensor(params[i+1].astype(np.float32)))
            i += 2
            
            block.depthwise.weight.assign(Tensor(params[i].astype(np.float32)))
            block.depthwise.bias.assign(Tensor(params[i+1].astype(np.float32)))
            i += 2
            
            if block.se is not None and params[i] is not None:
                block.se.fc1.weight.assign(Tensor(params[i].astype(np.float32)))
                block.se.fc1.bias.assign(Tensor(params[i+1].astype(np.float32)))
                block.se.fc2.weight.assign(Tensor(params[i+2].astype(np.float32)))
                block.se.fc2.bias.assign(Tensor(params[i+3].astype(np.float32)))
            i += 4
            
            block.project.weight.assign(Tensor(params[i].astype(np.float32)))
            block.project.bias.assign(Tensor(params[i+1].astype(np.float32)))
            i += 2
        
        # Head
        self.head_conv.weight.assign(Tensor(params[i].astype(np.float32)))
        self.head_conv.bias.assign(Tensor(params[i+1].astype(np.float32)))
        self.classifier.weight.assign(Tensor(params[i+2].astype(np.float32)))
        self.classifier.bias.assign(Tensor(params[i+3].astype(np.float32)))

# Alias for backward compatibility
ReimbursementModelV2 = EfficientReimbursementNet

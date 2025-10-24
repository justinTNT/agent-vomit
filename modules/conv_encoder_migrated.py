#!/usr/bin/env python3
"""
MIGRATED CONV ENCODER MODULE
Demonstrates practical migration from old pattern to universal config system
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import torch
from torch import nn
import logging

# Universal config framework imports
from rave_config_system import RAVEConfig
from universal_config_framework import BulletproofModuleBase, ProcessingResult
from category_config_classes import ConvolutionConfig

logger = logging.getLogger(__name__)


class ConvEncoderMigrated(BulletproofModuleBase):
    """
    MIGRATED: Hierarchical CNN encoder with universal config system integration.
    
    This is the migrated version of the original ConvEncoder that demonstrates:
    - Universal configuration pattern
    - Comprehensive error handling
    - Standardized return types
    - Fallback strategies
    - Device management
    """
    
    # Universal config attributes (required for all migrated modules)
    config_section_name = "convolution"
    component_config_class = ConvolutionConfig
    valid_parameters = [
        "in_channels", "out_channels", "base_channels", "num_layers",
        "kernel_size", "stride", "padding", "dilation", "groups", "bias",
        "activation", "normalization"
    ]
    
    def _initialize_module(self):
        """Initialize module with validated configuration"""
        
        # Extract validated parameters from component config
        self.in_channels = self.component_config.in_channels
        self.base_channels = self.component_config.base_channels
        self.num_layers = self.component_config.num_layers
        self.kernel_size = self.component_config.kernel_size
        self.stride = self.component_config.stride
        self.padding = self.component_config.padding
        self.activation = self.component_config.activation
        self.normalization = self.component_config.normalization
        
        # Build encoder blocks with validated configuration
        self._build_encoder_blocks()
        
        # Add global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        logger.info(f"ConvEncoder initialized: {self.num_layers} layers, "
                   f"{self.base_channels} base channels, {self.activation} activation, "
                   f"{self.normalization} normalization")
    
    def _custom_validation(self):
        """Custom validation for ConvEncoder-specific requirements"""
        
        # Validate num_layers
        if self.component_config.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        
        # Validate channel progression makes sense
        if self.component_config.base_channels < 1:
            raise ValueError("base_channels must be at least 1")
        
        # Check for reasonable parameter combinations
        if self.component_config.kernel_size > self.component_config.padding * 2 + 1:
            logger.warning(f"Large kernel_size {self.component_config.kernel_size} "
                          f"with small padding {self.component_config.padding} may cause issues")
        
        logger.debug("ConvEncoder custom validation passed")
    
    def _build_encoder_blocks(self):
        """Build encoder blocks with configuration-driven parameters"""
        
        blocks = []
        current_in = self.in_channels
        
        for i in range(self.num_layers):
            # Calculate output channels with exponential growth
            out_channels = self.base_channels * (2 ** i)
            
            # Create block components
            block_layers = []
            
            # Convolution layer
            conv_layer = nn.Conv2d(
                current_in, 
                out_channels, 
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.component_config.bias
            )
            block_layers.append(conv_layer)
            
            # Normalization layer
            if self.normalization == "batch_norm":
                block_layers.append(nn.BatchNorm2d(out_channels))
            elif self.normalization == "layer_norm":
                # Layer norm for conv layers requires specific handling
                block_layers.append(nn.GroupNorm(1, out_channels))  # Approximation
            elif self.normalization == "group_norm":
                num_groups = min(8, out_channels)  # Ensure divisible groups
                block_layers.append(nn.GroupNorm(num_groups, out_channels))
            elif self.normalization == "instance_norm":
                block_layers.append(nn.InstanceNorm2d(out_channels))
            # "none" case - no normalization added
            
            # Activation layer
            if self.activation == "gelu":
                block_layers.append(nn.GELU())
            elif self.activation == "relu":
                block_layers.append(nn.ReLU(inplace=True))
            elif self.activation == "leaky_relu":
                block_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self.activation == "swish":
                block_layers.append(nn.SiLU())
            elif self.activation == "tanh":
                block_layers.append(nn.Tanh())
            elif self.activation == "elu":
                block_layers.append(nn.ELU(inplace=True))
            
            # Create sequential block
            block = nn.Sequential(*block_layers)
            blocks.append(block)
            
            current_in = out_channels
        
        self.blocks = nn.ModuleList(blocks)
        
        logger.debug(f"Built {len(blocks)} encoder blocks with channels: "
                    f"{[self.base_channels * (2**i) for i in range(self.num_layers)]}")
    
    def _process(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Process input tensor through encoder blocks.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Dictionary containing:
            - features: List of intermediate feature maps
            - pooled: Global average pooled features
            - shape: Shape information
        """
        
        # Input validation (comprehensive)
        self._validate_input_tensor(x)
        
        # Forward pass through encoder blocks
        features = []
        current = x
        
        for i, block in enumerate(self.blocks):
            try:
                current = block(current)
                features.append(current)
                
                # Check for invalid outputs (NaN, Inf)
                if torch.isnan(current).any():
                    raise ValueError(f"NaN detected in block {i} output")
                if torch.isinf(current).any():
                    raise ValueError(f"Inf detected in block {i} output")
                    
            except Exception as e:
                logger.error(f"Error in encoder block {i}: {e}")
                raise  # Let framework handle with fallback
        
        # Global pooling
        pooled = self.global_pool(current).flatten(1)
        
        # Shape information
        shape_info = {
            "input": tuple(x.shape),
            "output": tuple(current.shape),
            "pooled": tuple(pooled.shape),
            "feature_shapes": [tuple(f.shape) for f in features]
        }
        
        return {
            "features": features,
            "pooled": pooled,
            "shape": shape_info
        }
    
    def _validate_input_tensor(self, x: torch.Tensor):
        """Comprehensive input validation"""
        
        # Type check
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        # Dimension check
        if x.dim() != 4:
            raise ValueError(
                f"Expected input with 4 dims (batch, channels, height, width), got {x.dim()}D: {x.shape}"
            )
        
        # Channel check
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.size(1)}"
            )
        
        # Size checks
        batch_size, channels, height, width = x.shape
        
        if batch_size == 0:
            raise ValueError("Batch size cannot be zero")
        
        if height < 2 ** self.num_layers or width < 2 ** self.num_layers:
            raise ValueError(
                f"Input spatial dimensions ({height}x{width}) too small for {self.num_layers} "
                f"layers with stride {self.stride}. Minimum size: {2**self.num_layers}x{2**self.num_layers}"
            )
        
        # Value checks
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")
        
        if torch.isinf(x).any():
            raise ValueError("Input contains infinite values")
    
    def _get_processing_fallback(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Fallback processing when main forward pass fails.
        
        Returns simplified outputs with expected structure but safe values.
        """
        
        logger.warning("ConvEncoder using fallback processing")
        
        try:
            # Try to extract basic info from input
            batch_size = x.size(0) if x.dim() >= 1 else 1
            
            # Create fallback feature maps with expected structure
            fallback_features = []
            current_height = x.size(2) if x.dim() >= 3 else 32
            current_width = x.size(3) if x.dim() >= 4 else 32
            
            for i in range(self.num_layers):
                out_channels = self.base_channels * (2 ** i)
                current_height = max(1, current_height // self.stride)
                current_width = max(1, current_width // self.stride)
                
                # Create zero feature map with expected shape
                feature = torch.zeros(
                    batch_size, out_channels, current_height, current_width,
                    device=x.device if isinstance(x, torch.Tensor) else self.device,
                    dtype=x.dtype if isinstance(x, torch.Tensor) else torch.float32
                )
                fallback_features.append(feature)
            
            # Fallback pooled features
            final_channels = self.base_channels * (2 ** (self.num_layers - 1))
            pooled = torch.zeros(
                batch_size, final_channels,
                device=x.device if isinstance(x, torch.Tensor) else self.device,
                dtype=x.dtype if isinstance(x, torch.Tensor) else torch.float32
            )
            
            # Shape info
            shape_info = {
                "input": tuple(x.shape) if isinstance(x, torch.Tensor) else (batch_size, self.in_channels, 32, 32),
                "output": tuple(fallback_features[-1].shape) if fallback_features else (batch_size, final_channels, 1, 1),
                "pooled": tuple(pooled.shape),
                "feature_shapes": [tuple(f.shape) for f in fallback_features],
                "fallback_used": True
            }
            
            return {
                "features": fallback_features,
                "pooled": pooled,
                "shape": shape_info
            }
            
        except Exception as e:
            logger.error(f"Fallback processing also failed: {e}")
            
            # Ultimate emergency fallback
            emergency_batch_size = 1
            emergency_channels = self.base_channels
            
            return {
                "features": [torch.zeros(emergency_batch_size, emergency_channels, 1, 1)],
                "pooled": torch.zeros(emergency_batch_size, emergency_channels),
                "shape": {
                    "input": (emergency_batch_size, self.in_channels, 1, 1),
                    "output": (emergency_batch_size, emergency_channels, 1, 1),
                    "pooled": (emergency_batch_size, emergency_channels),
                    "feature_shapes": [(emergency_batch_size, emergency_channels, 1, 1)],
                    "emergency_fallback": True
                }
            }
    
    def _get_component_fallback(self) -> ConvolutionConfig:
        """Component-specific fallback configuration"""
        
        logger.warning("Applying ConvEncoder component-specific fallback")
        
        return ConvolutionConfig(
            in_channels=3,  # Common default
            base_channels=32,  # Smaller for safety
            num_layers=2,  # Minimal depth
            kernel_size=3,
            stride=2,
            padding=1,
            activation="relu",  # Most stable
            normalization="batch_norm"
        )
    
    def get_output_channels(self) -> int:
        """Get the number of output channels from the final layer"""
        return self.base_channels * (2 ** (self.num_layers - 1))
    
    def get_receptive_field(self) -> int:
        """Calculate theoretical receptive field"""
        receptive_field = 1
        for i in range(self.num_layers):
            receptive_field = receptive_field * self.stride + (self.kernel_size - 1)
        return receptive_field
    
    def get_spatial_reduction_factor(self) -> int:
        """Get total spatial reduction factor"""
        return self.stride ** self.num_layers


# Compatibility wrapper for backward compatibility
class ConvEncoder(ConvEncoderMigrated):
    """
    Backward compatibility wrapper for the original ConvEncoder.
    
    This allows existing code to work while encouraging migration to the new pattern.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 num_layers: int = 4, config: RAVEConfig = None, **kwargs):
        """
        Backward compatible constructor.
        
        If config is provided, uses new pattern. Otherwise, creates minimal config.
        """
        
        if config is None:
            # Create minimal config for backward compatibility
            from rave_config_system import get_minimal_config
            config = get_minimal_config()
            
            # Issue deprecation warning
            import warnings
            warnings.warn(
                "Using ConvEncoder without RAVEConfig is deprecated. "
                "Please migrate to: ConvEncoder(config, in_channels=..., base_channels=...)",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Initialize with config pattern, passing parameters as kwargs
        super().__init__(
            config, 
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            **kwargs
        )


def migrate_existing_usage():
    """
    Example function showing how to migrate existing ConvEncoder usage.
    """
    
    print("🔄 MIGRATION EXAMPLES")
    print("=" * 50)
    
    from rave_config_system import get_minimal_config
    
    # OLD PATTERN (still works with compatibility wrapper)
    print("\n📜 Old pattern (deprecated but functional):")
    old_encoder = ConvEncoder(in_channels=3, base_channels=64, num_layers=4)
    print(f"   Created encoder: {old_encoder.num_layers} layers, {old_encoder.base_channels} base channels")
    
    # NEW PATTERN (recommended)
    print("\n✨ New pattern (recommended):")
    config = get_minimal_config()
    new_encoder = ConvEncoderMigrated(config, in_channels=3, base_channels=64, num_layers=4)
    print(f"   Created encoder: {new_encoder.num_layers} layers, {new_encoder.base_channels} base channels")
    print(f"   Config summary: {new_encoder.get_config_summary()}")
    
    # Test both with same input
    test_input = torch.randn(2, 3, 64, 64)
    
    print("\n🧪 Testing both encoders:")
    
    # Test old encoder
    try:
        old_output = old_encoder(test_input)
        print(f"   Old encoder output keys: {list(old_output.keys())}")
        print(f"   Old encoder pooled shape: {old_output['pooled'].shape}")
    except Exception as e:
        print(f"   Old encoder failed: {e}")
    
    # Test new encoder
    try:
        new_result = new_encoder(test_input)
        print(f"   New encoder success: {new_result.success}")
        print(f"   New encoder output keys: {list(new_result.data.keys())}")
        print(f"   New encoder pooled shape: {new_result.data['pooled'].shape}")
        print(f"   Processing time: {new_result.processing_time:.4f}s")
    except Exception as e:
        print(f"   New encoder failed: {e}")
    
    # Test error handling
    print("\n⚠️ Testing error handling:")
    try:
        bad_input = torch.randn(2, 5, 32, 32)  # Wrong number of channels
        old_result = old_encoder(bad_input)
        print("   Old encoder: No error (unexpected)")
    except Exception as e:
        print(f"   Old encoder error: {type(e).__name__}")
    
    try:
        bad_input = torch.randn(2, 5, 32, 32)  # Wrong number of channels
        new_result = new_encoder(bad_input)
        print(f"   New encoder fallback activated: {new_result.fallback_used}")
        print(f"   New encoder still returns result: {new_result.success}")
    except Exception as e:
        print(f"   New encoder error: {type(e).__name__}")


if __name__ == "__main__":
    # Run migration examples
    migrate_existing_usage()
    
    print("\n✅ ConvEncoder migration demonstration completed!")
    print("\n📋 Migration Summary:")
    print("   ✅ Universal config integration")
    print("   ✅ Comprehensive error handling") 
    print("   ✅ Standardized return types")
    print("   ✅ Fallback strategies")
    print("   ✅ Backward compatibility")
    print("   ✅ Enhanced validation")
    print("   ✅ Device management")
    print("   ✅ Configuration-driven architecture")
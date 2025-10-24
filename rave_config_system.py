#!/usr/bin/env python3
"""
RAVE CONFIG SYSTEM: Global, over-clockable configuration for maximum experimentation
Strategy: Config-first design with extreme parameter ranges for pushing architectural limits
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple, Literal, Dict, Any
import torch
import math

@dataclass
class AudioConfig:
    """Audio processing configuration - designed for extreme experimentation"""
    
    # Core audio parameters
    sample_rate: int = 44100  # High quality default, can go to 192000
    n_fft: int = 2048         # Large FFT for high resolution, can go to 8192+
    hop_length: int = 512     # 1/4 of n_fft, can be much smaller for extreme overlap
    win_length: Optional[int] = None  # Defaults to n_fft, can be varied independently
    
    # Frequency range (for maximum flexibility)
    f_min: float = 20.0       # Can go as low as 10Hz for sub-bass
    f_max: Optional[float] = None  # Defaults to sample_rate/2, can be limited
    
    # Mel/spectral parameters
    n_mels: int = 128         # High resolution default, can go to 512+
    n_chroma: int = 12        # Standard, can be increased for microtonal
    n_bark_bands: int = 24    # Psychoacoustic, can go to 50+
    
    # Extreme experimentation ranges
    max_n_fft: int = 16384    # For ultra-high resolution
    min_hop_length: int = 64  # For extreme overlap
    max_n_mels: int = 1024    # For ultra-high spectral resolution
    
    # Window and filtering
    window: str = 'hann'      # Can be 'hamming', 'blackman', 'bartlett'
    power: float = 2.0        # Power spectrum, can be 1.0 for magnitude
    normalized: bool = False  # Can normalize for different scales
    
    # Advanced audio processing
    pre_emphasis: float = 0.97        # For speech-like audio
    ref_level_db: float = 20.0        # Reference level
    min_level_db: float = -100.0      # Noise floor
    center: bool = True               # Center frames
    pad_mode: str = 'constant'        # Padding mode
    
    def __post_init__(self):
        """Validate and set derived parameters"""
        if self.win_length is None:
            self.win_length = self.n_fft
        if self.f_max is None:
            self.f_max = self.sample_rate / 2.0
            
        # Validation for extreme ranges
        assert self.n_fft <= self.max_n_fft, f"n_fft {self.n_fft} exceeds max {self.max_n_fft}"
        assert self.hop_length >= self.min_hop_length, f"hop_length {self.hop_length} below min {self.min_hop_length}"
        assert self.n_mels <= self.max_n_mels, f"n_mels {self.n_mels} exceeds max {self.max_n_mels}"

@dataclass
class ModelConfig:
    """Model architecture configuration - designed for over-clocking"""
    
    # Core model dimensions
    d_model: int = 512        # Large default, can go to 2048+ for extreme models
    latent_dim: int = 256     # Can go much higher for rich representations
    hidden_dim: int = 1024    # Can scale to 4096+ for massive models
    
    # Attention and transformer parameters
    n_heads: int = 16         # High default, can go to 32+ for extreme attention
    n_layers: int = 12        # Deep default, can go to 48+ layers
    d_ff: int = 2048          # Feed-forward dimension, can be 8192+ 
    d_k: int = 64             # Key dimension per head
    d_v: int = 64             # Value dimension per head
    
    # Convolution parameters
    base_channels: int = 128  # High channel count default
    max_channels: int = 1024  # Maximum channels for deep layers
    kernel_sizes: List[int] = field(default_factory=lambda: [15, 41, 41, 41, 5])
    strides: List[int] = field(default_factory=lambda: [1, 4, 4, 4, 1])
    dilations: List[int] = field(default_factory=lambda: [1, 1, 2, 4, 1])
    
    # Extreme experimentation ranges
    max_d_model: int = 4096   # For massive models
    max_n_heads: int = 64     # Extreme attention
    max_n_layers: int = 96    # Ultra-deep models
    max_latent_dim: int = 2048 # Rich latent spaces
    
    # Activation and normalization
    activation: str = 'gelu'  # Can be 'relu', 'swish', 'mish'
    norm_type: str = 'layer_norm'  # Can be 'batch_norm', 'group_norm'
    norm_eps: float = 1e-6    # Numerical stability
    
    # Regularization (for extreme models)
    dropout: float = 0.1      # Can go higher for massive models
    attention_dropout: float = 0.1
    path_dropout: float = 0.0  # Stochastic depth
    layer_drop_rate: float = 0.0  # Layer dropping
    
    # Architecture variants
    use_bias: bool = True
    use_causal_mask: bool = False
    use_relative_pos: bool = True
    use_rotary_pos: bool = False
    scale_attention: bool = True
    
    def __post_init__(self):
        """Validate extreme ranges"""
        assert self.d_model <= self.max_d_model, f"d_model {self.d_model} exceeds max {self.max_d_model}"
        assert self.n_heads <= self.max_n_heads, f"n_heads {self.n_heads} exceeds max {self.max_n_heads}"
        assert self.n_layers <= self.max_n_layers, f"n_layers {self.n_layers} exceeds max {self.max_n_layers}"
        assert self.latent_dim <= self.max_latent_dim, f"latent_dim {self.latent_dim} exceeds max {self.max_latent_dim}"
        
        # Ensure head dimensions are compatible
        assert self.d_model % self.n_heads == 0, f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"

@dataclass
class QuantizationConfig:
    """Quantization configuration - designed for extreme codebook experimentation"""
    
    # Core quantization parameters
    num_quantizers: int = 8         # High default, can go to 32+ for extreme quality
    codebook_size: int = 1024       # Can go to 8192+ for massive vocabularies
    codebook_dim: int = 256         # Can match or exceed model dimensions
    
    # Training dynamics
    commitment_cost: float = 0.25   # Can be tuned for different convergence
    ema_decay: float = 0.99         # Exponential moving average
    epsilon: float = 1e-5           # Numerical stability
    
    # Extreme experimentation
    max_num_quantizers: int = 64    # Ultra-high quality quantization
    max_codebook_size: int = 16384  # Massive vocabularies
    max_codebook_dim: int = 2048    # High-dimensional codes
    
    # Advanced quantization strategies
    shared_codebook: bool = False   # Share codebooks across quantizers
    quantizer_dropout: float = 0.0  # Stochastic quantization
    distance_metric: str = 'euclidean'  # Can be 'cosine', 'manhattan'
    init_method: str = 'uniform'    # Can be 'normal', 'xavier'
    
    # Hierarchical quantization
    use_hierarchical: bool = False  # Multi-scale quantization
    hierarchy_levels: int = 3       # Number of hierarchy levels
    
    # Residual quantization
    use_residual: bool = True       # Residual vector quantization
    residual_layers: int = 4        # Number of residual layers
    
    def __post_init__(self):
        """Validate extreme ranges"""
        assert self.num_quantizers <= self.max_num_quantizers
        assert self.codebook_size <= self.max_codebook_size
        assert self.codebook_dim <= self.max_codebook_dim

@dataclass
class ConvolutionConfig:
    """Convolution configuration - designed for architectural experimentation"""
    
    # Basic convolution parameters
    in_channels: int = 1            # Audio input channels
    out_channels: int = 256         # Output channels
    kernel_size: Union[int, Tuple[int, ...]] = 15
    stride: Union[int, Tuple[int, ...]] = 1
    padding: Union[int, Tuple[int, ...]] = 7
    dilation: Union[int, Tuple[int, ...]] = 1
    groups: int = 1
    bias: bool = True
    
    # Advanced convolution types
    conv_type: str = 'causal'       # 'causal', 'anticausal', 'standard'
    use_antialiasing: bool = True   # Anti-aliasing filters
    filter_type: str = 'lanczos'    # Anti-alias filter type
    filter_size: int = 5            # Anti-alias filter size
    
    # Multi-scale and grouped convolutions
    use_multiscale: bool = False    # Multi-scale convolutions
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    use_depthwise: bool = False     # Depthwise separable convolutions
    use_pointwise: bool = False     # Pointwise convolutions
    
    # Extreme architectural variants
    use_octave_conv: bool = False   # Octave convolutions
    use_ghost_conv: bool = False    # Ghost convolutions
    use_inverted_residual: bool = False  # MobileNet-style blocks
    expansion_ratio: float = 4.0    # For inverted residuals
    
    # Normalization and activation in conv blocks
    norm_type: str = 'batch_norm'   # 'batch_norm', 'layer_norm', 'group_norm'
    activation: str = 'relu'        # 'relu', 'gelu', 'swish', 'mish'
    use_spectral_norm: bool = False # Spectral normalization
    power_iterations: int = 1       # For spectral norm

@dataclass
class LossConfig:
    """Loss function configuration - designed for complex training objectives"""
    
    # Basic loss parameters
    reduction: str = 'mean'         # 'mean', 'sum', 'none'
    
    # Multi-scale STFT loss
    stft_scales: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (512, 50, 240),   # (n_fft, hop_length, win_length)
        (1024, 120, 600),
        (2048, 240, 1200),
        (4096, 480, 2400)  # Extreme high resolution
    ])
    stft_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 0.5])
    
    # Spectral loss components
    use_magnitude_loss: bool = True
    use_spectral_convergence: bool = True
    use_log_magnitude: bool = True
    use_phase_loss: bool = False    # Experimental phase loss
    
    # Perceptual losses
    use_mel_loss: bool = True
    use_perceptual_loss: bool = False  # Feature-based perceptual loss
    perceptual_layers: List[str] = field(default_factory=lambda: ['conv1', 'conv3', 'conv5'])
    
    # GAN losses
    gan_loss_type: str = 'hinge'    # 'hinge', 'lsgan', 'wgan'
    discriminator_weight: float = 1.0
    generator_weight: float = 1.0
    feature_matching_weight: float = 10.0
    
    # Regularization losses
    commitment_weight: float = 0.25  # For quantization
    diversity_weight: float = 0.1    # Encourage codebook diversity
    reconstruction_weight: float = 1.0
    
    # Extreme loss experimentation
    use_adversarial_feature_loss: bool = False
    use_consistency_loss: bool = False  # Cross-augmentation consistency
    use_contrastive_loss: bool = False  # Contrastive learning
    temperature: float = 0.1            # For contrastive loss

@dataclass
class TrainingConfig:
    """Training configuration - designed for extreme scale training"""
    
    # Optimization
    learning_rate: float = 1e-4     # Can go much higher for large batches
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'linear', 'exponential'
    warmup_steps: int = 4000
    max_steps: int = 1000000        # Long training for extreme quality
    
    # Batch and sequence parameters
    batch_size: int = 32            # Can scale to 512+ with large GPUs
    max_batch_size: int = 1024      # For gradient accumulation
    sequence_length: int = 65536    # Long sequences for audio
    max_sequence_length: int = 262144  # Extreme long context
    
    # Gradient and training dynamics
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    
    # Advanced training techniques
    use_ema: bool = True            # Exponential moving average
    ema_decay: float = 0.9999
    use_lookahead: bool = False     # Lookahead optimizer
    use_sam: bool = False           # Sharpness-aware minimization
    
    # Extreme scale settings
    distributed_training: bool = False
    num_gpus: int = 1
    use_deepspeed: bool = False     # For massive models
    use_model_parallel: bool = False

@dataclass 
class RAVEConfig:
    """Master RAVE configuration - combines all sub-configs for maximum flexibility"""
    
    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    convolution: ConvolutionConfig = field(default_factory=ConvolutionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Global settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float32
    seed: int = 42
    
    # Experiment metadata
    experiment_name: str = 'rave_extreme'
    version: str = '1.0'
    description: str = 'Over-clockable RAVE configuration'
    
    # Model architecture selection
    encoder_type: str = 'conv'      # 'conv', 'transformer', 'hybrid'
    decoder_type: str = 'conv'      # 'conv', 'transformer', 'hybrid'
    use_vae: bool = True            # VAE vs AE
    use_quantization: bool = True   # Enable quantization
    use_discriminator: bool = True  # Enable GAN training
    
    # Extreme experimentation flags
    enable_extreme_mode: bool = False  # Unlock all limits
    debug_mode: bool = False           # Extra validation
    profile_mode: bool = False         # Performance profiling
    
    def get_overclocked_config(self) -> 'RAVEConfig':
        """Return an over-clocked version for extreme experimentation"""
        config = self.__class__()
        
        # Over-clock audio parameters
        config.audio.n_fft = 4096
        config.audio.hop_length = 256
        config.audio.n_mels = 256
        
        # Over-clock model parameters
        config.model.d_model = 1024
        config.model.latent_dim = 512
        config.model.n_heads = 32
        config.model.n_layers = 24
        config.model.d_ff = 4096
        
        # Over-clock quantization
        config.quantization.num_quantizers = 16
        config.quantization.codebook_size = 2048
        config.quantization.codebook_dim = 512
        
        # Enable extreme mode
        config.enable_extreme_mode = True
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration for consistency"""
        try:
            # Validate dimension compatibility
            assert self.model.d_model % self.model.n_heads == 0
            assert self.quantization.codebook_dim <= self.model.latent_dim * 2  # Reasonable ratio
            
            # Validate audio parameters
            assert self.audio.hop_length <= self.audio.n_fft
            assert self.audio.f_min < self.audio.f_max
            
            return True
        except AssertionError as e:
            if self.debug_mode:
                print(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAVEConfig':
        """Create config from dictionary"""
        # Recursive dataclass creation
        audio_config = AudioConfig(**config_dict.get('audio', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        quantization_config = QuantizationConfig(**config_dict.get('quantization', {}))
        convolution_config = ConvolutionConfig(**config_dict.get('convolution', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        # Remove sub-configs from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['audio', 'model', 'quantization', 'convolution', 'loss', 'training']}
        
        return cls(
            audio=audio_config,
            model=model_config,
            quantization=quantization_config,
            convolution=convolution_config,
            loss=loss_config,
            training=training_config,
            **main_config
        )

# Predefined extreme configurations
def get_minimal_config() -> RAVEConfig:
    """Minimal configuration for testing"""
    config = RAVEConfig()
    config.model.d_model = 128
    config.model.latent_dim = 64
    config.model.n_heads = 4
    config.model.n_layers = 6
    config.quantization.num_quantizers = 4
    config.quantization.codebook_size = 256
    return config

def get_standard_config() -> RAVEConfig:
    """Standard production configuration"""
    return RAVEConfig()

def get_extreme_config() -> RAVEConfig:
    """Extreme over-clocked configuration"""
    return RAVEConfig().get_overclocked_config()

def get_research_config() -> RAVEConfig:
    """Research configuration with experimental features"""
    config = get_extreme_config()
    config.loss.use_phase_loss = True
    config.loss.use_perceptual_loss = True
    config.loss.use_contrastive_loss = True
    config.model.use_rotary_pos = True
    config.convolution.use_octave_conv = True
    config.enable_extreme_mode = True
    return config

if __name__ == "__main__":
    print("🚀 RAVE CONFIG SYSTEM - Over-clockable Architecture Configuration")
    print("=" * 80)
    
    # Test configurations
    configs = {
        'minimal': get_minimal_config(),
        'standard': get_standard_config(), 
        'extreme': get_extreme_config(),
        'research': get_research_config()
    }
    
    for name, config in configs.items():
        print(f"\n🎯 {name.upper()} CONFIG:")
        print(f"   d_model: {config.model.d_model}")
        print(f"   latent_dim: {config.model.latent_dim}")
        print(f"   n_heads: {config.model.n_heads}")
        print(f"   n_layers: {config.model.n_layers}")
        print(f"   quantizers: {config.quantization.num_quantizers}")
        print(f"   codebook_size: {config.quantization.codebook_size}")
        print(f"   n_fft: {config.audio.n_fft}")
        print(f"   Valid: {config.validate()}")
    
    print(f"\n✅ Config system ready for maximum experimentation!")
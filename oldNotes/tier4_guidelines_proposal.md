# Tier 4 Module Guidelines: Advanced/Experimental Components

## Overview

Tier 4 modules represent the most challenging category with specialized domain knowledge, experimental techniques, and ambiguous implementation choices. These guidelines establish a contract between modules and tests to manage complexity and ensure reliable generation.

## Tier 4 Module Analysis

### **PsychoacousticLoss** - Complexity Challenges:
- **Domain expertise**: Requires psychoacoustic modeling knowledge
- **Implementation ambiguity**: Multiple hearing models (Zwicker, Moore-Glasberg, etc.)
- **Computational complexity**: Frequency-dependent masking calculations
- **Research-level**: Limited standardized implementations

### **SelfAttentionGAN** - Complexity Challenges:
- **Architectural uncertainty**: Where to inject self-attention in GAN architectures
- **Multiple variants**: Spectral normalization, different attention mechanisms
- **Training instability**: Attention can destabilize GAN training
- **Integration complexity**: Combining with existing discriminator patterns

### **WaveGANDiscriminator** - Complexity Challenges:
- **Architecture choices**: 1D vs multi-scale approaches
- **Temporal receptive fields**: Balancing local vs global discrimination
- **Training dynamics**: Different from spectrogram-based discriminators
- **Parameter scaling**: Adapting from image GAN patterns

### **ChromaEncoder** - Complexity Challenges:
- **Music theory mapping**: 12-tone equal temperament assumptions
- **Frequency resolution**: Octave equivalence and tuning variations
- **Temporal aggregation**: How to pool chroma across time
- **Harmonic vs percussive**: Handling non-pitched content

### **OnsetDetector** - Complexity Challenges:
- **Algorithm selection**: Spectral flux, phase deviation, complex domain
- **Threshold adaptation**: Dynamic vs fixed thresholds
- **Multi-scale analysis**: Different time-frequency resolutions
- **Ground truth ambiguity**: What constitutes an "onset"

## Tier 4-Specific Guidelines

### 21. **Psychoacoustic Approximation Principle**
```python
# For psychoacoustic models, prioritize tractable approximations over perfect accuracy

# BAD: Implementing full critical band analysis with 240 bands
def critical_bands_full(audio):
    # Complex Bark scale implementation with all bands
    return compute_240_critical_bands(audio)  # Too complex for generation

# GOOD: Simplified critical band approximation
def critical_bands_approx(audio, num_bands=24):
    # Use mel-scale approximation of critical bands
    return mel_filterbank(audio, n_mels=num_bands)  # Tractable approximation

# Psychoacoustic models should be "perceptually plausible" rather than "perceptually perfect"
```

### 22. **Attention Integration Boundaries**
```python
# For attention-based architectures, establish clear integration points

# BAD: Attention everywhere without clear purpose
class AttentionGAN(nn.Module):
    def __init__(self, everywhere_attention=True):
        # Attention in generator, discriminator, loss, optimizer...
        pass

# GOOD: Targeted attention with explicit purpose
class SelfAttentionGAN(nn.Module):
    def __init__(self, attention_layer: int = -2, attention_type: str = 'spatial'):
        # Single attention injection point with clear role
        self.attention_layer = attention_layer  # Which layer gets attention
        self.attention_type = attention_type    # What kind of attention
```

### 23. **Domain-Specific Fallbacks**
```python
# For music/audio analysis modules, provide domain-agnostic fallbacks

# BAD: Assuming perfect pitch detection
def chroma_features(audio):
    f0 = precise_pitch_detection(audio)  # May fail on complex audio
    return compute_chroma(f0)

# GOOD: Robust chroma with fallback
def chroma_features(audio, method='harmonic', fallback='spectral'):
    try:
        if method == 'harmonic':
            return harmonic_chroma(audio)
    except:
        # Fallback to simple spectral method
        return spectral_chroma(audio)
```

### 24. **Research Component Versioning**
```python
# For experimental techniques, make algorithmic choices explicit

# BAD: Hidden implementation choices
class OnsetDetector(nn.Module):
    def __init__(self):
        # Which onset detection algorithm? Parameters?
        pass

# GOOD: Explicit algorithm selection
class OnsetDetector(nn.Module):
    def __init__(self, 
                 method: str = 'spectral_flux',  # Clear default
                 threshold_mode: str = 'adaptive',
                 window_size: int = 2048,
                 **kwargs):
        # All major algorithmic choices are parameters
        self.method = method
        self.threshold_mode = threshold_mode
```

### 25. **Experimental Feature Flags**
```python
# For cutting-edge techniques, provide conservative defaults

# BAD: Aggressive experimental features by default
class PsychoacousticLoss(nn.Module):
    def __init__(self):
        self.use_temporal_masking = True      # Experimental
        self.use_binaural_model = True        # Very experimental
        self.use_individual_hrtf = True       # Research-only

# GOOD: Conservative defaults with opt-in experimental features
class PsychoacousticLoss(nn.Module):
    def __init__(self, 
                 use_temporal_masking: bool = False,
                 use_advanced_features: bool = False,
                 experimental_mode: bool = False,
                 **kwargs):
        # Conservative defaults, explicit opt-in for experimental features
```

## Tier 4 Testing Contract

### **Functional Correctness > Research Accuracy**
```python
# Tests should verify functional behavior, not research-level accuracy

def test_psychoacoustic_loss():
    loss_fn = PsychoacousticLoss()
    
    # DON'T test: "Loss matches published psychoacoustic model to 0.1%"
    # DO test: "Loss produces reasonable gradients and is differentiable"
    
    audio1 = torch.randn(4, 16000)
    audio2 = torch.randn(4, 16000)
    
    loss = loss_fn(audio1, audio2)
    
    assert loss.requires_grad, "Loss should be differentiable"
    assert loss.item() >= 0, "Perceptual loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"
```

### **Integration Over Perfection**
```python
# Tests should verify integration with existing modules

def test_self_attention_gan():
    discriminator = SelfAttentionGAN(
        base_channels=64,
        attention_layer=2
    )
    
    # Test integration with existing training patterns
    real_audio = torch.randn(4, 1, 8192)
    fake_audio = torch.randn(4, 1, 8192)
    
    real_score = discriminator(real_audio)
    fake_score = discriminator(fake_audio)
    
    # Focus on shape compatibility and training stability
    assert real_score.shape == fake_score.shape, "Score shape consistency"
    assert real_score.requires_grad, "Gradients should flow"
```

### **Graceful Degradation Testing**
```python
# Tests should verify behavior with imperfect inputs

def test_onset_detector_robustness():
    detector = OnsetDetector()
    
    # Test with various challenging inputs
    test_cases = [
        torch.zeros(16000),           # Silence
        torch.randn(16000) * 0.001,   # Very quiet noise
        torch.ones(16000),            # DC signal
        torch.randn(100),             # Very short audio
    ]
    
    for audio in test_cases:
        onsets = detector(audio.unsqueeze(0))
        
        # Should not crash and should return reasonable format
        assert isinstance(onsets, torch.Tensor), "Should return tensor"
        assert onsets.dim() >= 1, "Should have at least 1 dimension"
        # Don't require specific onset accuracy for edge cases
```

## Implementation Complexity Budget

### **Tier 4 Complexity Limits**
1. **Maximum 4 major algorithmic choices** per module
2. **Maximum 2 research-level techniques** per module  
3. **Mandatory fallback implementations** for experimental features
4. **Explicit uncertainty quantification** where applicable

### **Required Parameterization**
- **Algorithm selection**: Which variant of the technique
- **Complexity level**: Simple vs advanced implementations
- **Fallback strategy**: What to do when primary method fails
- **Domain assumptions**: Musical vs general audio, etc.

## Success Criteria for Tier 4

### **Module Generation Success**
- ✅ **Compiles and runs** without errors
- ✅ **Produces differentiable outputs** for ML integration
- ✅ **Handles edge cases gracefully** (silence, noise, short audio)
- ✅ **Integrates with existing modules** (shape compatibility)

### **Research Authenticity (Secondary)**
- ⚠️ **Plausible algorithmic approach** (doesn't need to be state-of-art)
- ⚠️ **Reasonable parameter defaults** (documented in literature)
- ⚠️ **Recognized technique variants** (not novel inventions)

### **What We DON'T Require**
- ❌ **Research-level accuracy** (perceptual models need not match published results exactly)
- ❌ **State-of-the-art performance** (implementations should be functional, not optimal)
- ❌ **Complete feature coverage** (simplified versions of complex algorithms are acceptable)

## Proposed Testing Strategy

1. **Generate all 5 Tier 4 modules** using guidelines above
2. **Run functionality tests** (shape, gradients, edge cases)
3. **Integration tests** with existing toolkit modules
4. **Accept 80%+ functional success rate** (don't require research-level validation)
5. **Document implementation choices** and limitations clearly

This contract acknowledges that Tier 4 modules are inherently more experimental while maintaining the practical reliability needed for a usable toolkit.
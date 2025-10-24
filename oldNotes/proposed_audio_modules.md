# Proposed Audio/GAN Modules for Neural Audio Synthesis

Rated by **Significance** (how critical for RAVE/GAN) and **Confidence** (how reliably agents can generate).

## Tier 1: Essential (Significance: 9-10, Confidence: 8-10)

1. **MultiScaleDiscriminator** - Sig: 10, Conf: 9
   - Multiple discriminators at different time scales
   - Critical for GAN audio quality
   - Well-established pattern from HiFi-GAN/UnivNet

2. **PQMFFilterBank** - Sig: 10, Conf: 8
   - Pseudo-Quadrature Mirror Filters for multi-band processing
   - Core to RAVE's efficiency
   - Clear mathematical definition

3. **GANLoss** - Sig: 9, Conf: 10
   - LSGAN, WGAN-GP, hinge loss variants
   - Essential for stable GAN training
   - Simple, well-defined transformations

4. **SpectralNormalization** - Sig: 9, Conf: 9
   - Discriminator stabilization
   - Standard GAN technique
   - Clear implementation pattern

5. **WaveNetResBlock** - Sig: 9, Conf: 9
   - Dilated causal convolutions
   - Foundation of many neural vocoders
   - Well-documented architecture

## Tier 2: Very Useful (Significance: 7-8, Confidence: 7-9)

6. **SubPixelConv** - Sig: 8, Conf: 9
   - Efficient learned upsampling
   - Reduces checkerboard artifacts
   - Simple reshape operation

7. **FiLM** (Feature-wise Linear Modulation) - Sig: 8, Conf: 9
   - Conditional generation via affine transform
   - Key for controllable synthesis
   - Simple: y = γ(c) * x + β(c)

8. **MelSpectrogram** - Sig: 8, Conf: 8
   - Consistent mel-scale spectrograms
   - Standard audio representation
   - LibROSA-compatible implementation

9. **NoiseGenerator** - Sig: 7, Conf: 10
   - Structured/filtered noise injection
   - Important for breathy/noisy sounds
   - Very simple module

10. **AdaIN** (Adaptive Instance Norm) - Sig: 7, Conf: 9
    - Style transfer for audio
    - Useful for timbre control
    - Well-defined operation

11. **PitchEncoder** - Sig: 7, Conf: 7
    - F0 extraction and encoding
    - Useful for pitch control
    - Depends on method (CREPE, YIN, etc.)

## Tier 3: Specialized (Significance: 5-6, Confidence: 6-8)

12. **PerceptualLoss** - Sig: 6, Conf: 8
    - VGG-style loss for audio
    - Improves perceptual quality
    - Adaptation of image technique

13. **LoudnessExtractor** - Sig: 6, Conf: 8
    - A-weighting, ITU-R BS.1770
    - Perceptual loudness features
    - Standard algorithms

14. **GlobalConditioning** - Sig: 6, Conf: 9
    - Speaker/instrument embeddings
    - Simple concatenation/addition
    - Very straightforward

15. **PhaseAwareLoss** - Sig: 6, Conf: 6
    - Phase coherence penalties
    - Improves phase reconstruction
    - More complex mathematics

16. **MFCC** - Sig: 5, Conf: 9
    - Mel-frequency cepstral coefficients
    - Alternative features
    - Standard DSP operation

17. **LocalConditioning** - Sig: 5, Conf: 8
    - Frame-wise control signals
    - For fine-grained control
    - Alignment challenges

## Tier 4: Advanced/Experimental (Significance: 4-5, Confidence: 5-7)

18. **PsychoacousticLoss** - Sig: 5, Conf: 5
    - Frequency masking aware
    - Based on hearing models
    - Complex perceptual models

19. **SelfAttentionGAN** - Sig: 4, Conf: 7
    - Long-range dependencies
    - May be overkill for audio
    - Adds complexity

20. **WaveGANDiscriminator** - Sig: 4, Conf: 8
    - 1D discriminator variant
    - Alternative to spectrogram
    - Simple architecture

21. **ChromaEncoder** - Sig: 4, Conf: 7
    - Pitch class profiles
    - Useful for music
    - Specialized use case

22. **OnsetDetector** - Sig: 4, Conf: 6
    - Transient detection
    - For rhythm-aware synthesis
    - Multiple algorithms

## Summary

**High Priority (implement first):**
- MultiScaleDiscriminator
- PQMFFilterBank  
- GANLoss
- SpectralNormalization
- WaveNetResBlock

These 5 modules would immediately enable RAVE/GAN implementations with the existing VAE and audio modules.

**Total: 22 proposed modules** that maintain the single-responsibility pattern and would complete a professional neural audio synthesis toolkit.
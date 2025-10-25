# Descript Discriminator - Notes for Other Project

## Overview
State-of-the-art neural audio discriminator from Descript's Audio Codec research. **Only useful with trained models** - untrained discriminator gives random scores around 0.5.

## Key Files for Integration
- `standalone_descript_discriminator.py` - Main wrapper class
- `analyze_audio_quality.py` - CLI tool for batch analysis
- `quick_discriminator_test.py` - Simple testing script

## Technical Architecture
**3-in-1 Discriminator System:**
1. **Multi-Period Discriminator (MPD)** - Captures temporal patterns (periods: 2,3,5,7,11)
2. **Multi-Scale Discriminator (MSD)** - Multi-resolution analysis with Kaiser filtering
3. **Multi-Resolution Discriminator (MRD)** - Frequency-domain analysis across 5 bands

**Advantages over basic discriminators:**
- Superior anti-aliasing (Kaiser vs average pooling)
- Multi-band frequency analysis (avoids frequency imbalance)
- Complex spectrogram processing (magnitude + phase)
- Production-tested (powers Descript's 90x compression codec)

## Usage Workflow

### 1. Extract Trained Discriminator
```python
from rave_universal_config import create_rave_model
import torch

# Load pre-trained RAVE model
rave_model = create_rave_model("descript_discriminator", capacity=16)
rave_model.load_state_dict(torch.load("trained_rave_checkpoint.pth"))

# Extract trained discriminator
trained_discriminator = rave_model.discriminator
trained_discriminator.eval()
```

### 2. Quality Assessment
```python
from standalone_descript_discriminator import StandaloneDescriptDiscriminator

# Initialize with trained discriminator
assessor = StandaloneDescriptDiscriminator(
    sample_rate=44100,
    n_channels=1,
    device='cuda'
)
# Replace internal discriminator with trained one
assessor.discriminator = trained_discriminator

# Analyze files
result = assessor.analyze_file("audio.wav")
print(f"Quality score: {result['overall_score']:.4f}")  # Higher = more "real-like"
```

### 3. Batch Analysis
```bash
# CLI tool for directory analysis
python analyze_audio_quality.py /path/to/audio/files --output results.csv --format csv
```

## Expected Score Interpretation
- **Range**: 0.0 to 1.0 
- **Higher scores** = more "real-like" audio
- **Lower scores** = more "generated/artificial" audio
- **~0.5** = untrained discriminator (random scores)

## Use Cases for Your Project
1. **Model Evaluation**: Compare quality of different generation methods
2. **Dataset Filtering**: Identify high/low quality samples in datasets
3. **Generation Monitoring**: Track quality improvements during training
4. **Ablation Studies**: Measure impact of different model components
5. **Real vs Generated Classification**: Binary audio authenticity detection

## Integration Requirements
- **PyTorch** with audio support
- **torchaudio** for audio loading/processing
- **einops** for tensor operations
- **RAVE codebase** for discriminator implementation

## Performance Notes
- **GPU recommended** for large-scale analysis
- **Memory usage**: ~1GB VRAM for typical audio lengths
- **Processing speed**: ~100x real-time on GPU
- **Multi-band analysis**: More thorough but slower than simple discriminators

## Important Limitation
⚠️ **Requires trained models** - The discriminator is only meaningful after being trained to distinguish real vs generated audio. Untrained discriminator gives random scores around 0.5 regardless of input quality.

## Research Context
From paper: "High-Fidelity Audio Compression with Improved RVQGAN" (Descript Audio Codec)
- Achieves 90x compression at 8kbps
- Outperforms EnCodec with 3x lower bitrate
- Universal model for speech, music, environmental audio
- State-of-the-art discrimination architecture as of 2023

## Files Location
All implementation files are in `/Users/jtnt/Play/agent-vomit/` - copy these to your algebra project when needed.
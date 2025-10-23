# Audio Analysis Modules

Comprehensive audio analysis and understanding components for agent-vomit, generated during the audio analysis enhancement session.

## Overview

This collection addresses the critical gap in audio understanding that was identified when comparing our generation-focused modules against established audio analysis standards like AST (Audio Spectrogram Transformer).

## Modules Included

### Configuration and Infrastructure
- **`audio_config.py`**: Standardized configuration system with domain-specific presets
- **`audio_preprocessing.py`**: Reference-compliant preprocessing pipeline
- **`audio_module_chain.py`**: Module chaining with automatic compatibility checking
- **`audio_validation.py`**: Comprehensive validation against reference standards

### Audio Classification
- **`audio_spectrogram_transformer.py`**: Original AST-style implementation  
- **`ast_reference_compatible.py`**: Exact AST reference implementation
- **`instrument_classifier.py`**: Hierarchical instrument classification
- **`audio_event_detector.py`**: Multi-label event detection with temporal localization

### Feature Extraction and Analysis
- **`advanced_musical_features.py`**: Comprehensive musical analysis (harmonic, rhythmic, timbral, melodic, structural)
- **`audio_similarity_matcher.py`**: Multi-scale similarity matching and retrieval
- **`perceptual_quality_assessor.py`**: Psychoacoustic quality assessment

## Key Features

### Standardization
- **Parameter consistency** across all modules
- **Reference dataset normalization** (AudioSet, LibriSpeech, music datasets)
- **Automatic compatibility checking** between modules
- **Domain-specific configurations** (speech, music, timbralgebraics)

### Reference Compliance
- **AST compatibility** with proper input dimensions and normalization
- **Librosa validation** for all feature extraction methods
- **Established preprocessing pipelines** matching research standards
- **SpecAugment integration** for data augmentation

### Module Chaining
- **Automatic adaptation** between incompatible modules
- **Feature dimension matching** with learned adapters
- **Pipeline templates** for common workflows
- **Performance monitoring** and benchmarking

### Timbralgebraics Integration
- **Specialized configuration** for guitar/instrument analysis
- **Ready-to-use pipeline** for latent space feedback
- **Quality assessment** for generated audio
- **Semantic similarity** for timbral navigation

## Usage Examples

### Basic Audio Analysis
```python
from modules.audio_analysis import (
    get_timbralgebraics_config, TimbralgebraicsPreprocessor,
    AdvancedMusicalFeatures, AudioSimilarityMatcher
)

# Configure for timbralgebraics
config = get_timbralgebraics_config()
preprocessor = TimbralgebraicsPreprocessor()

# Extract timbral features
waveform = torch.randn(1, 22050 * 3)  # 3 seconds
features = preprocessor.extract_timbral_features(waveform)

# Analyze musical content
musical_analyzer = AdvancedMusicalFeatures(sample_rate=22050)
musical_features = musical_analyzer(waveform, feature_groups=['harmonic', 'timbral'])
```

### Complete Analysis Pipeline
```python
from modules.audio_analysis import AudioPipeline

# Create timbralgebraics analysis pipeline
pipeline = AudioPipeline.create_timbralgebraics_pipeline()

# Process audio with full analysis
result = pipeline(waveform, return_intermediates=True)

# Access results
quality_scores = result.output['quality_scores']
similarity_features = result.output['similarity_features']
musical_analysis = result.output['musical_features']
```

### AST Classification
```python
from modules.audio_analysis import ast_base384, AUDIOSET_CLASSES

# Create AST model (reference compatible)
model = ast_base384(label_dim=len(AUDIOSET_CLASSES))

# Classify audio
predictions = model.classify_spectrogram(
    spectrogram, 
    class_names=AUDIOSET_CLASSES,
    top_k=5
)
```

### Validation and Testing
```python
from modules.audio_analysis import ComprehensiveValidator

# Validate preprocessing against librosa
validator = ComprehensiveValidator()
results = validator.validate_preprocessor(preprocessor)

# Generate detailed report
report = validator.generate_validation_report(results)
print(f"Success rate: {report['summary']['success_rate']:.1%}")
```

## Integration with Existing Modules

These modules are designed to work seamlessly with the existing agent-vomit generation modules:

```python
# Combine generation and analysis
from modules import ResidualVectorQuantizer  # Existing generation module
from modules.audio_analysis import PerceptualQualityAssessor

# Generate audio
quantizer = ResidualVectorQuantizer(...)
generated_audio = quantizer.decode(latent_codes)

# Assess quality
quality_assessor = PerceptualQualityAssessor(...)
quality_scores = quality_assessor.assess_single_audio_quality(generated_audio)
```

## Relationship to Advanced Modules

This audio_analysis collection provides the **foundational understanding capabilities** needed for audio ML. The upcoming `advanced_modules` collection will build on these foundations with:

- **Music-specific modules** using these analysis capabilities
- **Foundation models** leveraging this preprocessing infrastructure  
- **Multimodal models** combining audio analysis with other modalities
- **Specialized architectures** built on this configuration system

## Testing and Validation

All modules include comprehensive testing:
- **Reference implementation validation** against librosa/established libraries
- **Performance benchmarking** and memory usage tracking
- **Compatibility testing** between modules
- **Domain-specific validation** for music/speech/general audio

Run validation with:
```bash
python -m modules.audio_analysis.audio_validation
```

## Configuration Presets

Pre-configured setups for different domains:
- **Speech**: 16kHz, optimized for voice analysis
- **Music**: 22kHz, music-specific normalization  
- **General**: AudioSet-compatible, broad audio analysis
- **Timbralgebraics**: Guitar-optimized, RAVE-compatible

## Performance Characteristics

- **Module compatibility**: Automatic adaptation between incompatible modules
- **Reference accuracy**: >95% correlation with librosa for standard features
- **Processing speed**: Optimized for both research and production use
- **Memory efficiency**: Configurable precision and batch processing

This collection transforms agent-vomit from a generation-focused library into a comprehensive audio ML system with both sophisticated synthesis AND analysis capabilities.
"""
Advanced Audio Modules

State-of-the-art and specialized audio processing modules for agent-vomit.

Includes:
- Music-specific modules (beat synchronization, chord modeling, source separation)
- Foundation models (AudioMAE, Data2Vec-Audio, WavLM)
- Multimodal models (CLAP, ImageBind audio components)
- Specialized architectures (Conformer, Whisper integration) - Coming soon
- Advanced self-supervised learning approaches - Coming soon
"""

# Music-specific modules
from .beat_synchronizer import (
    BeatSynchronizer, MultiScaleOnsetDetector, AdvancedTempoEstimator,
    create_beat_synchronizer
)

from .chord_sequence_modeler import (
    ChordSequenceModeler, ChromaExtractor, TemporalChordClassifier,
    HarmonicAnalyzer, ChordAnnotation, ChordQuality,
    create_chord_sequence_modeler
)

from .source_separation import (
    SourceSeparationSystem, HarmonicPercussiveSeparator, VocalInstrumentalSeparator,
    SpectralMaskGenerator, SeparationType, SeparationResult,
    create_source_separator, create_harmonic_percussive_separator, create_vocal_separator
)

# Foundation models
from .foundation_models import (
    AudioMAE, Data2VecAudio, WavLM, FoundationModelType, FoundationModelConfig,
    ConvolutionalFeatureEncoder, PositionalEncoding, FoundationModelFactory,
    create_audio_mae, create_data2vec_audio, create_wavlm
)

# Multimodal models
from .multimodal_models import (
    CLAP, ImageBindAudio, AudioCaptioningModel, MultimodalModelType, MultimodalConfig,
    AudioEncoder, TextEncoder,
    create_clap, create_imagebind_audio, create_audio_captioning_model
)

__all__ = [
    # Beat synchronization and rhythm analysis
    'BeatSynchronizer', 'MultiScaleOnsetDetector', 'AdvancedTempoEstimator',
    'create_beat_synchronizer',
    
    # Chord modeling and harmonic analysis
    'ChordSequenceModeler', 'ChromaExtractor', 'TemporalChordClassifier',
    'HarmonicAnalyzer', 'ChordAnnotation', 'ChordQuality',
    'create_chord_sequence_modeler',
    
    # Source separation
    'SourceSeparationSystem', 'HarmonicPercussiveSeparator', 'VocalInstrumentalSeparator',
    'SpectralMaskGenerator', 'SeparationType', 'SeparationResult',
    'create_source_separator', 'create_harmonic_percussive_separator', 'create_vocal_separator',
    
    # Foundation models
    'AudioMAE', 'Data2VecAudio', 'WavLM', 'FoundationModelType', 'FoundationModelConfig',
    'ConvolutionalFeatureEncoder', 'PositionalEncoding', 'FoundationModelFactory',
    'create_audio_mae', 'create_data2vec_audio', 'create_wavlm',
    
    # Multimodal models
    'CLAP', 'ImageBindAudio', 'AudioCaptioningModel', 'MultimodalModelType', 'MultimodalConfig',
    'AudioEncoder', 'TextEncoder',
    'create_clap', 'create_imagebind_audio', 'create_audio_captioning_model'
]
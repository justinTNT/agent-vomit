"""
Audio Analysis Modules

Comprehensive audio analysis and understanding components for agent-vomit.

Includes:
- Audio classification (AST, instrument detection, event detection)
- Advanced musical feature extraction
- Audio similarity and retrieval
- Perceptual quality assessment
- Standardized configuration and preprocessing
- Module chaining and validation frameworks

Generated during audio analysis enhancement session.
"""

# Configuration and preprocessing
from .audio_config import (
    AudioModuleConfig, AudioDomain, NormalizationMethod,
    get_speech_config, get_music_config, get_timbralgebraics_config,
    get_ast_compatible_config
)

from .audio_preprocessing import (
    StandardAudioPreprocessor, ASTCompatiblePreprocessor,
    TimbralgebraicsPreprocessor, MultiScalePreprocessor,
    ReferenceDatasetStats
)

# Classification modules
from .audio_spectrogram_transformer import (
    AudioSpectrogramTransformer, ast_base_384, ast_small_224, ast_tiny_224
)

from .ast_reference_compatible import (
    AudioSpectrogramTransformerReference, ast_base384, ast_base224,
    ast_small224, ast_tiny224
)

from .instrument_classifier import (
    InstrumentClassifier, create_basic_instrument_classifier,
    create_guitar_classifier, INSTRUMENT_FAMILIES, COMMON_INSTRUMENTS
)

from .audio_event_detector import (
    AudioEventDetector, create_esc50_detector, create_music_event_detector,
    create_speech_event_detector, ESC50_CLASSES
)

# Feature extraction
from .advanced_musical_features import (
    AdvancedMusicalFeatures, extract_comprehensive_features
)

# Similarity and quality
from .audio_similarity_matcher import (
    AudioSimilarityMatcher, create_spectral_similarity_matcher,
    create_harmonic_similarity_matcher, create_comprehensive_similarity_matcher
)

from .perceptual_quality_assessor import (
    PerceptualQualityAssessor, create_speech_quality_assessor,
    create_music_quality_assessor, create_general_quality_assessor
)

# Infrastructure
from .audio_module_chain import (
    AudioModuleChain, AudioPipeline, ModuleChainMode,
    chain_modules, create_adaptive_chain
)

from .audio_validation import (
    ComprehensiveValidator, AudioFeatureValidator, 
    PerformanceBenchmarker, ValidationResult, BenchmarkResult
)

__all__ = [
    # Configuration
    'AudioModuleConfig', 'AudioDomain', 'NormalizationMethod',
    'get_speech_config', 'get_music_config', 'get_timbralgebraics_config',
    
    # Preprocessing
    'StandardAudioPreprocessor', 'ASTCompatiblePreprocessor',
    'TimbralgebraicsPreprocessor', 'ReferenceDatasetStats',
    
    # Classification
    'AudioSpectrogramTransformer', 'AudioSpectrogramTransformerReference',
    'InstrumentClassifier', 'AudioEventDetector',
    'ast_base384', 'create_basic_instrument_classifier', 'create_esc50_detector',
    
    # Feature extraction
    'AdvancedMusicalFeatures', 'extract_comprehensive_features',
    
    # Similarity and quality
    'AudioSimilarityMatcher', 'PerceptualQualityAssessor',
    'create_comprehensive_similarity_matcher', 'create_music_quality_assessor',
    
    # Infrastructure
    'AudioModuleChain', 'AudioPipeline', 'ModuleChainMode',
    'ComprehensiveValidator', 'chain_modules'
]
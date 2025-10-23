"""
Audio Module Validation and Testing Framework

Validates implementations against reference standards, benchmarks performance,
and ensures compatibility with established audio processing libraries.
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

from .audio_config import AudioModuleConfig
from .audio_preprocessing import StandardAudioPreprocessor, ReferenceDatasetStats


@dataclass
class ValidationResult:
    """Result from validation testing."""
    test_name: str
    passed: bool
    score: Optional[float] = None
    reference_score: Optional[float] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


@dataclass
class BenchmarkResult:
    """Result from performance benchmarking."""
    module_name: str
    input_shape: Tuple[int, ...]
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    memory_usage: float
    throughput: float  # samples per second


class AudioFeatureValidator:
    """
    Validates audio feature extraction against reference implementations.
    
    Compares our implementations with librosa and other established libraries.
    """
    
    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance
        
    def validate_mel_spectrogram(
        self,
        preprocessor: StandardAudioPreprocessor,
        test_audio: torch.Tensor,
        sample_rate: int
    ) -> ValidationResult:
        """Validate mel-spectrogram extraction against librosa."""
        start_time = time.time()
        
        try:
            # Our implementation
            our_mel = preprocessor.extract_mel_spectrogram(test_audio)
            
            # Librosa reference
            audio_np = test_audio.cpu().numpy()[0]  # Take first sample
            ref_mel = librosa.feature.melspectrogram(
                y=audio_np,
                sr=sample_rate,
                n_fft=preprocessor.config.n_fft,
                hop_length=preprocessor.config.hop_length,
                n_mels=preprocessor.config.n_mels,
                fmin=preprocessor.config.f_min,
                fmax=preprocessor.config.f_max
            )
            ref_mel_db = librosa.power_to_db(ref_mel)
            
            # Compare (account for normalization differences)
            our_mel_np = our_mel[0].cpu().numpy()
            
            # Compute correlation (more robust than absolute difference)
            correlation = np.corrcoef(our_mel_np.flatten(), ref_mel_db.flatten())[0, 1]
            
            # Compute relative error
            rel_error = np.mean(np.abs(our_mel_np - ref_mel_db) / (np.abs(ref_mel_db) + 1e-8))
            
            passed = correlation > 0.95 and rel_error < 0.5  # Allow for normalization differences
            
            return ValidationResult(
                test_name="mel_spectrogram_vs_librosa",
                passed=passed,
                score=correlation,
                reference_score=1.0,
                execution_time=time.time() - start_time,
                details={
                    'correlation': correlation,
                    'relative_error': rel_error,
                    'our_shape': our_mel_np.shape,
                    'ref_shape': ref_mel_db.shape
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="mel_spectrogram_vs_librosa",
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
            
    def validate_mfcc(
        self,
        preprocessor: StandardAudioPreprocessor,
        test_audio: torch.Tensor,
        sample_rate: int
    ) -> ValidationResult:
        """Validate MFCC extraction against librosa."""
        start_time = time.time()
        
        try:
            # Our implementation
            our_mfcc = preprocessor.extract_mfcc(test_audio)
            
            # Librosa reference
            audio_np = test_audio.cpu().numpy()[0]
            ref_mfcc = librosa.feature.mfcc(
                y=audio_np,
                sr=sample_rate,
                n_mfcc=preprocessor.config.n_mfcc,
                n_fft=preprocessor.config.n_fft,
                hop_length=preprocessor.config.hop_length
            )
            
            # Compare
            our_mfcc_np = our_mfcc[0].cpu().numpy()
            correlation = np.corrcoef(our_mfcc_np.flatten(), ref_mfcc.flatten())[0, 1]
            
            passed = correlation > 0.9  # MFCC can have more variation due to DCT differences
            
            return ValidationResult(
                test_name="mfcc_vs_librosa",
                passed=passed,
                score=correlation,
                reference_score=1.0,
                execution_time=time.time() - start_time,
                details={'correlation': correlation}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="mfcc_vs_librosa",
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
            
    def validate_chroma(
        self,
        preprocessor: StandardAudioPreprocessor,
        test_audio: torch.Tensor,
        sample_rate: int
    ) -> ValidationResult:
        """Validate chroma extraction against librosa."""
        start_time = time.time()
        
        try:
            # Our implementation
            our_chroma = preprocessor.extract_chroma(test_audio)
            
            # Librosa reference
            audio_np = test_audio.cpu().numpy()[0]
            ref_chroma = librosa.feature.chroma_stft(
                y=audio_np,
                sr=sample_rate,
                n_fft=preprocessor.config.n_fft,
                hop_length=preprocessor.config.hop_length,
                n_chroma=preprocessor.config.n_chroma
            )
            
            # Compare
            our_chroma_np = our_chroma[0].cpu().numpy()
            correlation = np.corrcoef(our_chroma_np.flatten(), ref_chroma.flatten())[0, 1]
            
            passed = correlation > 0.8  # Chroma can vary due to tuning and normalization
            
            return ValidationResult(
                test_name="chroma_vs_librosa",
                passed=passed,
                score=correlation,
                reference_score=1.0,
                execution_time=time.time() - start_time,
                details={'correlation': correlation}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="chroma_vs_librosa",
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
            
    def validate_spectral_features(
        self,
        preprocessor: StandardAudioPreprocessor,
        test_audio: torch.Tensor,
        sample_rate: int
    ) -> ValidationResult:
        """Validate spectral features against librosa."""
        start_time = time.time()
        
        try:
            # Our implementation
            spectral_features = preprocessor.extract_spectral_features(test_audio)
            
            # Librosa references
            audio_np = test_audio.cpu().numpy()[0]
            stft = librosa.stft(
                audio_np,
                n_fft=preprocessor.config.n_fft,
                hop_length=preprocessor.config.hop_length
            )
            magnitude = np.abs(stft)
            
            ref_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0]
            ref_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)[0]
            ref_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate)[0]
            
            # Compare features
            our_centroid = spectral_features['spectral_centroid'][0].cpu().numpy()
            our_bandwidth = spectral_features['spectral_bandwidth'][0].cpu().numpy()
            our_rolloff = spectral_features['spectral_rolloff'][0].cpu().numpy()
            
            centroid_corr = np.corrcoef(our_centroid, ref_centroid)[0, 1]
            bandwidth_corr = np.corrcoef(our_bandwidth, ref_bandwidth)[0, 1]
            rolloff_corr = np.corrcoef(our_rolloff, ref_rolloff)[0, 1]
            
            avg_correlation = np.mean([centroid_corr, bandwidth_corr, rolloff_corr])
            passed = avg_correlation > 0.8
            
            return ValidationResult(
                test_name="spectral_features_vs_librosa",
                passed=passed,
                score=avg_correlation,
                reference_score=1.0,
                execution_time=time.time() - start_time,
                details={
                    'centroid_correlation': centroid_corr,
                    'bandwidth_correlation': bandwidth_corr,
                    'rolloff_correlation': rolloff_corr
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="spectral_features_vs_librosa",
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )


class ModelPerformanceValidator:
    """
    Validates model performance and accuracy.
    
    Tests models on known datasets and compares against published results.
    """
    
    def __init__(self):
        self.known_results = {
            'ast_audioset': {'map': 0.485, 'accuracy': None},
            'ast_esc50': {'map': None, 'accuracy': 0.956},
            'ast_speechcommands': {'map': None, 'accuracy': 0.981}
        }
        
    def validate_classification_accuracy(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        model_name: str
    ) -> ValidationResult:
        """Validate classification accuracy on test data."""
        start_time = time.time()
        
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(test_data)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output', None))
                else:
                    logits = outputs
                    
                if logits is None:
                    raise ValueError("Could not extract logits from model output")
                    
                # Compute accuracy
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == test_labels).float().mean().item()
                
                # Compare with known results
                reference_acc = None
                if model_name in self.known_results:
                    reference_acc = self.known_results[model_name].get('accuracy')
                    
                # Pass if within reasonable range of reference (or no reference available)
                passed = reference_acc is None or abs(accuracy - reference_acc) < 0.1
                
                return ValidationResult(
                    test_name=f"classification_accuracy_{model_name}",
                    passed=passed,
                    score=accuracy,
                    reference_score=reference_acc,
                    execution_time=time.time() - start_time,
                    details={
                        'num_samples': len(test_data),
                        'num_classes': logits.shape[1],
                        'predictions_shape': predictions.shape
                    }
                )
                
        except Exception as e:
            return ValidationResult(
                test_name=f"classification_accuracy_{model_name}",
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
            
    def validate_feature_consistency(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor
    ) -> ValidationResult:
        """Validate that model produces consistent features."""
        start_time = time.time()
        
        try:
            model.eval()
            
            # Run same input multiple times
            outputs = []
            for _ in range(3):
                with torch.no_grad():
                    output = model(test_data)
                    if isinstance(output, dict):
                        # Extract main feature
                        feature = output.get('features', output.get('embeddings', output.get('logits')))
                    else:
                        feature = output
                    outputs.append(feature)
                    
            # Check consistency
            max_diff = 0.0
            for i in range(1, len(outputs)):
                diff = torch.max(torch.abs(outputs[0] - outputs[i])).item()
                max_diff = max(max_diff, diff)
                
            # Should be exactly the same (deterministic)
            passed = max_diff < 1e-6
            
            return ValidationResult(
                test_name="feature_consistency",
                passed=passed,
                score=1.0 - max_diff,
                reference_score=1.0,
                execution_time=time.time() - start_time,
                details={'max_difference': max_diff}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="feature_consistency",
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )


class PerformanceBenchmarker:
    """
    Benchmarks module performance and resource usage.
    """
    
    def __init__(self, num_warmup: int = 3, num_runs: int = 10):
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        
    def benchmark_module(
        self,
        module: torch.nn.Module,
        input_data: torch.Tensor,
        device: torch.device = torch.device('cpu')
    ) -> BenchmarkResult:
        """Benchmark a single module."""
        module = module.to(device)
        input_data = input_data.to(device)
        module.eval()
        
        # Warmup
        for _ in range(self.num_warmup):
            with torch.no_grad():
                _ = module(input_data)
                
        # Benchmark
        times = []
        torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
        
        for _ in range(self.num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = module(input_data)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            
        # Memory usage
        if device.type == 'cuda':
            memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory_usage = 0.0  # Difficult to measure CPU memory accurately
            
        # Compute statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Throughput (samples per second)
        batch_size = input_data.shape[0] if input_data.ndim > 1 else 1
        throughput = batch_size / mean_time
        
        return BenchmarkResult(
            module_name=type(module).__name__,
            input_shape=tuple(input_data.shape),
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            memory_usage=memory_usage,
            throughput=throughput
        )
        
    def benchmark_pipeline(
        self,
        modules: List[torch.nn.Module],
        input_data: torch.Tensor,
        device: torch.device = torch.device('cpu')
    ) -> List[BenchmarkResult]:
        """Benchmark a pipeline of modules."""
        results = []
        
        current_data = input_data
        for module in modules:
            # Benchmark individual module
            result = self.benchmark_module(module, current_data, device)
            results.append(result)
            
            # Update data for next module (simplified)
            try:
                with torch.no_grad():
                    output = module(current_data.to(device))
                    if isinstance(output, dict):
                        # Extract main tensor
                        for key in ['features', 'embeddings', 'output']:
                            if key in output:
                                current_data = output[key]
                                break
                    else:
                        current_data = output
            except:
                # If we can't process the output, use dummy data
                current_data = torch.randn(input_data.shape[0], 512)
                
        return results


class ComprehensiveValidator:
    """
    Comprehensive validation suite for audio modules.
    
    Combines feature validation, performance validation, and benchmarking.
    """
    
    def __init__(self):
        self.feature_validator = AudioFeatureValidator()
        self.performance_validator = ModelPerformanceValidator()
        self.benchmarker = PerformanceBenchmarker()
        
    def validate_preprocessor(
        self,
        preprocessor: StandardAudioPreprocessor,
        sample_rate: int = 22050,
        duration: float = 3.0
    ) -> List[ValidationResult]:
        """Validate a preprocessing module."""
        # Generate test audio
        test_audio = torch.randn(2, int(sample_rate * duration))
        
        results = []
        
        # Validate against librosa
        results.append(self.feature_validator.validate_mel_spectrogram(
            preprocessor, test_audio, sample_rate
        ))
        results.append(self.feature_validator.validate_mfcc(
            preprocessor, test_audio, sample_rate
        ))
        results.append(self.feature_validator.validate_chroma(
            preprocessor, test_audio, sample_rate
        ))
        results.append(self.feature_validator.validate_spectral_features(
            preprocessor, test_audio, sample_rate
        ))
        
        return results
        
    def validate_classifier(
        self,
        classifier: torch.nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...] = (10, 22050),
        num_classes: int = 10
    ) -> List[ValidationResult]:
        """Validate a classification module."""
        results = []
        
        # Generate test data
        test_data = torch.randn(*input_shape)
        test_labels = torch.randint(0, num_classes, (input_shape[0],))
        
        # Validate accuracy (with dummy data)
        results.append(self.performance_validator.validate_classification_accuracy(
            classifier, test_data, test_labels, model_name
        ))
        
        # Validate consistency
        results.append(self.performance_validator.validate_feature_consistency(
            classifier, test_data
        ))
        
        return results
        
    def benchmark_module(
        self,
        module: torch.nn.Module,
        input_shape: Tuple[int, ...] = (4, 22050),
        device: torch.device = torch.device('cpu')
    ) -> BenchmarkResult:
        """Benchmark a single module."""
        test_data = torch.randn(*input_shape)
        return self.benchmarker.benchmark_module(module, test_data, device)
        
    def generate_validation_report(
        self,
        results: List[ValidationResult],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'summary': {
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results if r.passed),
                'failed_tests': sum(1 for r in results if not r.passed),
                'success_rate': sum(1 for r in results if r.passed) / len(results) if results else 0
            },
            'detailed_results': []
        }
        
        for result in results:
            report['detailed_results'].append({
                'test_name': result.test_name,
                'passed': result.passed,
                'score': result.score,
                'reference_score': result.reference_score,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'details': result.details
            })
            
        # Save report if requested
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        return report


# Utility functions for common validation tasks
def validate_all_preprocessors(
    config_variants: List[AudioModuleConfig]
) -> Dict[str, List[ValidationResult]]:
    """Validate preprocessing across different configurations."""
    validator = ComprehensiveValidator()
    all_results = {}
    
    for i, config in enumerate(config_variants):
        preprocessor = StandardAudioPreprocessor(config)
        results = validator.validate_preprocessor(preprocessor, config.sample_rate)
        all_results[f'config_{i}_{config.domain.value}'] = results
        
    return all_results


def quick_validation_check(module: torch.nn.Module, input_shape: Tuple[int, ...]) -> bool:
    """Quick validation check for a module."""
    try:
        test_input = torch.randn(*input_shape)
        with torch.no_grad():
            output = module(test_input)
        return True
    except Exception as e:
        warnings.warn(f"Quick validation failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Test comprehensive validation
    from .audio_config import get_music_config
    from .audio_preprocessing import StandardAudioPreprocessor
    
    # Create validator
    validator = ComprehensiveValidator()
    
    # Test preprocessing validation
    config = get_music_config()
    preprocessor = StandardAudioPreprocessor(config)
    
    print("Validating preprocessor...")
    results = validator.validate_preprocessor(preprocessor)
    
    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"{status} {result.test_name}: {result.score:.3f} (ref: {result.reference_score})")
        
    # Generate report
    report = validator.generate_validation_report(results)
    print(f"\nValidation Summary:")
    print(f"  Success rate: {report['summary']['success_rate']:.1%}")
    print(f"  Passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
    
    # Quick benchmark
    print(f"\nBenchmarking preprocessor...")
    benchmark = validator.benchmark_module(preprocessor, (2, 22050 * 3))
    print(f"  Mean time: {benchmark.mean_time*1000:.1f}ms")
    print(f"  Throughput: {benchmark.throughput:.1f} samples/sec")
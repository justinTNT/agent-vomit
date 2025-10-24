#!/usr/bin/env python3
"""
Guitar Texture Exploration Tools

Advanced tools for exploring and manipulating the structured guitar texture space
created by the superior architecture. Provides compositional control, intelligent
interpolation, and semantic texture mapping.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .guitar_texture_architecture import (
    GuitarTextureModel, GuitarTextureConfig, TextureLevel,
    spherical_interpolation
)


class InterpolationStrategy(Enum):
    """Different strategies for texture interpolation"""
    LINEAR = "linear"
    SPHERICAL = "spherical"
    COMPOSITIONAL = "compositional"
    MUSICAL = "musical"


class TextureDimension(Enum):
    """Semantic texture dimensions for guitar sounds"""
    BRIGHTNESS = "brightness"
    WARMTH = "warmth"
    SATURATION = "saturation"
    ATTACK = "attack"
    SUSTAIN = "sustain"
    REVERB = "reverb"
    COMPRESSION = "compression"
    FRET_NOISE = "fret_noise"


@dataclass
class TextureAnalysis:
    """Analysis of guitar texture characteristics"""
    brightness: float
    warmth: float
    saturation: float
    attack_sharpness: float
    sustain_length: float
    reverb_amount: float
    compression_ratio: float
    fret_noise_level: float
    spectral_centroid: float
    harmonic_ratio: float
    confidence: float


@dataclass
class InterpolationConfig:
    """Configuration for texture interpolation"""
    strategy: InterpolationStrategy = InterpolationStrategy.SPHERICAL
    num_steps: int = 10
    preserve_content_from: str = 'source'  # 'source', 'target', or 'both'
    hierarchical_weights: Dict[str, float] = None
    smooth_transitions: bool = True
    
    def __post_init__(self):
        if self.hierarchical_weights is None:
            self.hierarchical_weights = {
                TextureLevel.COARSE.value: 1.0,
                TextureLevel.FINE.value: 1.0,
                TextureLevel.DETAIL.value: 1.0
            }


class GuitarTextureExplorer:
    """
    Advanced guitar texture exploration with compositional control
    
    Provides high-level interface for:
    1. Intelligent texture interpolation
    2. Compositional texture control
    3. Semantic texture mapping
    4. Texture space exploration
    """
    
    def __init__(self, model: GuitarTextureModel, config: GuitarTextureConfig):
        self.model = model
        self.config = config
        self.model.eval()
        
        # Texture analysis components
        self.texture_analyzer = TextureAnalyzer()
        self.semantic_mapper = SemanticTextureMapper(model)
        
        # Texture memory for exploration
        self.texture_memory = TextureMemory()
        
    def explore_interpolation_space(self, 
                                  source_audio: torch.Tensor,
                                  target_audio: torch.Tensor,
                                  config: InterpolationConfig) -> Dict[str, Any]:
        """
        Explore the interpolation space between two guitar textures
        
        Args:
            source_audio: [batch, 1, time] source guitar audio
            target_audio: [batch, 1, time] target guitar audio  
            config: Interpolation configuration
            
        Returns:
            Dict with interpolated audio, analysis, and metadata
        """
        
        with torch.no_grad():
            # Extract representations
            source_output = self.model(source_audio, quantize_texture=True)
            target_output = self.model(target_audio, quantize_texture=True)
            
            # Analyze source and target textures
            source_analysis = self.texture_analyzer.analyze(source_audio)
            target_analysis = self.texture_analyzer.analyze(target_audio)
            
            # Perform interpolation based on strategy
            if config.strategy == InterpolationStrategy.COMPOSITIONAL:
                interpolation_result = self._compositional_interpolation(
                    source_output, target_output, config
                )
            elif config.strategy == InterpolationStrategy.MUSICAL:
                interpolation_result = self._musical_interpolation(
                    source_output, target_output, config
                )
            else:
                interpolation_result = self._standard_interpolation(
                    source_output, target_output, config
                )
            
            # Analyze interpolation progression
            progression_analysis = self._analyze_interpolation_progression(
                interpolation_result['interpolated_audio'],
                source_analysis,
                target_analysis
            )
            
            return {
                'interpolated_audio': interpolation_result['interpolated_audio'],
                'interpolation_weights': interpolation_result['weights'],
                'source_analysis': source_analysis,
                'target_analysis': target_analysis,
                'progression_analysis': progression_analysis,
                'metadata': {
                    'strategy': config.strategy.value,
                    'num_steps': config.num_steps,
                    'hierarchical_weights': config.hierarchical_weights
                }
            }
    
    def _compositional_interpolation(self,
                                   source_output: Dict[str, torch.Tensor],
                                   target_output: Dict[str, torch.Tensor], 
                                   config: InterpolationConfig) -> Dict[str, Any]:
        """
        Compositional interpolation using hierarchical texture codes
        
        Allows independent control over different texture levels:
        - Keep coarse texture from source, fine texture from target
        - Blend specific hierarchical levels with different weights
        """
        
        source_indices = source_output['level_indices']
        target_indices = target_output['level_indices']
        
        interpolated_audio = []
        weights = []
        
        for step in range(config.num_steps):
            alpha = step / (config.num_steps - 1)
            
            # Create hybrid texture codes
            hybrid_indices = {}
            step_weights = {}
            
            for level in [TextureLevel.COARSE, TextureLevel.FINE, TextureLevel.DETAIL]:
                level_name = level.value
                level_weight = config.hierarchical_weights[level_name]
                effective_alpha = alpha * level_weight
                
                # Decide whether to use source or target codes
                if effective_alpha < 0.5:
                    hybrid_indices[level_name] = source_indices[level_name]
                    step_weights[level_name] = 'source'
                else:
                    hybrid_indices[level_name] = target_indices[level_name]
                    step_weights[level_name] = 'target'
            
            # Reconstruct from hybrid codes
            # Convert indices back to full quantizer format
            indices_tensor = torch.stack([
                hybrid_indices[TextureLevel.COARSE.value],
                hybrid_indices[TextureLevel.FINE.value], 
                hybrid_indices[TextureLevel.DETAIL.value]
            ], dim=-1).unsqueeze(1)  # [batch, 1, num_levels]
            
            # Decode texture from hybrid indices
            hybrid_texture = self.model.quantizer.decode_from_indices(indices_tensor)
            
            # Use content from source or target based on config
            if config.preserve_content_from == 'source':
                content_z = source_output['content_z']
            elif config.preserve_content_from == 'target':
                content_z = target_output['content_z']
            else:  # 'both' - interpolate content too
                content_z = spherical_interpolation(
                    source_output['content_z'], 
                    target_output['content_z'], 
                    alpha
                )
            
            # Reconstruct audio
            hybrid_audio = self.model.vae.decoder(hybrid_texture, content_z)
            interpolated_audio.append(hybrid_audio)
            weights.append(step_weights)
        
        return {
            'interpolated_audio': interpolated_audio,
            'weights': weights
        }
    
    def _musical_interpolation(self,
                             source_output: Dict[str, torch.Tensor],
                             target_output: Dict[str, torch.Tensor],
                             config: InterpolationConfig) -> Dict[str, Any]:
        """
        Musically-informed interpolation that preserves musical coherence
        
        Uses texture analysis to ensure interpolation maintains musical qualities
        """
        
        interpolated_audio = []
        weights = []
        
        for step in range(config.num_steps):
            alpha = step / (config.num_steps - 1)
            
            # Smooth alpha curve for more musical transitions
            if config.smooth_transitions:
                # S-curve for smoother transitions
                alpha = 3 * alpha**2 - 2 * alpha**3
            
            # Interpolate in texture space with musical constraints
            source_texture = source_output['texture_z']
            target_texture = target_output['texture_z']
            
            # Use spherical interpolation for better texture blending
            interpolated_texture = spherical_interpolation(
                source_texture, target_texture, alpha
            )
            
            # Content interpolation strategy
            if config.preserve_content_from == 'source':
                content_z = source_output['content_z']
            elif config.preserve_content_from == 'target':
                content_z = target_output['content_z']
            else:
                # Musical content interpolation (preserve rhythm, blend harmonics)
                content_z = self._musical_content_interpolation(
                    source_output['content_z'],
                    target_output['content_z'],
                    alpha
                )
            
            # Reconstruct
            interpolated_audio_step = self.model.vae.decoder(interpolated_texture, content_z)
            interpolated_audio.append(interpolated_audio_step)
            weights.append(alpha)
        
        return {
            'interpolated_audio': interpolated_audio,
            'weights': weights
        }
    
    def _standard_interpolation(self,
                              source_output: Dict[str, torch.Tensor],
                              target_output: Dict[str, torch.Tensor],
                              config: InterpolationConfig) -> Dict[str, Any]:
        """Standard linear or spherical interpolation"""
        
        interpolated_audio = []
        weights = []
        
        for step in range(config.num_steps):
            alpha = step / (config.num_steps - 1)
            weights.append(alpha)
            
            if config.strategy == InterpolationStrategy.SPHERICAL:
                interpolated_texture = spherical_interpolation(
                    source_output['texture_z'], 
                    target_output['texture_z'], 
                    alpha
                )
            else:  # LINEAR
                interpolated_texture = (
                    (1 - alpha) * source_output['texture_z'] + 
                    alpha * target_output['texture_z']
                )
            
            # Content handling
            if config.preserve_content_from == 'source':
                content_z = source_output['content_z']
            elif config.preserve_content_from == 'target':
                content_z = target_output['content_z']
            else:
                content_z = (
                    (1 - alpha) * source_output['content_z'] + 
                    alpha * target_output['content_z']
                )
            
            interpolated_audio_step = self.model.vae.decoder(interpolated_texture, content_z)
            interpolated_audio.append(interpolated_audio_step)
        
        return {
            'interpolated_audio': interpolated_audio,
            'weights': weights
        }
    
    def _musical_content_interpolation(self,
                                     source_content: torch.Tensor,
                                     target_content: torch.Tensor,
                                     alpha: float) -> torch.Tensor:
        """
        Musically-informed content interpolation
        
        Preserves rhythmic elements while blending harmonic content
        """
        
        # For now, use spherical interpolation
        # Could be enhanced with musical analysis
        return spherical_interpolation(source_content, target_content, alpha)
    
    def _analyze_interpolation_progression(self,
                                         interpolated_audio: List[torch.Tensor],
                                         source_analysis: TextureAnalysis,
                                         target_analysis: TextureAnalysis) -> List[TextureAnalysis]:
        """Analyze texture characteristics throughout interpolation"""
        
        progression = []
        for audio in interpolated_audio:
            analysis = self.texture_analyzer.analyze(audio)
            progression.append(analysis)
        
        return progression
    
    def explore_texture_space(self, 
                            center_audio: torch.Tensor,
                            exploration_radius: float = 0.5,
                            num_samples: int = 8) -> Dict[str, Any]:
        """
        Explore texture space around a center point
        
        Args:
            center_audio: Center point for exploration
            exploration_radius: How far to explore from center
            num_samples: Number of exploration samples
            
        Returns:
            Dict with explored audio samples and analysis
        """
        
        with torch.no_grad():
            # Extract center representation
            center_output = self.model(center_audio, quantize_texture=True)
            center_texture = center_output['texture_z']
            center_content = center_output['content_z']
            
            # Generate exploration directions in texture space
            batch_size, texture_dim = center_texture.shape
            
            explored_audio = []
            exploration_vectors = []
            
            for i in range(num_samples):
                # Random direction in texture space
                direction = F.normalize(torch.randn_like(center_texture), dim=-1)
                exploration_vectors.append(direction)
                
                # Move in that direction
                explored_texture = center_texture + exploration_radius * direction
                
                # Reconstruct with original content
                explored_audio_sample = self.model.vae.decoder(explored_texture, center_content)
                explored_audio.append(explored_audio_sample)
            
            # Analyze exploration results
            exploration_analysis = []
            for audio in explored_audio:
                analysis = self.texture_analyzer.analyze(audio)
                exploration_analysis.append(analysis)
            
            return {
                'center_audio': center_audio,
                'explored_audio': explored_audio,
                'exploration_vectors': exploration_vectors,
                'exploration_analysis': exploration_analysis,
                'exploration_radius': exploration_radius
            }
    
    def create_texture_morphing_sequence(self,
                                       audio_samples: List[torch.Tensor],
                                       transition_length: int = 5) -> torch.Tensor:
        """
        Create smooth morphing sequence between multiple guitar textures
        
        Args:
            audio_samples: List of guitar audio samples to morph between
            transition_length: Number of steps between each sample
            
        Returns:
            Complete morphing sequence
        """
        
        morphing_sequence = []
        
        for i in range(len(audio_samples) - 1):
            source = audio_samples[i]
            target = audio_samples[i + 1]
            
            # Create interpolation config
            config = InterpolationConfig(
                strategy=InterpolationStrategy.MUSICAL,
                num_steps=transition_length,
                preserve_content_from='source',
                smooth_transitions=True
            )
            
            # Interpolate between consecutive samples
            interpolation_result = self.explore_interpolation_space(source, target, config)
            
            # Add interpolated samples (excluding last to avoid duplication)
            morphing_sequence.extend(interpolation_result['interpolated_audio'][:-1])
        
        # Add final sample
        morphing_sequence.append(audio_samples[-1])
        
        return morphing_sequence


class TextureAnalyzer:
    """Analyzes guitar texture characteristics"""
    
    def __init__(self):
        pass
    
    def analyze(self, audio: torch.Tensor) -> TextureAnalysis:
        """
        Analyze texture characteristics of guitar audio
        
        Args:
            audio: [batch, channels] audio tensor (mel-spectrogram)
            
        Returns:
            TextureAnalysis with computed characteristics
        """
        
        # Convert to numpy for analysis
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        
        # Ensure we have a single sample
        if len(audio_np.shape) > 1:
            audio_np = audio_np[0]  # Take first sample
        
        # Basic spectral analysis
        # For mel-spectrogram input, compute texture features
        
        # Brightness (high frequency energy)
        high_freq_energy = np.mean(audio_np[-20:])  # Top 20 mel bins
        low_freq_energy = np.mean(audio_np[:20])    # Bottom 20 mel bins
        brightness = high_freq_energy / (low_freq_energy + 1e-8)
        brightness = np.clip(brightness, 0, 5) / 5  # Normalize to 0-1
        
        # Warmth (mid-low frequency balance)
        mid_freq_energy = np.mean(audio_np[20:40])  # Mid frequencies
        warmth = mid_freq_energy / (high_freq_energy + 1e-8)
        warmth = np.clip(warmth, 0, 3) / 3  # Normalize to 0-1
        
        # Saturation (energy variance indicating distortion)
        saturation = np.var(audio_np)
        saturation = np.clip(saturation, 0, 1)  # Already roughly normalized
        
        # Attack sharpness (energy in first few frames)
        attack_energy = np.mean(audio_np[:5]) if len(audio_np) > 5 else np.mean(audio_np)
        overall_energy = np.mean(audio_np)
        attack_sharpness = attack_energy / (overall_energy + 1e-8)
        attack_sharpness = np.clip(attack_sharpness, 0, 3) / 3
        
        # Sustain (energy decay rate)
        if len(audio_np) > 10:
            early_energy = np.mean(audio_np[:len(audio_np)//3])
            late_energy = np.mean(audio_np[2*len(audio_np)//3:])
            sustain_length = late_energy / (early_energy + 1e-8)
        else:
            sustain_length = 0.5
        sustain_length = np.clip(sustain_length, 0, 1)
        
        # Reverb (spectral smoothness)
        spectral_diff = np.diff(audio_np)
        reverb_amount = 1.0 / (1.0 + np.var(spectral_diff))  # Smoother = more reverb
        
        # Compression (dynamic range)
        dynamic_range = np.max(audio_np) - np.min(audio_np)
        compression_ratio = 1.0 - np.clip(dynamic_range, 0, 1)
        
        # Fret noise (high frequency irregularity)
        if len(audio_np) > 40:
            high_freq_section = audio_np[-20:]
            fret_noise_level = np.var(high_freq_section)
            fret_noise_level = np.clip(fret_noise_level, 0, 1)
        else:
            fret_noise_level = 0.0
        
        # Additional metrics
        spectral_centroid = np.sum(np.arange(len(audio_np)) * audio_np) / (np.sum(audio_np) + 1e-8)
        spectral_centroid = spectral_centroid / len(audio_np)  # Normalize
        
        harmonic_ratio = low_freq_energy / (high_freq_energy + 1e-8)
        harmonic_ratio = np.clip(harmonic_ratio, 0, 5) / 5
        
        # Confidence based on energy level
        confidence = np.clip(overall_energy, 0, 1)
        
        return TextureAnalysis(
            brightness=float(brightness),
            warmth=float(warmth),
            saturation=float(saturation),
            attack_sharpness=float(attack_sharpness),
            sustain_length=float(sustain_length),
            reverb_amount=float(reverb_amount),
            compression_ratio=float(compression_ratio),
            fret_noise_level=float(fret_noise_level),
            spectral_centroid=float(spectral_centroid),
            harmonic_ratio=float(harmonic_ratio),
            confidence=float(confidence)
        )


class SemanticTextureMapper:
    """Maps between semantic texture descriptions and latent representations"""
    
    def __init__(self, model: GuitarTextureModel):
        self.model = model
        self.texture_directions = {}
        
    def learn_texture_direction(self,
                              negative_samples: List[torch.Tensor],
                              positive_samples: List[torch.Tensor],
                              direction_name: str) -> torch.Tensor:
        """
        Learn direction in texture space corresponding to semantic concept
        
        Args:
            negative_samples: Audio samples with low amount of texture quality
            positive_samples: Audio samples with high amount of texture quality
            direction_name: Name of texture dimension (e.g., 'brightness')
            
        Returns:
            Direction vector in texture space
        """
        
        with torch.no_grad():
            # Extract texture representations
            negative_textures = []
            positive_textures = []
            
            for audio in negative_samples:
                output = self.model(audio, quantize_texture=False)
                negative_textures.append(output['texture_mu'])
            
            for audio in positive_samples:
                output = self.model(audio, quantize_texture=False)
                positive_textures.append(output['texture_mu'])
            
            # Compute centroids
            negative_centroid = torch.stack(negative_textures).mean(dim=0)
            positive_centroid = torch.stack(positive_textures).mean(dim=0)
            
            # Direction vector
            direction = F.normalize(positive_centroid - negative_centroid, dim=-1)
            
            # Store direction
            self.texture_directions[direction_name] = direction
            
            return direction
    
    def apply_semantic_transformation(self,
                                    audio: torch.Tensor,
                                    transformations: Dict[str, float]) -> torch.Tensor:
        """
        Apply semantic transformations to audio
        
        Args:
            audio: Input audio
            transformations: Dict of {dimension_name: intensity} transformations
            
        Returns:
            Transformed audio
        """
        
        with torch.no_grad():
            # Extract current representation
            output = self.model(audio, quantize_texture=False)
            current_texture = output['texture_mu']
            content = output['content_mu']
            
            # Apply transformations
            modified_texture = current_texture.clone()
            
            for dimension, intensity in transformations.items():
                if dimension in self.texture_directions:
                    direction = self.texture_directions[dimension]
                    modified_texture += intensity * direction
            
            # Reconstruct
            transformed = self.model.vae.decoder(modified_texture, content)
            
            return transformed


class TextureMemory:
    """Memory bank for storing and retrieving texture examples"""
    
    def __init__(self):
        self.stored_textures = []
        self.metadata = []
    
    def store_texture(self, 
                     audio: torch.Tensor,
                     texture_representation: torch.Tensor,
                     metadata: Dict[str, Any]):
        """Store texture example with metadata"""
        
        self.stored_textures.append({
            'audio': audio.detach().cpu(),
            'texture_rep': texture_representation.detach().cpu(),
            'metadata': metadata
        })
    
    def retrieve_similar(self, 
                        query_texture: torch.Tensor,
                        k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve k most similar textures"""
        
        if not self.stored_textures:
            return []
        
        similarities = []
        for item in self.stored_textures:
            stored_texture = item['texture_rep']
            similarity = F.cosine_similarity(
                query_texture.cpu(), 
                stored_texture, 
                dim=-1
            ).mean().item()
            similarities.append((similarity, item))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in similarities[:k]]


def create_guitar_texture_explorer(model: GuitarTextureModel, 
                                 config: GuitarTextureConfig) -> GuitarTextureExplorer:
    """Factory function to create texture explorer"""
    return GuitarTextureExplorer(model, config)


# Export main components
__all__ = [
    'GuitarTextureExplorer',
    'TextureAnalyzer',
    'SemanticTextureMapper',
    'TextureMemory',
    'InterpolationStrategy',
    'TextureDimension',
    'TextureAnalysis',
    'InterpolationConfig',
    'create_guitar_texture_explorer'
]
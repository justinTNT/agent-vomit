#!/usr/bin/env python3
import os
import re
from pathlib import Path

def camel_to_snake(name):
    """Convert CamelCase to snake_case."""
    # Insert underscore before capital letters
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before capital letter followed by lowercase
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    # Handle special cases
    s3 = s2.replace('STFT', 'stft').replace('ViT', 'vit')
    return s3.lower()

# First clean up the bad rename
bad_files = Path('candidates/agent_claude').glob('l*.py')
for f in bad_files:
    f.unlink()

# Get original files
original_files = {
    'AdaptiveComputation.py': 'adaptive_computation.py',
    'AntiAliasedConv.py': 'antialiased_conv.py',
    'AttentionDecoder.py': 'attention_decoder.py',
    'AutoEncoder.py': 'autoencoder.py',
    'CausalConv1d.py': 'causal_conv1d.py',
    'ContrastiveLearner.py': 'contrastive_learner.py',
    'ConvEncoder.py': 'conv_encoder.py',
    'CrossModalFusion.py': 'cross_modal_fusion.py',
    'DataSampler.py': 'data_sampler.py',
    'DataValidator.py': 'data_validator.py',
    'DataVersioner.py': 'data_versioner.py',
    'FeatureStore.py': 'feature_store.py',
    'GraphEncoder.py': 'graph_encoder.py',
    'MemoryBank.py': 'memory_bank.py',
    'MultiScaleSTFTLoss.py': 'multi_scale_stft_loss.py',
    'ResidualVectorQuantizer.py': 'residual_vector_quantizer.py',
    'SequenceEncoder.py': 'sequence_encoder.py',
    'SequenceToSequenceModel.py': 'sequence_to_sequence_model.py',
    'SetEncoder.py': 'set_encoder.py',
    'SnakeActivation.py': 'snake_activation.py',
    'StreamJoiner.py': 'stream_joiner.py',
    'StreamProcessor.py': 'stream_processor.py',
    'TimeSeriesEncoder.py': 'time_series_encoder.py',
    'TransformerBlock.py': 'transformer_block.py',
    'ViTPatchEncoder.py': 'vit_patch_encoder.py'
}

# Rename files
for old_name, new_name in original_files.items():
    old_path = Path('candidates/agent_claude') / old_name
    new_path = Path('candidates/agent_claude') / new_name
    if old_path.exists():
        old_path.rename(new_path)
        print(f"Renamed {old_name} -> {new_name}")

print("\nFiles after renaming:")
for f in sorted(Path('candidates/agent_claude').glob('*.py')):
    print(f"  {f.name}")
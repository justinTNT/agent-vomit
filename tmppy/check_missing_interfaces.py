#!/usr/bin/env python3
"""Check which module interfaces are missing from guidelines."""

# All 25 modules
ALL_MODULES = [
    'transformer_block',
    'conv_encoder', 
    'sequence_encoder',
    'attention_decoder',
    'vit_patch_encoder',
    'cross_modal_fusion',
    'time_series_encoder',
    'set_encoder',
    'contrastive_learner',
    'autoencoder_vae',
    'sequence_to_sequence',
    'graph_encoder',
    'memory_bank_retriever',
    'adaptive_computation',
    'stream_processor',
    'data_validator',
    'feature_store',
    'data_versioner',
    'stream_joiner',
    'data_sampler',
    'snake_activation',
    'causal_conv',
    'stft_loss',
    'antialiased_conv',
    'residual_vector_quantizer'
]

# Modules with interface specs in guidelines (based on grep)
DEFINED_INTERFACES = [
    'transformer_block',
    'conv_encoder',
    'data_versioner',
    'stream_joiner',
    'snake_activation'
]

print("Modules WITH interface specifications in guidelines:")
for m in DEFINED_INTERFACES:
    print(f"  ✓ {m}")

print("\nModules MISSING interface specifications:")
missing = [m for m in ALL_MODULES if m not in DEFINED_INTERFACES]
for m in missing:
    print(f"  ✗ {m}")

print(f"\nTotal: {len(DEFINED_INTERFACES)}/25 modules have interface specs")
print(f"Missing: {len(missing)}/25 modules")
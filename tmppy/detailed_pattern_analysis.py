#!/usr/bin/env python3
"""
Detailed analysis of return patterns to identify natural patterns vs artificial standardization.
"""

def analyze_natural_patterns():
    """
    Categorize modules by their natural complexity and expected return patterns.
    """
    
    # Classes that NATURALLY should return dictionaries (complex, multi-output)
    naturally_dict = {
        'AdaptiveComputationTime': 'Returns output + ponder_cost + n_updates + halting_probabilities',
        'PonderNet': 'Returns output + all_outputs + halt_probabilities + regularization_loss + expected_steps', 
        'UniversalTransformer': 'Complex adaptive computation with multiple outputs',
        'AutoEncoder': 'Returns reconstruction + latent representation',
        'VAE': 'Returns reconstruction + mu + logvar + latent (VAE outputs)',
        'ConditionalVAE': 'Returns reconstruction + mu + logvar + latent + condition',
        'ConvEncoder': 'Returns output + features + metadata (hierarchical features)',
        'AttentionDecoder': 'Returns output + features (decoder with attention)',
        'CrossModalAttention': 'Returns modal_1_output + modal_2_output + attention weights',
        'CrossModalFusion': 'Returns fused + modal_1_features + modal_2_features',
        'ContrastiveLearner': 'Returns loss + embeddings_1 + embeddings_2 + projections',
        'GraphEncoder': 'Returns graph_embedding + node_features + batch',
        'SequenceEncoder': 'Returns output + features + attention',
        'SequenceToSequenceModel': 'Returns logits + encoder_output + decoder_output',
        'SetEncoder': 'Returns pooled + elements + shape',
        'TimeSeriesEncoder': 'Returns pooled + sequence + shape',
        'ViTPatchEncoder': 'Returns pooled + tokens + patch_embeddings',
    }
    
    # Classes that NATURALLY should return tuples (structured multiple outputs)
    naturally_tuple = {
        'VectorQuantizer': 'Returns quantized + indices + commitment_loss',
        'ResidualVectorQuantizer': 'Returns quantized + indices + commitment_loss', 
        'ResidualVectorQuantizerWrapper': 'Returns quantized + indices + commitment_loss',
        'WaveNetBlock': 'Returns output + skip connection',
        'CausalConv1d': 'Returns output + state (when return_state=True)',
        'DataSampler': 'Returns sampled_data + sampled_labels',
        'DataValidator': 'Returns data + validation_results',
        'STFTLoss': 'Returns loss + x_stft + y_stft (when return_stfts=True)',
        'MultiScaleSTFTLoss': 'Returns total_loss + per_scale_losses',
    }
    
    # Classes that NATURALLY should return single tensors (simple operations)
    naturally_tensor = {
        'MultiHeadAttention': 'Simple attention operation',
        'FeedForwardNetwork': 'Simple feedforward layer',
        'DecoderBlock': 'Single block operation',
        'ConvDecoder': 'Simple decoder operation',
        'PositionalEncoding': 'Simple encoding operation',
        'SetAttention': 'Simple attention operation',
        'SetTransformerBlock': 'Single transformer block',
        'InducedSetAttentionBlock': 'Single attention block',
        'PoolingByMultiheadAttention': 'Pooling operation',
        'PatchEmbedding': 'Simple embedding operation',
        'PositionalEncoding2D': 'Simple position encoding',
        'LowPassFilter1d': 'Simple filtering operation',
        'AntialiasedConv1d': 'Simple convolution operation',
        'AntialiasedConvTranspose1d': 'Simple transposed convolution',
        'CausalConvTranspose1d': 'Simple causal convolution',
        'TemporalConvBlock': 'Single conv block',
        'PositionalEncodingTime': 'Simple time encoding',
        'GraphConvLayer': 'Single graph conv layer',
        'GraphAttentionLayer': 'Single attention layer',
        'EdgeConvLayer': 'Single edge conv layer',
        'ContrastiveLoss': 'Loss computation',
        'ProjectionHead': 'Simple projection',
        'SnakeActivation': 'Activation function',
        'SpectralConvergenceLoss': 'Loss computation',
        'LogSTFTMagnitudeLoss': 'Loss computation',  
        'MelSpectrogramLoss': 'Loss computation',
        'StreamJoiner': 'Simple joining operation',
        'DataVersioner': 'Simple versioning operation',
        'FeatureStore': 'Simple storage operation',
        'MemoryBank': 'Simple memory operation',
        'Retriever': 'Simple retrieval operation',
    }
    
    # Special cases that might return different types based on parameters
    conditional_returns = {
        'TransformerBlock': 'Returns dict with output for consistency, could be tensor',
        'SnakeBeta': 'Returns dict with output, could be tensor', 
        'StreamProcessor': 'Returns tensor or None based on processing state',
        'BlurPool1d': 'Simple pooling, should return tensor',
    }
    
    print("🧠 NATURAL PATTERN ANALYSIS")
    print("=" * 80)
    
    print(f"\n📚 NATURALLY COMPLEX (Should return Dict): {len(naturally_dict)} classes")
    print("These classes have multiple meaningful outputs that benefit from named access:")
    for class_name, reason in naturally_dict.items():
        print(f"  ✅ {class_name}: {reason}")
    
    print(f"\n📝 NATURALLY STRUCTURED (Should return Tuple): {len(naturally_tuple)} classes") 
    print("These classes have multiple outputs in a specific order:")
    for class_name, reason in naturally_tuple.items():
        print(f"  ✅ {class_name}: {reason}")
    
    print(f"\n📦 NATURALLY SIMPLE (Should return Tensor): {len(naturally_tensor)} classes")
    print("These are simple operations with single primary output:")
    for class_name, reason in naturally_tensor.items():
        print(f"  ✅ {class_name}: {reason}")
    
    print(f"\n🤔 CONDITIONAL/SPECIAL CASES: {len(conditional_returns)} classes")
    for class_name, reason in conditional_returns.items():
        print(f"  ⚠️  {class_name}: {reason}")
    
    print(f"\n\n📊 NATURAL PATTERN DISTRIBUTION")
    print("=" * 80)
    total_classes = len(naturally_dict) + len(naturally_tuple) + len(naturally_tensor) + len(conditional_returns)
    dict_pct = len(naturally_dict) / total_classes * 100
    tuple_pct = len(naturally_tuple) / total_classes * 100  
    tensor_pct = len(naturally_tensor) / total_classes * 100
    
    print(f"Naturally Complex (Dict):     {len(naturally_dict):2d} classes ({dict_pct:5.1f}%)")
    print(f"Naturally Structured (Tuple): {len(naturally_tuple):2d} classes ({tuple_pct:5.1f}%)")
    print(f"Naturally Simple (Tensor):    {len(naturally_tensor):2d} classes ({tensor_pct:5.1f}%)")
    print(f"Conditional/Special:          {len(conditional_returns):2d} classes")
    
    print(f"\n\n🎯 RECOMMENDATIONS BASED ON NATURAL PATTERNS")
    print("=" * 80)
    
    print("✅ PRESERVE MIXED PATTERNS - This is actually CORRECT design:")
    print("   • Complex multi-output modules → Dictionary returns")
    print("   • Simple single-output modules → Direct tensor returns")  
    print("   • Structured multi-output → Tuple returns")
    print()
    print("❌ AVOID artificial uniformity that forces inappropriate return types:")
    print("   • DON'T force simple operations to return dicts")
    print("   • DON'T force complex operations to return single tensors")
    print()
    print("🔧 The original codebase shows GOOD DESIGN PRINCIPLES:")
    print("   • Return type matches the semantic complexity")
    print("   • Users get appropriate interface for each module type")
    print("   • No unnecessary wrapping of simple operations")

if __name__ == "__main__":
    analyze_natural_patterns()
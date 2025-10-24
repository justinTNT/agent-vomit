#!/usr/bin/env python3
"""
Generate reference test vectors for language-independent module testing
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from modules.guitar_texture_architecture import (
    GuitarTextureModel, GuitarTextureConfig, create_guitar_texture_model
)

def tensor_to_list(tensor):
    """Convert tensor to JSON-serializable list"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().tolist()
    return tensor

def generate_test_vectors(config_name: str = "default") -> Dict[str, Any]:
    """Generate comprehensive test vectors"""
    
    # Create model with known config
    config = GuitarTextureConfig(
        texture_dim=64,
        content_dim=128,
        codebook_sizes=[16, 64, 256],  # Small for testing
        n_mels=40
    )
    
    model = create_guitar_texture_model(config)
    model.eval()
    
    test_vectors = {
        "config": {
            "texture_dim": config.texture_dim,
            "content_dim": config.content_dim,
            "codebook_sizes": config.codebook_sizes,
            "n_mels": config.n_mels
        },
        "tests": {}
    }
    
    # Set deterministic behavior
    torch.manual_seed(42)
    
    # Test 1: Hierarchical Quantization
    with torch.no_grad():
        texture_input = torch.randn(2, config.texture_dim)
        quant_output = model.quantizer(texture_input)
        
        test_vectors["tests"]["hierarchical_quantization"] = {
            "input": tensor_to_list(texture_input),
            "outputs": {
                "quantized": tensor_to_list(quant_output["quantized"]),
                "level_indices": {
                    level: tensor_to_list(indices) 
                    for level, indices in quant_output["level_indices"].items()
                },
                "commitment_loss": float(quant_output["commitment_loss"])
            }
        }
    
    # Test 2: Texture-Content Disentanglement
    torch.manual_seed(42)
    with torch.no_grad():
        audio_input = torch.randn(1, 1, 256)  # Small audio chunk
        vae_output = model.vae(audio_input, sample_texture=False, sample_content=False)
        
        test_vectors["tests"]["texture_content_disentanglement"] = {
            "input": tensor_to_list(audio_input),
            "outputs": {
                "texture_mu": tensor_to_list(vae_output["texture_mu"]),
                "content_mu": tensor_to_list(vae_output["content_mu"]),
                "reconstruction": tensor_to_list(vae_output["reconstruction"])
            }
        }
    
    # Test 3: Spherical Interpolation
    torch.manual_seed(42)
    with torch.no_grad():
        from modules.guitar_texture_architecture import spherical_interpolation
        
        a = torch.randn(1, config.texture_dim)
        b = torch.randn(1, config.texture_dim)
        
        interpolations = {}
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = spherical_interpolation(a, b, alpha)
            interpolations[f"alpha_{alpha}"] = tensor_to_list(result)
        
        test_vectors["tests"]["spherical_interpolation"] = {
            "inputs": {
                "vector_a": tensor_to_list(a),
                "vector_b": tensor_to_list(b)
            },
            "outputs": interpolations
        }
    
    # Test 4: Texture Transfer
    torch.manual_seed(42)
    with torch.no_grad():
        source_audio = torch.randn(1, 1, 256)
        target_audio = torch.randn(1, 1, 256)
        
        transfer_result = model.vae.texture_transfer(source_audio, target_audio)
        
        test_vectors["tests"]["texture_transfer"] = {
            "inputs": {
                "source_audio": tensor_to_list(source_audio),
                "target_audio": tensor_to_list(target_audio)
            },
            "output": tensor_to_list(transfer_result)
        }
    
    return test_vectors

def save_test_vectors(vectors: Dict[str, Any], output_path: str):
    """Save test vectors to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(vectors, f, indent=2)
    print(f"Test vectors saved to {output_path}")

if __name__ == "__main__":
    vectors = generate_test_vectors()
    save_test_vectors(vectors, "test_vectors/guitar_texture_reference.json")
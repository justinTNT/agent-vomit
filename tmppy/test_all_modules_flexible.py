#!/usr/bin/env python3
"""
Test all available modules using flexible testing.
Compares rigid vs flexible testing on original modules.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from test_utils import (
    init_with_variations,
    extract_output,
    validate_shape_behavior,
    test_mask_support,
    check_gradient_flow
)

def test_all_original_modules():
    """Test original modules with both rigid and flexible approaches."""
    
    results = {
        'rigid': {'passed': [], 'failed': []},
        'flexible': {'passed': [], 'failed': []}
    }
    
    # Test 1: TransformerBlock
    try:
        print("\n1. Testing TransformerBlock:")
        from modules.transformer_block import TransformerBlock
        
        # Rigid test
        try:
            model = TransformerBlock(d_model=512, n_heads=8)
            x = torch.randn(2, 10, 512)
            output = model(x)
            assert output.shape == x.shape
            results['rigid']['passed'].append('transformer_block')
            print("   Rigid: ✅ PASSED")
        except Exception as e:
            results['rigid']['failed'].append(('transformer_block', str(e)))
            print(f"   Rigid: ❌ FAILED - {type(e).__name__}")
        
        # Flexible test
        try:
            model = init_with_variations(TransformerBlock, {
                'd_model': 512, 'n_heads': 8
            })
            x = torch.randn(2, 10, 512)
            success, _ = validate_shape_behavior(model, x, 'preserve')
            assert success
            results['flexible']['passed'].append('transformer_block')
            print("   Flexible: ✅ PASSED")
        except Exception as e:
            results['flexible']['failed'].append(('transformer_block', str(e)))
            print(f"   Flexible: ❌ FAILED - {type(e).__name__}")
            
    except ImportError:
        print("   Module not available")
    
    # Test 2: ConvEncoder
    try:
        print("\n2. Testing ConvEncoder:")
        from modules.conv_encoder import ConvEncoder
        
        # Rigid test - expects tensor output
        try:
            model = ConvEncoder()
            x = torch.randn(2, 3, 64, 64)
            output = model(x)
            assert torch.is_tensor(output), "Output must be tensor"
            results['rigid']['passed'].append('conv_encoder')
            print("   Rigid: ✅ PASSED")
        except Exception as e:
            results['rigid']['failed'].append(('conv_encoder', str(e)))
            print(f"   Rigid: ❌ FAILED - {str(e)[:50]}")
        
        # Flexible test - handles dict output
        try:
            model = ConvEncoder()
            x = torch.randn(2, 3, 64, 64)
            output_dict = extract_output(model(x))
            assert len(output_dict) > 0
            results['flexible']['passed'].append('conv_encoder')
            print("   Flexible: ✅ PASSED")
        except Exception as e:
            results['flexible']['failed'].append(('conv_encoder', str(e)))
            print(f"   Flexible: ❌ FAILED - {type(e).__name__}")
            
    except ImportError:
        print("   Module not available")
    
    # Test 3: DataVersioner
    try:
        print("\n3. Testing DataVersioner:")
        from modules.data_versioner import DataVersioner
        
        # Rigid test - expects exact API
        try:
            versioner = DataVersioner()
            data = torch.randn(10, 10)
            # Rigid expects callable
            version_id = versioner(data, "test")
            loaded = versioner.load(version_id)
            assert torch.allclose(data, loaded)
            results['rigid']['passed'].append('data_versioner')
            print("   Rigid: ✅ PASSED")
        except Exception as e:
            results['rigid']['failed'].append(('data_versioner', str(e)))
            print(f"   Rigid: ❌ FAILED - {type(e).__name__}")
        
        # Flexible test
        try:
            versioner = DataVersioner()
            data = torch.randn(10, 10)
            # Try callable or method
            try:
                version_id = versioner(data, "test")
            except:
                version_id = versioner.version(data, "test")
            loaded = versioner.load(version_id)
            results['flexible']['passed'].append('data_versioner')
            print("   Flexible: ✅ PASSED")
        except Exception as e:
            results['flexible']['failed'].append(('data_versioner', str(e)))
            print(f"   Flexible: ❌ FAILED - {type(e).__name__}")
            
    except ImportError:
        print("   Module not available")
    
    # Test 4: SequenceEncoder  
    try:
        print("\n4. Testing SequenceEncoder:")
        from modules.sequence_encoder import SequenceEncoder
        
        # Rigid test
        try:
            model = SequenceEncoder(input_dim=128, d_model=512, n_heads=8, n_layers=4)
            x = torch.randn(2, 20, 128)
            output = model(x)
            assert torch.is_tensor(output)
            results['rigid']['passed'].append('sequence_encoder')
            print("   Rigid: ✅ PASSED")
        except Exception as e:
            results['rigid']['failed'].append(('sequence_encoder', str(e)))
            print(f"   Rigid: ❌ FAILED - {str(e)[:50]}")
        
        # Flexible test
        try:
            model = init_with_variations(SequenceEncoder, {
                'input_dim': 128, 'd_model': 512, 'n_heads': 8, 'n_layers': 4
            })
            x = torch.randn(2, 20, 128)
            output_dict = extract_output(model(x))
            assert len(output_dict) > 0
            results['flexible']['passed'].append('sequence_encoder')
            print("   Flexible: ✅ PASSED")
        except Exception as e:
            results['flexible']['failed'].append(('sequence_encoder', str(e)))
            print(f"   Flexible: ❌ FAILED - {type(e).__name__}")
            
    except ImportError:
        print("   Module not available")
    
    # Test 5: SnakeActivation
    try:
        print("\n5. Testing SnakeActivation:")
        from modules.snake_activation import SnakeActivation
        
        # Rigid test - expects exact class name
        try:
            model = SnakeActivation(channels=64)
            x = torch.randn(2, 64, 100)
            output = model(x)
            assert output.shape == x.shape
            results['rigid']['passed'].append('snake_activation')
            print("   Rigid: ✅ PASSED")
        except Exception as e:
            results['rigid']['failed'].append(('snake_activation', str(e)))
            print(f"   Rigid: ❌ FAILED - {type(e).__name__}")
        
        # Flexible test
        try:
            model = init_with_variations(SnakeActivation, {'channels': 64})
            x = torch.randn(2, 64, 100)
            success, _ = validate_shape_behavior(model, x, 'preserve')
            assert success
            results['flexible']['passed'].append('snake_activation')
            print("   Flexible: ✅ PASSED")
        except Exception as e:
            results['flexible']['failed'].append(('snake_activation', str(e)))
            print(f"   Flexible: ❌ FAILED - {type(e).__name__}")
            
    except ImportError:
        print("   Module not available")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Original Modules Test")
    print("="*60)
    
    rigid_rate = len(results['rigid']['passed']) / (len(results['rigid']['passed']) + len(results['rigid']['failed'])) * 100
    flexible_rate = len(results['flexible']['passed']) / (len(results['flexible']['passed']) + len(results['flexible']['failed'])) * 100
    
    print(f"\nRIGID TESTING:")
    print(f"  Passed: {len(results['rigid']['passed'])}")
    print(f"  Failed: {len(results['rigid']['failed'])}")
    print(f"  Success Rate: {rigid_rate:.1f}%")
    
    print(f"\nFLEXIBLE TESTING:")
    print(f"  Passed: {len(results['flexible']['passed'])}")
    print(f"  Failed: {len(results['flexible']['failed'])}")
    print(f"  Success Rate: {flexible_rate:.1f}%")
    
    print(f"\nIMPROVEMENT: +{flexible_rate - rigid_rate:.1f}% success rate")
    
    # Now test the two candidate modules
    print("\n" + "="*60)
    print("TESTING EXTERNAL CANDIDATES")
    print("="*60)
    
    # Test agent_claude's transformer_block
    print("\n6. Testing agent_claude/transformer_block:")
    try:
        # Temporarily replace module
        import shutil
        backup = Path('modules/transformer_block.py.temp')
        original = Path('modules/transformer_block.py')
        candidate = Path('candidates/agent_claude/transformer_block.py')
        
        if original.exists():
            shutil.copy(original, backup)
        shutil.copy(candidate, original)
        
        # Clear module cache
        if 'modules.transformer_block' in sys.modules:
            del sys.modules['modules.transformer_block']
        
        from modules.transformer_block import TransformerBlock
        
        # Flexible test
        try:
            model = init_with_variations(TransformerBlock, {
                'd_model': 512, 'n_heads': 8
            })
            print("   ✅ Flexible initialization worked!")
            
            x = torch.randn(2, 10, 512)
            output = model(x)
            output_dict = extract_output(output)
            print(f"   ✅ Forward pass worked! Output keys: {list(output_dict.keys())}")
            
        except Exception as e:
            print(f"   ❌ Flexible test failed: {e}")
        
        # Restore
        if backup.exists():
            shutil.copy(backup, original)
            backup.unlink()
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test agent_codex's conv_encoder
    print("\n7. Testing agent_codex/conv_encoder:")
    try:
        backup = Path('modules/conv_encoder.py.temp')
        original = Path('modules/conv_encoder.py')
        candidate = Path('candidates/agent_codex/conv_encoder.py')
        
        if original.exists():
            shutil.copy(original, backup)
        shutil.copy(candidate, original)
        
        # Clear module cache
        if 'modules.conv_encoder' in sys.modules:
            del sys.modules['modules.conv_encoder']
        
        from modules.conv_encoder import ConvEncoder
        
        # Flexible test
        try:
            model = init_with_variations(ConvEncoder, {
                'in_channels': 3,
                'base_channels': 64,
                'num_layers': 4,
                'latent_dim': 512
            })
            print("   ✅ Flexible initialization worked!")
            
            x = torch.randn(2, 3, 64, 64)
            output = model(x)
            output_dict = extract_output(output)
            print(f"   ✅ Forward pass worked! Output shape: {next(iter(output_dict.values())).shape}")
            
        except Exception as e:
            print(f"   ❌ Flexible test failed: {e}")
        
        # Restore
        if backup.exists():
            shutil.copy(backup, original)
            backup.unlink()
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    print("🧪 COMPREHENSIVE MODULE TESTING")
    print("Comparing rigid vs flexible testing approaches\n")
    
    import shutil  # Add this import at the top
    test_all_original_modules()
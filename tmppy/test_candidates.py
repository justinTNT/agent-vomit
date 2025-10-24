#!/usr/bin/env python3
"""
Test harness for validating candidate module implementations.

Usage:
    python test_candidates.py [candidate_dir] [--module MODULE_NAME] [--verbose]
    
Examples:
    # Test all modules from a candidate
    python test_candidates.py candidates/agent_alpha
    
    # Test specific module
    python test_candidates.py candidates/agent_alpha --module transformer_block
    
    # Verbose output
    python test_candidates.py candidates/agent_alpha --verbose
"""

import os
import sys
import importlib.util
import argparse
import json
from datetime import datetime
from pathlib import Path
import traceback
import subprocess


# Module test mapping
MODULE_TESTS = {
    # ML Components
    'transformer_block': 'test_transformer_block',
    'conv_encoder': 'test_conv_encoder',
    'sequence_encoder': 'test_sequence_encoder',
    'attention_decoder': 'test_attention_decoder',
    'vit_patch_encoder': 'test_vit_patch_encoder',
    'cross_modal': 'test_cross_modal',
    'timeseries_encoder': 'test_timeseries_encoder',
    'set_encoder': 'test_set_encoder',
    'contrastive': 'test_contrastive',
    'autoencoder': 'test_autoencoder',
    'seq2seq': 'test_seq2seq',
    'graph_encoder': 'test_graph_encoder',
    'memory_retriever': 'test_memory_retriever',
    'adaptive_computation': 'test_adaptive_computation',
    
    # Data Pipeline Components
    'stream_processor': 'test_stream_processor',
    'data_validator': 'test_data_validator',
    'feature_store': 'test_feature_store',
    'data_versioner': 'test_data_versioner',
    'stream_joiner': 'test_stream_joiner',
    'data_sampler': 'test_data_sampler',
    
    # Audio Components
    'snake_activation': 'test_snake_activation',
    'causal_conv': 'test_causal_conv',
    'stft_loss': 'test_stft_loss',
    'antialiased_conv': 'test_antialiased_conv',
    'residual_vector_quantizer': 'test_residual_vector_quantizer',
}


class CandidateTester:
    def __init__(self, candidate_dir, verbose=False):
        self.candidate_dir = Path(candidate_dir)
        self.verbose = verbose
        self.results = {
            'candidate': str(candidate_dir),
            'timestamp': datetime.now().isoformat(),
            'modules': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'missing': 0,
                'error': 0
            }
        }
    
    def test_module(self, module_name):
        """Test a single module."""
        print(f"\n{'='*60}")
        print(f"Testing {module_name}")
        print(f"{'='*60}")
        
        result = {
            'status': 'unknown',
            'error': None,
            'test_output': None
        }
        
        # Check if module exists
        module_path = self.candidate_dir / f"{module_name}.py"
        if not module_path.exists():
            result['status'] = 'missing'
            result['error'] = f"Module file not found: {module_path}"
            print(f"❌ MISSING: {module_path}")
            self.results['summary']['missing'] += 1
            return result
        
        # Get test file
        test_name = MODULE_TESTS.get(module_name)
        if not test_name:
            result['status'] = 'error'
            result['error'] = f"No test mapping for module: {module_name}"
            print(f"❌ ERROR: No test found for {module_name}")
            self.results['summary']['error'] += 1
            return result
        
        test_path = Path('tests') / f"{test_name}.py"
        if not test_path.exists():
            result['status'] = 'error'
            result['error'] = f"Test file not found: {test_path}"
            print(f"❌ ERROR: Test file missing: {test_path}")
            self.results['summary']['error'] += 1
            return result
        
        # Create temporary symlink to candidate module
        original_module_path = Path('modules') / f"{module_name}.py"
        backup_path = None
        
        try:
            # Backup original if it exists
            if original_module_path.exists():
                backup_path = original_module_path.with_suffix('.py.backup')
                original_module_path.rename(backup_path)
            
            # Link candidate module
            os.symlink(module_path.absolute(), original_module_path)
            
            # Run test
            print(f"Running {test_path}...")
            
            cmd = [sys.executable, str(test_path)]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, 'PYTHONPATH': str(Path.cwd())}
            )
            
            output, _ = process.communicate()
            
            if self.verbose:
                print(output)
            
            if process.returncode == 0:
                result['status'] = 'passed'
                result['test_output'] = output
                print(f"✅ PASSED: {module_name}")
                self.results['summary']['passed'] += 1
            else:
                result['status'] = 'failed'
                result['test_output'] = output
                result['error'] = f"Tests failed with return code {process.returncode}"
                print(f"❌ FAILED: {module_name}")
                if not self.verbose:
                    print("Use --verbose to see test output")
                self.results['summary']['failed'] += 1
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['test_output'] = traceback.format_exc()
            print(f"❌ ERROR: {str(e)}")
            self.results['summary']['error'] += 1
            
        finally:
            # Restore original module
            if original_module_path.exists() or original_module_path.is_symlink():
                original_module_path.unlink()
            if backup_path and backup_path.exists():
                backup_path.rename(original_module_path)
        
        return result
    
    def test_all(self):
        """Test all modules."""
        for module_name in MODULE_TESTS.keys():
            self.results['modules'][module_name] = self.test_module(module_name)
            self.results['summary']['total'] += 1
    
    def test_single(self, module_name):
        """Test a single module."""
        if module_name not in MODULE_TESTS:
            print(f"Unknown module: {module_name}")
            print(f"Available modules: {', '.join(MODULE_TESTS.keys())}")
            return
        
        self.results['modules'][module_name] = self.test_module(module_name)
        self.results['summary']['total'] += 1
    
    def save_results(self):
        """Save test results to file."""
        results_dir = Path('test_results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        candidate_name = self.candidate_dir.name
        results_file = results_dir / f"{candidate_name}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return results_file
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Candidate: {self.candidate_dir}")
        print(f"Total modules: {self.results['summary']['total']}")
        print(f"✅ Passed: {self.results['summary']['passed']}")
        print(f"❌ Failed: {self.results['summary']['failed']}")
        print(f"📭 Missing: {self.results['summary']['missing']}")
        print(f"🚨 Errors: {self.results['summary']['error']}")
        
        success_rate = 0
        if self.results['summary']['total'] > 0:
            success_rate = (self.results['summary']['passed'] / 
                          self.results['summary']['total'] * 100)
        
        print(f"\nSuccess rate: {success_rate:.1f}%")
        
        # List failed modules
        if self.results['summary']['failed'] > 0:
            print("\nFailed modules:")
            for module, result in self.results['modules'].items():
                if result['status'] == 'failed':
                    print(f"  - {module}")


def main():
    parser = argparse.ArgumentParser(
        description='Test candidate module implementations'
    )
    parser.add_argument(
        'candidate_dir',
        help='Directory containing candidate modules'
    )
    parser.add_argument(
        '--module',
        help='Test specific module only'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed test output'
    )
    
    args = parser.parse_args()
    
    # Validate candidate directory
    candidate_dir = Path(args.candidate_dir)
    if not candidate_dir.exists():
        print(f"Error: Candidate directory not found: {candidate_dir}")
        sys.exit(1)
    
    # Create tester
    tester = CandidateTester(candidate_dir, verbose=args.verbose)
    
    # Run tests
    if args.module:
        tester.test_single(args.module)
    else:
        tester.test_all()
    
    # Save results and print summary
    tester.save_results()
    tester.print_summary()
    
    # Exit with appropriate code
    if tester.results['summary']['failed'] > 0 or \
       tester.results['summary']['error'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
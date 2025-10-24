#!/usr/bin/env python3
"""
Visualize test results with charts and graphs.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_latest_results():
    """Load the most recent comprehensive test results."""
    results_dir = Path('test_results')
    claude_files = list(results_dir.glob('agent_claude_comprehensive_*.json'))
    codex_files = list(results_dir.glob('agent_codex_comprehensive_*.json'))
    
    if not claude_files or not codex_files:
        print("No comprehensive test results found!")
        return None, None
    
    # Get most recent files
    claude_file = max(claude_files, key=lambda p: p.stat().st_mtime)
    codex_file = max(codex_files, key=lambda p: p.stat().st_mtime)
    
    with open(claude_file) as f:
        claude_results = json.load(f)
    
    with open(codex_file) as f:
        codex_results = json.load(f)
    
    return claude_results, codex_results


def print_visual_summary(claude_results, codex_results):
    """Print a visual summary of results."""
    print("\n" + "="*80)
    print("VISUAL TEST RESULTS SUMMARY")
    print("="*80)
    
    # Status symbols
    symbols = {
        'passed': '✅',
        'partial': '⚠️',
        'failed': '❌',
        'missing': '📭',
        'error': '🚨'
    }
    
    # Print side-by-side comparison
    print("\nModule Status Comparison:")
    print("-" * 80)
    print(f"{'Module':<30} {'Agent Claude':<15} {'Agent Codex':<15} {'Match'}")
    print("-" * 80)
    
    matches = 0
    for module_name in sorted(claude_results['modules'].keys()):
        claude_status = claude_results['modules'][module_name]['status']
        codex_status = codex_results['modules'].get(module_name, {}).get('status', 'unknown')
        
        claude_sym = symbols.get(claude_status, '❓')
        codex_sym = symbols.get(codex_status, '❓')
        match = '✓' if claude_status == codex_status else '✗'
        
        if claude_status == codex_status:
            matches += 1
        
        print(f"{module_name:<30} {claude_sym} {claude_status:<13} {codex_sym} {codex_status:<13} {match}")
    
    print("-" * 80)
    print(f"Matching results: {matches}/25 ({matches/25*100:.1f}%)")
    
    # Print bar charts
    print("\n\nStatus Distribution:")
    print("-" * 80)
    
    for agent_name, results in [("Agent Claude", claude_results), ("Agent Codex", codex_results)]:
        print(f"\n{agent_name}:")
        summary = results['summary']
        total = summary['total']
        
        for status in ['passed', 'partial', 'failed', 'missing', 'error']:
            count = summary[status]
            percentage = count / total * 100
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            
            print(f"  {symbols[status]} {status:<8} [{bar:<50}] {count:>2}/{total} ({percentage:>4.1f}%)")
    
    # Print test type failure analysis
    print("\n\nTest Type Failure Analysis:")
    print("-" * 80)
    
    for agent_name, results in [("Agent Claude", claude_results), ("Agent Codex", codex_results)]:
        print(f"\n{agent_name}:")
        test_failures = defaultdict(int)
        
        for module in results['modules'].values():
            for test_name, error in module.get('tests_failed', []):
                # Categorize by test type
                if 'forward' in test_name:
                    test_failures['Forward Pass'] += 1
                elif 'shape' in test_name:
                    test_failures['Shape Validation'] += 1
                elif 'gradient' in test_name:
                    test_failures['Gradient Flow'] += 1
                elif 'method' in test_name:
                    test_failures['Method Missing'] += 1
                elif 'callable' in test_name:
                    test_failures['Callable Interface'] += 1
                else:
                    test_failures['Other'] += 1
        
        total_failures = sum(test_failures.values())
        if total_failures > 0:
            for test_type, count in sorted(test_failures.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_failures * 100
                bar_length = int(percentage / 2)
                bar = '▓' * bar_length
                print(f"  {test_type:<20} [{bar:<50}] {count:>2} ({percentage:>4.1f}%)")
        else:
            print("  No test failures!")
    
    # Module categories analysis
    print("\n\nModule Category Analysis:")
    print("-" * 80)
    
    categories = {
        'Transformers': ['transformer_block', 'attention_decoder', 'sequence_encoder', 'cross_modal_fusion'],
        'Vision': ['conv_encoder', 'vit_patch_encoder', 'antialiased_conv'],
        'Audio': ['snake_activation', 'stft_loss', 'causal_conv'],
        'Data Management': ['data_sampler', 'data_validator', 'data_versioner', 'feature_store', 'stream_joiner', 'stream_processor'],
        'Memory/Attention': ['memory_bank_retriever', 'adaptive_computation'],
        'Generative': ['autoencoder_vae', 'sequence_to_sequence', 'residual_vector_quantizer'],
        'Specialized': ['contrastive_learner', 'set_encoder', 'graph_encoder', 'time_series_encoder']
    }
    
    for agent_name, results in [("Agent Claude", claude_results), ("Agent Codex", codex_results)]:
        print(f"\n{agent_name} - Success by Category:")
        
        for category, modules in categories.items():
            working = 0
            total = 0
            
            for module in modules:
                if module in results['modules']:
                    total += 1
                    status = results['modules'][module]['status']
                    if status in ['passed', 'partial']:
                        working += 1
            
            if total > 0:
                success_rate = working / total * 100
                bar_length = int(success_rate / 2)
                bar = '▰' * bar_length + '▱' * (50 - bar_length)
                print(f"  {category:<15} [{bar}] {working}/{total} ({success_rate:>4.1f}%)")


def main():
    """Main function."""
    claude_results, codex_results = load_latest_results()
    
    if not claude_results or not codex_results:
        return
    
    print_visual_summary(claude_results, codex_results)
    
    # Save timestamps
    print(f"\n\nResults from:")
    print(f"  Claude: {claude_results['timestamp']}")
    print(f"  Codex:  {codex_results['timestamp']}")


if __name__ == "__main__":
    main()
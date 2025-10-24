#!/usr/bin/env python3
"""
Analysis of failing modules from comprehensive coverage results
Identifies RAVE-critical vs non-critical failures
"""

import json

def analyze_failures():
    # Load the comprehensive results
    with open('comprehensive_coverage_results.json', 'r') as f:
        data = json.load(f)
    
    # Extract all failing modules
    failures = []
    for result in data['all_results']:
        if not result['result']['module_passed']:
            failure_info = {
                'module_file': result['module_file'],
                'class_name': result['class_name'],
                'directory': result['directory'],
                'error_type': 'init_error' if result['result']['init_error'] else 'forward_error',
                'error_message': result['result']['init_error'] or (
                    result['result']['results'][0]['error'] if result['result']['results'] else 'Unknown error'
                )
            }
            failures.append(failure_info)
    
    print(f"Found {len(failures)} failing modules out of {data['metadata']['classes_discovered']} total")
    print("=" * 80)
    
    # Categorize by RAVE criticality
    rave_critical = []
    rave_useful = []
    non_critical = []
    
    # RAVE-critical modules (essential for audio VAE functionality)
    critical_keywords = [
        'autoencoder', 'vae', 'encoder', 'decoder', 'conv', 'causal', 
        'stft', 'spectral', 'antialiased', 'quantizer', 'audio', 'residual'
    ]
    
    # RAVE-useful modules (beneficial for audio processing)
    useful_keywords = [
        'attention', 'transformer', 'time_series', 'sequence', 'snake', 
        'mel', 'loss', 'discriminator', 'gan'
    ]
    
    for failure in failures:
        module_name = failure['module_file'].lower() + '/' + failure['class_name'].lower()
        
        is_critical = any(keyword in module_name for keyword in critical_keywords)
        is_useful = any(keyword in module_name for keyword in useful_keywords)
        
        if is_critical:
            rave_critical.append(failure)
        elif is_useful:
            rave_useful.append(failure)
        else:
            non_critical.append(failure)
    
    # Print analysis
    print(f"RAVE-CRITICAL FAILURES ({len(rave_critical)}):")
    print("=" * 50)
    for i, failure in enumerate(rave_critical, 1):
        print(f"{i}. {failure['class_name']} ({failure['module_file']})")
        print(f"   Error: {failure['error_message'][:100]}...")
        print(f"   Type: {failure['error_type']}")
        print()
    
    print(f"RAVE-USEFUL FAILURES ({len(rave_useful)}):")
    print("=" * 50)
    for i, failure in enumerate(rave_useful, 1):
        print(f"{i}. {failure['class_name']} ({failure['module_file']})")
        print(f"   Error: {failure['error_message'][:100]}...")
        print(f"   Type: {failure['error_type']}")
        print()
    
    print(f"NON-CRITICAL FAILURES ({len(non_critical)}):")
    print("=" * 50)
    for i, failure in enumerate(non_critical, 1):
        print(f"{i}. {failure['class_name']} ({failure['module_file']})")
        print(f"   Error: {failure['error_message'][:100]}...")
        print(f"   Type: {failure['error_type']}")
        print()
    
    # Categorize errors by type
    print("ERROR TYPE BREAKDOWN:")
    print("=" * 50)
    
    init_errors = [f for f in failures if f['error_type'] == 'init_error']
    forward_errors = [f for f in failures if f['error_type'] == 'forward_error']
    
    print(f"Initialization errors: {len(init_errors)}")
    print(f"Forward pass errors: {len(forward_errors)}")
    print()
    
    # Common error patterns
    missing_args = [f for f in init_errors if 'missing' in f['error_message']]
    unexpected_args = [f for f in init_errors if 'unexpected keyword argument' in f['error_message']]
    shape_errors = [f for f in forward_errors if any(x in f['error_message'].lower() for x in ['shape', 'dimension', 'size', 'channel'])]
    
    print(f"Missing required arguments: {len(missing_args)}")
    print(f"Unexpected keyword arguments: {len(unexpected_args)}")
    print(f"Shape/dimension mismatches: {len(shape_errors)}")
    print()
    
    # Priority recommendations
    print("PRIORITY RECOMMENDATIONS:")
    print("=" * 50)
    print("1. IMMEDIATE ATTENTION (RAVE-Critical):")
    priority_critical = [f for f in rave_critical if f['class_name'] in [
        'AutoEncoder', 'VAE', 'ConvDecoder', 'ConditionalVAE', 
        'ResidualVectorQuantizer', 'CausalConvTranspose1d', 'AntialiasedConv1d'
    ]]
    
    for failure in priority_critical:
        print(f"   - {failure['class_name']}: {failure['error_message'][:80]}...")
    
    print()
    print("2. SECONDARY PRIORITY (Audio Processing):")
    audio_failures = [f for f in rave_critical if 'audio' in f['module_file'].lower() or 'stft' in f['module_file'].lower()]
    for failure in audio_failures[:5]:  # Top 5
        print(f"   - {failure['class_name']}: {failure['error_message'][:80]}...")
    
    print()
    print("3. TERTIARY PRIORITY (Supporting Components):")
    for failure in rave_useful[:3]:  # Top 3
        print(f"   - {failure['class_name']}: {failure['error_message'][:80]}...")
    
    return {
        'total_failures': len(failures),
        'rave_critical': len(rave_critical),
        'rave_useful': len(rave_useful),
        'non_critical': len(non_critical),
        'priority_critical': priority_critical,
        'all_failures': failures
    }

if __name__ == "__main__":
    results = analyze_failures()
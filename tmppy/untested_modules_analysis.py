#!/usr/bin/env python3
"""
Comprehensive analysis of untested modules in the codebase.
Finds all modules that exist but are not covered by current test configurations.
"""

import json
import os
import glob
from pathlib import Path
import ast
import re

def find_all_module_files():
    """Find all Python module files in the codebase."""
    module_patterns = [
        "modules/**/*.py",
        "crossfade/**/*.py", 
        "audio_analysis/**/*.py",
        "audio-ml-extensions/**/*.py",
        "advanced_modules/**/*.py",
        "data_pipelines/**/*.py"
    ]
    
    all_files = []
    for pattern in module_patterns:
        files = glob.glob(pattern, recursive=True)
        # Filter out __init__.py, __pycache__, and temp files
        files = [f for f in files if not f.endswith('__init__.py') 
                 and '__pycache__' not in f 
                 and not f.endswith('.temp')]
        all_files.extend(files)
    
    return sorted(all_files)

def extract_classes_from_file(file_path):
    """Extract class names from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return classes
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def load_test_configs():
    """Load all test configuration files to see what's currently tested."""
    config_files = [
        "standardized_tests/comprehensive_modules_standardized.json",
        "ultimate_working_modules_config.json",
        "final_breakthrough_validation_report.json"
    ]
    
    tested_modules = set()
    tested_classes = set()
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different config formats
                if 'modules' in data:
                    for module in data['modules']:
                        if 'module_name' in module:
                            tested_modules.add(module['module_name'])
                        if 'class_name' in module:
                            tested_classes.add(module['class_name'])
                
                # Handle breakthrough validation report format
                if 'comprehensive_module_coverage' in data:
                    coverage = data['comprehensive_module_coverage']
                    if 'module_categories' in coverage:
                        for category, info in coverage['module_categories'].items():
                            if 'examples' in info:
                                tested_modules.update(info['examples'])
                
            except Exception as e:
                print(f"Error loading {config_file}: {e}")
    
    return tested_modules, tested_classes

def analyze_module_coverage():
    """Main analysis function."""
    print("🔍 Analyzing module coverage...")
    
    # Find all module files
    all_files = find_all_module_files()
    print(f"📁 Found {len(all_files)} module files")
    
    # Load test configurations
    tested_modules, tested_classes = load_test_configs()
    print(f"✅ Found {len(tested_modules)} tested modules and {len(tested_classes)} tested classes")
    
    # Analyze each file
    untested_modules = []
    
    for file_path in all_files:
        # Get module name from file path
        module_name = Path(file_path).stem
        
        # Extract classes from file
        classes = extract_classes_from_file(file_path)
        
        # Check if module or any of its classes are tested
        module_tested = (
            module_name in tested_modules or
            any(cls in tested_classes for cls in classes)
        )
        
        if not module_tested and classes:  # Only include files with classes
            untested_modules.append({
                'file_path': file_path,
                'module_name': module_name,
                'classes': classes,
                'description': get_module_description(file_path, classes)
            })
    
    return untested_modules

def get_module_description(file_path, classes):
    """Generate a description based on file path and class names."""
    path_parts = file_path.split('/')
    
    descriptions = {
        'audio-ml-extensions': 'Audio machine learning extension',
        'orchestration': 'ML orchestration and workflow',
        'audio_gan': 'Generative adversarial network for audio',
        'crossfade': 'Audio crossfading and transition',
        'audio_analysis': 'Audio analysis and feature extraction',
        'advanced_modules': 'Advanced audio processing',
        'modules': 'Core neural network module'
    }
    
    # Try to match path components to descriptions
    for part in path_parts:
        if part in descriptions:
            desc_base = descriptions[part]
            break
    else:
        desc_base = "Audio processing module"
    
    # Add more specific description based on class names
    if classes:
        first_class = classes[0]
        if 'GAN' in first_class or 'Discriminator' in first_class:
            return f"{desc_base} - GAN-based audio generation"
        elif 'Attention' in first_class:
            return f"{desc_base} - Attention mechanism"
        elif 'Norm' in first_class:
            return f"{desc_base} - Normalization layers"
        elif 'Conv' in first_class:
            return f"{desc_base} - Convolutional operations"
        elif 'Encoder' in first_class:
            return f"{desc_base} - Feature encoding"
        elif 'Generator' in first_class:
            return f"{desc_base} - Signal generation"
        elif 'Optimizer' in first_class:
            return f"{desc_base} - Parameter optimization"
        elif 'Profiler' in first_class:
            return f"{desc_base} - Performance profiling"
    
    return desc_base

def main():
    """Main function."""
    print("🎵 Untested Module Analysis")
    print("=" * 50)
    
    untested = analyze_module_coverage()
    
    if not untested:
        print("🎉 All modules with classes are covered by tests!")
        return
    
    print(f"\n🚨 Found {len(untested)} untested modules:\n")
    
    # Group by directory for better organization
    by_directory = {}
    for module in untested:
        dir_path = '/'.join(module['file_path'].split('/')[:-1])
        if dir_path not in by_directory:
            by_directory[dir_path] = []
        by_directory[dir_path].append(module)
    
    for directory, modules in sorted(by_directory.items()):
        print(f"📂 {directory}/")
        for module in sorted(modules, key=lambda x: x['module_name']):
            print(f"  ❌ {module['module_name']}.py")
            print(f"     🏷️  Classes: {', '.join(module['classes'])}")
            print(f"     📝 {module['description']}")
            print(f"     📍 {module['file_path']}")
            print()
    
    print(f"\n📊 Summary:")
    print(f"   Total untested modules: {len(untested)}")
    print(f"   Total untested classes: {sum(len(m['classes']) for m in untested)}")
    
    # Generate recommendations
    print(f"\n💡 Recommendations:")
    print(f"   1. Prioritize testing modules in core paths (modules/, crossfade/)")
    print(f"   2. Focus on modules with multiple classes for maximum coverage")
    print(f"   3. Consider grouping related modules for efficient testing")

if __name__ == "__main__":
    main()
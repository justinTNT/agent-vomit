#!/usr/bin/env python3
"""
VALIDATE RAVE-CRITICAL MODULES: Focus on production-quality validation of core RAVE components
Strategy: Use signature introspection to properly test the 13 failing modules
"""

import torch
import sys
import importlib.util
from pathlib import Path
import json
from robust_module_validator import RobustModuleValidator

def load_failing_modules():
    """Load the list of failing modules from comprehensive results"""
    with open('comprehensive_coverage_results.json') as f:
        results = json.load(f)
    
    failing_modules = []
    for result in results['all_results']:
        if not result['result']['module_passed']:
            failing_modules.append({
                'module_file': result['module_file'],
                'class_name': result['class_name'],
                'directory': result['directory'],
                'error': result['result'].get('init_error', 'Unknown error')
            })
    
    return failing_modules

def import_module_safely(module_path: str):
    """Safely import a module from file path"""
    try:
        full_path = Path(f"/Users/jtnt/Play/agent-vomit/{module_path}.py")
        if not full_path.exists():
            return None
        
        # Add module directory to path
        module_dir = full_path.parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        
        spec = importlib.util.spec_from_file_location(
            full_path.stem,
            str(full_path)
        )
        
        if spec is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module
        
    except Exception as e:
        print(f"Failed to import {module_path}: {str(e)}")
        return None

def categorize_failures_by_importance(failing_modules):
    """Categorize failing modules by RAVE importance"""
    
    rave_core = []  # Essential for RAVE
    rave_useful = []  # Useful but not essential
    rave_optional = []  # Nice to have
    
    core_keywords = ['autoencoder', 'vae', 'conv', 'quantiz', 'stft', 'loss', 'discriminator']
    useful_keywords = ['transformer', 'attention', 'wavenet', 'audio']
    
    for module in failing_modules:
        module_name = module['module_file'].lower()
        class_name = module['class_name'].lower()
        
        if any(kw in module_name or kw in class_name for kw in core_keywords):
            rave_core.append(module)
        elif any(kw in module_name or kw in class_name for kw in useful_keywords):
            rave_useful.append(module)
        else:
            rave_optional.append(module)
    
    return rave_core, rave_useful, rave_optional

def main():
    """Robustly validate the failing modules with focus on RAVE components"""
    
    print("🔬 ROBUST VALIDATION OF FAILING MODULES")
    print("=" * 70)
    print("Strategy: Production-quality testing with signature introspection")
    print("Focus: Core RAVE components for audio generation")
    print()
    
    # Load failing modules
    failing_modules = load_failing_modules()
    print(f"Total failing modules: {len(failing_modules)}")
    
    # Categorize by RAVE importance
    rave_core, rave_useful, rave_optional = categorize_failures_by_importance(failing_modules)
    
    print(f"RAVE-critical modules: {len(rave_core)}")
    print(f"RAVE-useful modules: {len(rave_useful)}")
    print(f"RAVE-optional modules: {len(rave_optional)}")
    print()
    
    # Initialize robust validator
    validator = RobustModuleValidator()
    
    # Results tracking
    validation_results = {
        'rave_core': [],
        'rave_useful': [],
        'rave_optional': [],
        'summary': {}
    }
    
    # Validate RAVE-core modules first
    print("🎯 VALIDATING RAVE-CORE MODULES:")
    print("-" * 50)
    
    for category_name, modules in [('rave_core', rave_core), ('rave_useful', rave_useful), ('rave_optional', rave_optional)]:
        print(f"\n📋 {category_name.upper()} MODULES ({len(modules)} modules):")
        
        for module_info in modules:
            module_path = module_info['module_file']
            class_name = module_info['class_name']
            
            print(f"\n🔍 {class_name} ({module_path})")
            print(f"   Previous error: {module_info['error']}")
            
            # Import module
            module = import_module_safely(module_path)
            if module is None:
                result = {
                    'module_file': module_path,
                    'class_name': class_name,
                    'status': 'import_failed',
                    'error': 'Could not import module'
                }
                validation_results[category_name].append(result)
                print("   ❌ Could not import module")
                continue
            
            # Get the class
            try:
                module_class = getattr(module, class_name)
            except AttributeError:
                result = {
                    'module_file': module_path,
                    'class_name': class_name,
                    'status': 'class_not_found',
                    'error': f'Class {class_name} not found in module'
                }
                validation_results[category_name].append(result)
                print(f"   ❌ Class {class_name} not found")
                continue
            
            # Robustly validate
            result = validator.validate_module_robustly(module_class, module_path)
            validation_results[category_name].append(result)
            
            # Report result
            if result['status'] == 'success':
                print("   ✅ PASSED - Module validated successfully")
                print(f"      Signature: {result['signature']}")
                test_summary = [f"Test {r['test_id']}: {r['status']}" for r in result['test_results']]
                print(f"      Tests: {', '.join(test_summary)}")
            else:
                print(f"   ❌ FAILED - {result['status']}")
                print(f"      Error: {result.get('error', 'Unknown error')}")
                if 'signature' in result:
                    print(f"      Signature: {result['signature']}")
    
    # Generate summary
    print("\n" + "=" * 70)
    print("🎯 ROBUST VALIDATION SUMMARY")
    print("=" * 70)
    
    for category_name in ['rave_core', 'rave_useful', 'rave_optional']:
        results = validation_results[category_name]
        if not results:
            continue
            
        total = len(results)
        successful = len([r for r in results if r['status'] == 'success'])
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"{category_name.upper()}: {successful}/{total} ({success_rate:.1f}% success)")
        
        # Show specific failures for core modules
        if category_name == 'rave_core':
            failures = [r for r in results if r['status'] != 'success']
            if failures:
                print("  Critical failures:")
                for failure in failures:
                    print(f"    - {failure['class_name']}: {failure.get('error', 'Unknown error')}")
        
        validation_results['summary'][category_name] = {
            'total': total,
            'successful': successful,
            'success_rate': success_rate
        }
    
    # Overall assessment
    core_success_rate = validation_results['summary'].get('rave_core', {}).get('success_rate', 0)
    
    print(f"\n📊 RAVE READINESS ASSESSMENT:")
    if core_success_rate >= 90:
        print("🎉 EXCELLENT: 90%+ of core RAVE modules validated")
        print("✅ Ready for production RAVE implementation")
        readiness = "READY"
    elif core_success_rate >= 75:
        print("🔥 GOOD: 75%+ of core RAVE modules validated")
        print("🔧 Minor fixes needed for production readiness")
        readiness = "NEARLY_READY"
    else:
        print("⚠️  MORE WORK NEEDED: Core RAVE modules need attention")
        print("🛠️  Focus on fixing critical component failures")
        readiness = "NEEDS_WORK"
    
    validation_results['summary']['overall_assessment'] = readiness
    validation_results['summary']['core_success_rate'] = core_success_rate
    
    # Save detailed results
    with open('robust_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: robust_validation_results.json")
    print("\n🎯 ROBUST VALIDATION COMPLETE")
    print("Focus: Production-quality RAVE component testing")

if __name__ == "__main__":
    main()
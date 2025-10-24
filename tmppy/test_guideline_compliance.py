#!/usr/bin/env python3
"""Test which modules comply with guidelines_v3.md requirements."""

import ast
import inspect
from pathlib import Path

def check_module_compliance(module_path):
    """Check if a module follows guideline requirements."""
    results = {
        'has_kwargs': False,
        'extends_nn_module': False,
        'has_forward': False,
        'parameter_types_annotated': False,
        'docstrings': False
    }
    
    with open(module_path, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except:
        return results
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if extends nn.Module
            for base in node.bases:
                if isinstance(base, ast.Attribute) and base.attr == 'Module':
                    results['extends_nn_module'] = True
            
            # Check for docstring
            if ast.get_docstring(node):
                results['docstrings'] = True
            
            # Check methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == '__init__':
                        # Check for **kwargs
                        if item.args.kwarg is not None:
                            results['has_kwargs'] = True
                        
                        # Check for type annotations
                        if any(arg.annotation for arg in item.args.args):
                            results['parameter_types_annotated'] = True
                    
                    elif item.name == 'forward':
                        results['has_forward'] = True
    
    return results


def main():
    print("🔍 MODULE GUIDELINE COMPLIANCE CHECK")
    print("=" * 60 + "\n")
    
    modules_dir = Path('modules')
    compliance_stats = {
        'total': 0,
        'has_kwargs': 0,
        'extends_nn_module': 0,
        'has_forward': 0,
        'parameter_types_annotated': 0,
        'docstrings': 0
    }
    
    non_compliant = []
    
    for module_file in sorted(modules_dir.glob('*.py')):
        if module_file.name == '__init__.py':
            continue
            
        compliance_stats['total'] += 1
        results = check_module_compliance(module_file)
        
        # Update stats
        for key, value in results.items():
            if value:
                compliance_stats[key] += 1
        
        # Check full compliance
        fully_compliant = all([
            results['has_kwargs'],
            results['extends_nn_module'],
            results['has_forward']
        ])
        
        status = "✅" if fully_compliant else "❌"
        print(f"{status} {module_file.stem:30}", end="")
        
        issues = []
        if not results['has_kwargs']:
            issues.append("no **kwargs")
        if not results['extends_nn_module']:
            issues.append("not nn.Module")
        if not results['has_forward']:
            issues.append("no forward()")
        
        if issues:
            print(f" Issues: {', '.join(issues)}")
            non_compliant.append(module_file.stem)
        else:
            print(" Fully compliant")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPLIANCE SUMMARY")
    print("=" * 60)
    print(f"Total modules: {compliance_stats['total']}")
    print(f"Accept **kwargs: {compliance_stats['has_kwargs']} ({compliance_stats['has_kwargs']/compliance_stats['total']*100:.1f}%)")
    print(f"Extend nn.Module: {compliance_stats['extends_nn_module']} ({compliance_stats['extends_nn_module']/compliance_stats['total']*100:.1f}%)")
    print(f"Have forward(): {compliance_stats['has_forward']} ({compliance_stats['has_forward']/compliance_stats['total']*100:.1f}%)")
    print(f"Type annotations: {compliance_stats['parameter_types_annotated']} ({compliance_stats['parameter_types_annotated']/compliance_stats['total']*100:.1f}%)")
    print(f"Have docstrings: {compliance_stats['docstrings']} ({compliance_stats['docstrings']/compliance_stats['total']*100:.1f}%)")
    
    print(f"\nFully compliant: {compliance_stats['total'] - len(non_compliant)} ({(compliance_stats['total'] - len(non_compliant))/compliance_stats['total']*100:.1f}%)")
    
    if non_compliant:
        print(f"\nNon-compliant modules ({len(non_compliant)}):")
        for module in non_compliant[:10]:  # Show first 10
            print(f"  - {module}")
        if len(non_compliant) > 10:
            print(f"  ... and {len(non_compliant) - 10} more")


if __name__ == "__main__":
    main()
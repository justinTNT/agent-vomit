#!/usr/bin/env python3
"""
Analyze the original return format patterns across all modules.
This script identifies what the original return format patterns were BEFORE any changes.
"""

import os
import re
import ast
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

def extract_forward_returns(module_content: str, module_name: str) -> List[Dict]:
    """
    Extract return statements from forward() methods in a module.
    Returns list of return pattern info dicts.
    """
    returns = []
    
    try:
        tree = ast.parse(module_content)
    except SyntaxError:
        print(f"Warning: Could not parse {module_name}")
        return returns
    
    class ForwardMethodVisitor(ast.NodeVisitor):
        def __init__(self):
            self.in_forward_method = False
            self.current_class = None
            self.current_method = None
            
        def visit_ClassDef(self, node):
            old_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = old_class
            
        def visit_FunctionDef(self, node):
            old_method = self.current_method
            old_in_forward = self.in_forward_method
            
            self.current_method = node.name
            if node.name == 'forward':
                self.in_forward_method = True
            
            self.generic_visit(node)
            
            self.current_method = old_method
            self.in_forward_method = old_in_forward
            
        def visit_Return(self, node):
            if self.in_forward_method and self.current_class and node.value:
                return_info = {
                    'module': module_name,
                    'class': self.current_class,
                    'return_ast_type': type(node.value).__name__,
                    'is_dict': False,
                    'is_tensor': False,
                    'is_tuple': False,
                    'dict_keys': [],
                    'tuple_elements': 0,
                    'raw_return': ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                }
                
                # Analyze return type
                if isinstance(node.value, ast.Dict):
                    return_info['is_dict'] = True
                    if node.value.keys:
                        return_info['dict_keys'] = [
                            ast.unparse(key) if hasattr(ast, 'unparse') else str(key) 
                            for key in node.value.keys if key is not None
                        ]
                elif isinstance(node.value, ast.Tuple):
                    return_info['is_tuple'] = True
                    return_info['tuple_elements'] = len(node.value.elts)
                elif isinstance(node.value, ast.Name):
                    # Single variable return - could be tensor
                    return_info['is_tensor'] = True
                elif isinstance(node.value, ast.Call):
                    # Function call return - could be tensor
                    return_info['is_tensor'] = True
                    
                returns.append(return_info)
    
    visitor = ForwardMethodVisitor()
    visitor.visit(tree)
    return returns

def analyze_patterns(returns_data: List[Dict]) -> Dict:
    """Analyze the return patterns and generate statistics."""
    
    total_returns = len(returns_data)
    dict_returns = [r for r in returns_data if r['is_dict']]
    tensor_returns = [r for r in returns_data if r['is_tensor'] and not r['is_dict']]
    tuple_returns = [r for r in returns_data if r['is_tuple']]
    
    # Count common dict keys
    all_dict_keys = []
    for r in dict_returns:
        all_dict_keys.extend([k.strip("'\"") for k in r['dict_keys']])
    
    dict_key_counts = Counter(all_dict_keys)
    
    # Group by module
    by_module = defaultdict(list)
    for r in returns_data:
        by_module[r['module']].append(r)
    
    # Group by class
    by_class = defaultdict(list)
    for r in returns_data:
        by_class[f"{r['module']}.{r['class']}"].append(r)
    
    return {
        'total_returns': total_returns,
        'dict_returns': len(dict_returns),
        'tensor_returns': len(tensor_returns),
        'tuple_returns': len(tuple_returns),
        'dict_percentage': (len(dict_returns) / total_returns * 100) if total_returns > 0 else 0,
        'tensor_percentage': (len(tensor_returns) / total_returns * 100) if total_returns > 0 else 0,
        'tuple_percentage': (len(tuple_returns) / total_returns * 100) if total_returns > 0 else 0,
        'common_dict_keys': dict_key_counts.most_common(10),
        'by_module': dict(by_module),
        'by_class': dict(by_class),
        'all_returns': returns_data
    }

def main():
    modules_dir = "/Users/jtnt/Play/agent-vomit/modules"
    all_returns = []
    
    print("🔍 ANALYZING ORIGINAL RETURN FORMAT PATTERNS")
    print("=" * 60)
    
    # Process all Python files in modules directory
    for filename in sorted(os.listdir(modules_dir)):
        if filename.endswith('.py') and filename != '__init__.py' and not filename.endswith('.temp'):
            filepath = os.path.join(modules_dir, filename)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            module_returns = extract_forward_returns(content, filename[:-3])
            all_returns.extend(module_returns)
            
            if module_returns:
                print(f"\n📁 {filename}")
                for ret in module_returns:
                    pattern = "DICT" if ret['is_dict'] else "TENSOR" if ret['is_tensor'] else "TUPLE" if ret['is_tuple'] else "OTHER"
                    keys_info = f" keys={ret['dict_keys']}" if ret['is_dict'] and ret['dict_keys'] else ""
                    print(f"  └─ {ret['class']}.forward() → {pattern}{keys_info}")
    
    print(f"\n\n📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    analysis = analyze_patterns(all_returns)
    
    print(f"Total forward() return statements analyzed: {analysis['total_returns']}")
    print(f"\nReturn Pattern Breakdown:")
    print(f"  📚 Dictionary returns: {analysis['dict_returns']} ({analysis['dict_percentage']:.1f}%)")
    print(f"  📦 Direct tensor returns: {analysis['tensor_returns']} ({analysis['tensor_percentage']:.1f}%)")
    print(f"  📝 Tuple returns: {analysis['tuple_returns']} ({analysis['tuple_percentage']:.1f}%)")
    
    if analysis['common_dict_keys']:
        print(f"\nMost Common Dictionary Keys:")
        for key, count in analysis['common_dict_keys']:
            print(f"  '{key}': {count} occurrences")
    
    print(f"\n\n📋 DETAILED MODULE BREAKDOWN")
    print("=" * 60)
    
    for module, returns in sorted(analysis['by_module'].items()):
        dict_count = len([r for r in returns if r['is_dict']])
        tensor_count = len([r for r in returns if r['is_tensor'] and not r['is_dict']])
        tuple_count = len([r for r in returns if r['is_tuple']])
        
        print(f"\n{module}.py ({len(returns)} classes):")
        print(f"  Dict: {dict_count}, Tensor: {tensor_count}, Tuple: {tuple_count}")
        
        for ret in returns:
            pattern = "DICT" if ret['is_dict'] else "TENSOR" if ret['is_tensor'] else "TUPLE" if ret['is_tuple'] else "OTHER"
            if ret['is_dict'] and ret['dict_keys']:
                pattern += f"({', '.join(ret['dict_keys'])})"
            print(f"    {ret['class']} → {pattern}")
    
    print(f"\n\n🎯 CONCLUSION")
    print("=" * 60)
    
    if analysis['dict_percentage'] > 50:
        print("✅ MAJORITY PATTERN: Dictionary returns are the ORIGINAL dominant pattern")
        print(f"   {analysis['dict_returns']}/{analysis['total_returns']} classes use dictionary returns")
        print("   → Dictionary pattern appears to be the natural/preferred style")
    elif analysis['tensor_percentage'] > 50:
        print("✅ MAJORITY PATTERN: Direct tensor returns are the ORIGINAL dominant pattern")
        print(f"   {analysis['tensor_returns']}/{analysis['total_returns']} classes use direct tensor returns")
        print("   → Direct tensor pattern appears to be the natural/preferred style")
    else:
        print("🤔 MIXED PATTERN: No clear dominant pattern in original code")
        print("   → Both approaches were used somewhat equally")
    
    # Identify any artificial standardization
    if analysis['dict_percentage'] > 40 and analysis['tensor_percentage'] > 40:
        print("\n⚠️  POTENTIAL ARTIFICIAL STANDARDIZATION DETECTED")
        print("    The patterns suggest this codebase may have had mixed approaches")
        print("    rather than a single consistent standard that should be preserved.")
    
    print(f"\nRecommendation:")
    if analysis['dict_percentage'] >= 60:
        print("📖 Preserve/restore dictionary return pattern as the original standard")
    elif analysis['tensor_percentage'] >= 60:
        print("📦 Preserve/restore direct tensor return pattern as the original standard")
    else:
        print("🔀 Accept that mixed patterns were the original state")
        print("   Consider standardizing based on functionality rather than enforcing uniformity")

if __name__ == "__main__":
    main()
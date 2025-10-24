#!/usr/bin/env python3
"""
Comprehensive Module Builder for Universal Test Framework

Systematically discovers all available modules and builds optimal test configurations
for maximum coverage and success rate.
"""

import os
import ast
import importlib.util
import json
import torch
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModuleInfo:
    """Information about a discovered module"""
    module_path: str
    module_name: str
    class_name: str
    init_signature: Dict[str, Any]
    forward_signature: Optional[Dict[str, Any]]
    methods: List[str]
    complexity: str  # 'simple', 'medium', 'complex'
    category: str    # 'core', 'audio', 'data', 'advanced'

class ComprehensiveModuleBuilder:
    """Discovers and analyzes all available modules for test generation"""
    
    def __init__(self, modules_dir: str = "modules"):
        self.modules_dir = Path(modules_dir)
        self.discovered_modules = []
        self.working_modules = []
        self.problematic_modules = []
        
    def discover_all_modules(self) -> List[ModuleInfo]:
        """Discover all modules and their classes"""
        print("🔍 Discovering all available modules...")
        
        # Find all Python files
        module_files = []
        for root, dirs, files in os.walk(self.modules_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    module_files.append(Path(root) / file)
        
        print(f"Found {len(module_files)} module files")
        
        # Analyze each module
        for module_file in module_files:
            try:
                module_info = self._analyze_module_file(module_file)
                if module_info:
                    self.discovered_modules.extend(module_info)
            except Exception as e:
                print(f"⚠️  Failed to analyze {module_file}: {e}")
        
        print(f"✅ Discovered {len(self.discovered_modules)} module classes")
        return self.discovered_modules
    
    def _analyze_module_file(self, module_file: Path) -> List[ModuleInfo]:
        """Analyze a single module file for classes"""
        module_infos = []
        
        # Read and parse the file
        try:
            with open(module_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception as e:
            return []
        
        # Find all class definitions that inherit from nn.Module
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it inherits from nn.Module
                if self._inherits_from_nn_module(node):
                    try:
                        module_info = self._create_module_info(module_file, node)
                        if module_info:
                            module_infos.append(module_info)
                    except Exception as e:
                        print(f"⚠️  Failed to analyze class {node.name} in {module_file}: {e}")
        
        return module_infos
    
    def _inherits_from_nn_module(self, class_node: ast.ClassDef) -> bool:
        """Check if class inherits from nn.Module"""
        for base in class_node.bases:
            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                if base.value.id == 'nn' and base.attr == 'Module':
                    return True
            elif isinstance(base, ast.Name) and base.id in ['Module', 'nn.Module']:
                return True
        return False
    
    def _create_module_info(self, module_file: Path, class_node: ast.ClassDef) -> Optional[ModuleInfo]:
        """Create module info by importing and inspecting the class"""
        
        # Build module path
        relative_path = module_file.relative_to(self.modules_dir)
        module_path_parts = list(relative_path.with_suffix('').parts)
        module_name = '.'.join(module_path_parts)
        
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the class
            cls = getattr(module, class_node.name)
            
            # Analyze signatures
            init_sig = self._analyze_init_signature(cls)
            forward_sig = self._analyze_forward_signature(cls)
            methods = self._get_public_methods(cls)
            
            # Categorize
            complexity = self._assess_complexity(init_sig, forward_sig, methods)
            category = self._categorize_module(module_name, class_node.name)
            
            return ModuleInfo(
                module_path=str(module_file),
                module_name=module_name,
                class_name=class_node.name,
                init_signature=init_sig,
                forward_signature=forward_sig,
                methods=methods,
                complexity=complexity,
                category=category
            )
            
        except Exception as e:
            print(f"⚠️  Could not import {module_name}.{class_node.name}: {e}")
            return None
    
    def _analyze_init_signature(self, cls) -> Dict[str, Any]:
        """Analyze __init__ method signature"""
        try:
            sig = inspect.signature(cls.__init__)
            params = {}
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                params[name] = {
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
            return params
        except Exception:
            return {}
    
    def _analyze_forward_signature(self, cls) -> Optional[Dict[str, Any]]:
        """Analyze forward method signature"""
        try:
            if hasattr(cls, 'forward'):
                sig = inspect.signature(cls.forward)
                params = {}
                for name, param in sig.parameters.items():
                    if name == 'self':
                        continue
                    params[name] = {
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                        'required': param.default == inspect.Parameter.empty
                    }
                return params
        except Exception:
            pass
        return None
    
    def _get_public_methods(self, cls) -> List[str]:
        """Get all public methods"""
        methods = []
        for name in dir(cls):
            if not name.startswith('_') and callable(getattr(cls, name)):
                methods.append(name)
        return methods
    
    def _assess_complexity(self, init_sig: Dict, forward_sig: Optional[Dict], methods: List[str]) -> str:
        """Assess module complexity"""
        required_params = sum(1 for p in init_sig.values() if p['required'])
        forward_params = len(forward_sig) if forward_sig else 0
        
        if required_params <= 2 and forward_params <= 1 and len(methods) <= 5:
            return 'simple'
        elif required_params <= 5 and forward_params <= 3 and len(methods) <= 10:
            return 'medium'
        else:
            return 'complex'
    
    def _categorize_module(self, module_name: str, class_name: str) -> str:
        """Categorize module by type"""
        name_lower = f"{module_name}.{class_name}".lower()
        
        if any(x in name_lower for x in ['audio', 'stft', 'mel', 'spectral', 'beat', 'music']):
            return 'audio'
        elif any(x in name_lower for x in ['data', 'stream', 'version', 'store', 'sample']):
            return 'data'
        elif any(x in name_lower for x in ['transformer', 'attention', 'conv', 'activation', 'quantizer']):
            return 'core'
        else:
            return 'advanced'
    
    def prioritize_modules(self) -> Tuple[List[ModuleInfo], List[ModuleInfo], List[ModuleInfo]]:
        """Prioritize modules by likelihood of success"""
        
        simple_modules = [m for m in self.discovered_modules if m.complexity == 'simple']
        medium_modules = [m for m in self.discovered_modules if m.complexity == 'medium']
        complex_modules = [m for m in self.discovered_modules if m.complexity == 'complex']
        
        # Sort by category priority
        category_priority = {'core': 0, 'audio': 1, 'data': 2, 'advanced': 3}
        
        simple_modules.sort(key=lambda x: category_priority.get(x.category, 4))
        medium_modules.sort(key=lambda x: category_priority.get(x.category, 4))
        complex_modules.sort(key=lambda x: category_priority.get(x.category, 4))
        
        return simple_modules, medium_modules, complex_modules
    
    def generate_test_configs(self, modules: List[ModuleInfo], target_count: int = 50) -> Dict[str, Any]:
        """Generate comprehensive test configurations"""
        
        configs = []
        count = 0
        
        for module_info in modules:
            if count >= target_count:
                break
                
            config = self._generate_single_test_config(module_info)
            if config:
                configs.append(config)
                count += 1
        
        return {
            "modules": configs,
            "metadata": {
                "total_modules": len(configs),
                "generation_strategy": "comprehensive_discovery",
                "priority_order": "simple->medium->complex, core->audio->data->advanced"
            }
        }
    
    def _generate_single_test_config(self, module_info: ModuleInfo) -> Optional[Dict[str, Any]]:
        """Generate test config for a single module"""
        
        try:
            # Generate init params
            init_params = self._generate_init_params(module_info)
            
            # Generate test cases
            tests = self._generate_test_cases(module_info)
            
            config = {
                "module_name": module_info.module_name,
                "class_name": module_info.class_name,
                "class_variations": [module_info.class_name],
                "init_params": init_params,
                "tests": tests,
                "description": f"Auto-generated config for {module_info.module_name}.{module_info.class_name}",
                "complexity": module_info.complexity,
                "category": module_info.category
            }
            
            return config
            
        except Exception as e:
            print(f"⚠️  Failed to generate config for {module_info.module_name}.{module_info.class_name}: {e}")
            return None
    
    def _generate_init_params(self, module_info: ModuleInfo) -> Dict[str, Any]:
        """Generate sensible init parameters"""
        params = {}
        
        for param_name, param_info in module_info.init_signature.items():
            if param_info['default'] is not None:
                continue  # Skip optional params with defaults
                
            # Generate based on parameter name patterns
            param_lower = param_name.lower()
            
            if 'dim' in param_lower or 'size' in param_lower:
                if 'hidden' in param_lower or 'embed' in param_lower:
                    params[param_name] = 512
                elif 'input' in param_lower or 'in_' in param_lower:
                    params[param_name] = 512
                elif 'output' in param_lower or 'out_' in param_lower:
                    params[param_name] = 256
                else:
                    params[param_name] = 256
                    
            elif 'channel' in param_lower:
                if 'in_' in param_lower or 'input' in param_lower:
                    params[param_name] = 3
                else:
                    params[param_name] = 64
                    
            elif 'head' in param_lower:
                params[param_name] = 8
                
            elif 'layer' in param_lower:
                params[param_name] = 6
                
            elif 'vocab' in param_lower:
                params[param_name] = 1000
                
            elif 'dropout' in param_lower:
                params[param_name] = 0.1
                
            elif 'rate' in param_lower and 'sample' in param_lower:
                params[param_name] = 44100
                
            elif 'length' in param_lower or 'len' in param_lower:
                params[param_name] = 512
                
            elif param_lower in ['encoder', 'decoder']:
                params[param_name] = None  # Will be handled by parameter variations
                
            else:
                # Default sensible values
                if param_info['annotation'] and 'int' in param_info['annotation']:
                    params[param_name] = 256
                elif param_info['annotation'] and 'float' in param_info['annotation']:
                    params[param_name] = 0.1
                elif param_info['annotation'] and 'bool' in param_info['annotation']:
                    params[param_name] = True
                else:
                    params[param_name] = 256
        
        return params
    
    def _generate_test_cases(self, module_info: ModuleInfo) -> List[Dict[str, Any]]:
        """Generate test cases for a module"""
        tests = []
        
        # Always include forward test
        tests.append({
            "name": "forward",
            "test_type": "forward",
            "input_tensor": self._generate_input_tensor(module_info),
            "tolerance": 0.0001,
            "timeout": 10.0,
            "description": f"forward test for {module_info.class_name}.forward"
        })
        
        # Shape test
        tests.append({
            "name": "shape_preserve",
            "test_type": "shape",
            "input_tensor": self._generate_input_tensor(module_info),
            "tolerance": 0.0001,
            "timeout": 10.0,
            "description": f"shape test for {module_info.class_name}"
        })
        
        # Gradient test for simple/medium modules
        if module_info.complexity in ['simple', 'medium']:
            tests.append({
                "name": "gradient",
                "test_type": "gradient", 
                "input_tensor": self._generate_input_tensor(module_info, requires_grad=True),
                "tolerance": 0.0001,
                "timeout": 10.0,
                "description": f"gradient test for {module_info.class_name}"
            })
        
        return tests
    
    def _generate_input_tensor(self, module_info: ModuleInfo, requires_grad: bool = False) -> Dict[str, Any]:
        """Generate appropriate input tensor configuration"""
        
        # Analyze forward signature to determine tensor shape
        if module_info.forward_signature:
            first_param = next(iter(module_info.forward_signature.values()), None)
            if first_param and first_param['annotation']:
                annotation = first_param['annotation'].lower()
                if 'tensor' in annotation:
                    # Generate based on module category and init params
                    shape = self._infer_tensor_shape(module_info)
                    dtype = "torch.float32"
                    
                    # Some modules expect long tensors (embedding, tokenization)
                    if any(x in f"{module_info.module_name}.{module_info.class_name}".lower() 
                           for x in ['embed', 'vocab', 'token', 'sequence']):
                        dtype = "torch.long"
                        requires_grad = False  # Long tensors don't have gradients
                    
                    return {
                        "type": "tensor",
                        "shape": shape,
                        "dtype": dtype,
                        "requires_grad": requires_grad,
                        "device": "cpu"
                    }
        
        # Default tensor configuration
        return {
            "type": "tensor",
            "shape": [2, 512],
            "dtype": "torch.float32",
            "requires_grad": requires_grad,
            "device": "cpu"
        }
    
    def _infer_tensor_shape(self, module_info: ModuleInfo) -> List[int]:
        """Infer appropriate tensor shape from module info"""
        
        init_params = module_info.init_signature
        category = module_info.category
        
        # Audio modules typically work with 1D or 2D signals
        if category == 'audio':
            sample_rate = 44100
            duration = 1.0  # 1 second
            if any('sample_rate' in p for p in init_params):
                return [2, int(sample_rate * duration)]  # [batch, samples]
            else:
                return [2, 1024]  # Default audio frame size
        
        # Transformer/attention modules
        if any(x in module_info.class_name.lower() for x in ['transformer', 'attention']):
            seq_len = 10
            d_model = init_params.get('d_model', {}).get('default', 512)
            if isinstance(d_model, dict):
                d_model = 512
            return [2, seq_len, d_model]
        
        # Convolutional modules
        if 'conv' in module_info.class_name.lower():
            in_channels = 3
            for param_name, param_info in init_params.items():
                if 'in_channel' in param_name.lower():
                    default_val = param_info.get('default')
                    if isinstance(default_val, int):
                        in_channels = default_val
                    break
            return [2, in_channels, 256]  # [batch, channels, length/width]
        
        # Sequence modules
        if any(x in module_info.class_name.lower() for x in ['sequence', 'lstm', 'gru', 'rnn']):
            return [2, 10, 512]  # [batch, seq_len, feature_dim]
        
        # Default shape
        return [2, 512]

def main():
    """Main function to build comprehensive test configuration"""
    
    builder = ComprehensiveModuleBuilder()
    
    # Discover all modules
    modules = builder.discover_all_modules()
    print(f"\\n📊 Module Discovery Summary:")
    print(f"Total modules discovered: {len(modules)}")
    
    # Show breakdown by category and complexity
    categories = {}
    complexities = {}
    
    for module in modules:
        categories[module.category] = categories.get(module.category, 0) + 1
        complexities[module.complexity] = complexities.get(module.complexity, 0) + 1
    
    print(f"\\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print(f"\\nBy complexity:")
    for comp, count in sorted(complexities.items()):
        print(f"  {comp}: {count}")
    
    # Prioritize modules
    simple, medium, complex = builder.prioritize_modules()
    
    print(f"\\n🎯 Prioritization:")
    print(f"Simple modules: {len(simple)}")
    print(f"Medium modules: {len(medium)}")
    print(f"Complex modules: {len(complex)}")
    
    # Generate comprehensive test configs
    print(f"\\n🏗️  Generating comprehensive test configurations...")
    
    # Start with simple, add medium, then some complex
    all_modules = simple + medium + complex[:10]  # Limit complex modules
    
    config = builder.generate_test_configs(all_modules, target_count=60)
    
    # Save configuration
    output_file = "comprehensive_test_config.json"
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Generated comprehensive test configuration with {len(config['modules'])} modules")
    print(f"📁 Saved to: {output_file}")
    
    # Show sample of generated modules
    print(f"\\n📋 Sample modules included:")
    for i, module_config in enumerate(config['modules'][:10]):
        print(f"  {i+1:2d}. {module_config['module_name']}.{module_config['class_name']} ({module_config['complexity']}, {module_config['category']})")
    
    if len(config['modules']) > 10:
        print(f"  ... and {len(config['modules']) - 10} more")

if __name__ == "__main__":
    main()
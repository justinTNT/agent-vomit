#!/usr/bin/env python3
"""
Standardize all existing tests into unified schema for F# wrapping
"""

import json
import torch
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class StandardTestConfig:
    """Standardized test configuration schema"""
    name: str
    test_type: str  # 'forward', 'shape', 'gradient', 'method', 'callable'
    args: List[Any] = None
    kwargs: Dict[str, Any] = None
    input_tensor: Dict[str, Any] = None  # Serialized tensor metadata
    expected_shape: str = None
    method_name: str = None
    tolerance: float = 1e-4
    timeout: float = 10.0
    requires_grad: bool = False
    description: str = ""

@dataclass  
class StandardModuleConfig:
    """Standardized module configuration schema"""
    module_name: str
    class_name: str
    class_variations: List[str] = None
    init_params: Dict[str, Any] = None
    init_param_variations: Dict[str, List[str]] = None
    tests: List[StandardTestConfig] = None
    description: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.class_variations is None:
            self.class_variations = [self.class_name]
        if self.init_params is None:
            self.init_params = {}
        if self.tests is None:
            self.tests = []

def serialize_tensor_metadata(tensor: torch.Tensor) -> Dict[str, Any]:
    """Convert tensor to serializable metadata"""
    return {
        'type': 'tensor',
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'requires_grad': tensor.requires_grad,
        'device': str(tensor.device)
    }

def convert_legacy_test_config(legacy_config: Dict[str, Any]) -> StandardModuleConfig:
    """Convert existing test config to standardized format"""
    
    # Extract basic module info
    module_name = legacy_config['module']
    class_name = legacy_config['class']
    class_variations = legacy_config.get('class_variations', [class_name])
    init_params = legacy_config.get('init_params', {})
    
    # Convert test configurations
    standard_tests = []
    for test in legacy_config.get('tests', []):
        test_name = test['name']
        test_type = test['type']
        
        # Handle different test argument formats
        args = []
        kwargs = {}
        input_tensor = None
        
        if 'args' in test:
            for arg in test['args']:
                if isinstance(arg, torch.Tensor):
                    args.append(serialize_tensor_metadata(arg))
                else:
                    args.append(arg)
        
        if 'input' in test:
            if isinstance(test['input'], torch.Tensor):
                input_tensor = serialize_tensor_metadata(test['input'])
        
        standard_test = StandardTestConfig(
            name=test_name,
            test_type=test_type,
            args=args if args else None,
            kwargs=kwargs if kwargs else None,
            input_tensor=input_tensor,
            expected_shape=test.get('expected'),
            method_name=test.get('method'),
            description=f"{test_type} test for {class_name}.{test_name}"
        )
        standard_tests.append(standard_test)
    
    return StandardModuleConfig(
        module_name=module_name,
        class_name=class_name,
        class_variations=class_variations,
        init_params=init_params,
        tests=standard_tests,
        description=f"Standardized config for {module_name}.{class_name}"
    )

def standardize_all_existing_tests():
    """Convert all existing test configurations to standard format"""
    
    # Import the comprehensive test configuration
    from test_all_25_modules_comprehensive import ALL_MODULES_CONFIG
    
    standardized_configs = []
    
    for legacy_config in ALL_MODULES_CONFIG:
        try:
            standard_config = convert_legacy_test_config(legacy_config)
            standardized_configs.append(standard_config)
            print(f"✅ Standardized {standard_config.module_name}.{standard_config.class_name}")
        except Exception as e:
            print(f"❌ Failed to standardize {legacy_config.get('module', 'unknown')}: {e}")
    
    return standardized_configs

def save_standardized_configs(configs: List[StandardModuleConfig], output_path: str):
    """Save standardized configs to JSON"""
    
    # Convert to serializable format
    serializable_configs = []
    for config in configs:
        config_dict = asdict(config)
        serializable_configs.append(config_dict)
    
    output_data = {
        "schema_version": "1.0",
        "description": "Standardized test configurations for cross-language testing",
        "total_modules": len(configs),
        "modules": serializable_configs
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"📁 Saved {len(configs)} standardized configs to {output_path}")

if __name__ == "__main__":
    print("Standardizing all existing test configurations...")
    
    # Create output directory
    Path("standardized_tests").mkdir(exist_ok=True)
    
    # Convert all existing tests
    standardized = standardize_all_existing_tests()
    
    # Save to JSON
    output_path = "standardized_tests/all_modules_standardized.json"
    save_standardized_configs(standardized, output_path)
    
    print(f"\n🎯 Standardization complete!")
    print(f"📊 Total modules standardized: {len(standardized)}")
    print(f"📋 Output file: {output_path}")
    print(f"🔄 These configs are now ready for F# wrapping!")
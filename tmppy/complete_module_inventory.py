#!/usr/bin/env python3
"""
Complete Module Inventory Generator

Systematically analyzes ALL Python modules in the codebase to create a comprehensive
inventory for testing coverage analysis.
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ModuleInfo:
    """Information about a single module"""
    file_path: str
    relative_path: str
    directory: str
    module_name: str
    classes: List[str]
    nn_module_classes: List[str]
    callable_classes: List[str]
    imports: List[str]
    torch_nn_imports: bool
    has_forward_method: bool
    complexity: str  # simple, medium, complex
    estimated_params: str  # low, medium, high
    has_tests: bool
    priority: int  # 1 (highest) to 5 (lowest)
    lines_of_code: int
    dependencies: List[str]


class ModuleAnalyzer:
    """Analyzes Python modules to extract metadata"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.test_files = set()
        self.module_infos: List[ModuleInfo] = []
        
        # Find all test files first
        self._find_test_files()
    
    def _find_test_files(self):
        """Find all test files to check coverage"""
        for test_file in self.root_dir.rglob("test_*.py"):
            # Extract module name from test file
            module_name = test_file.stem.replace("test_", "")
            self.test_files.add(module_name)
    
    def analyze_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract information
            relative_path = str(file_path.relative_to(self.root_dir))
            directory = str(file_path.parent.relative_to(self.root_dir))
            module_name = file_path.stem
            
            # Skip __init__.py and test files for analysis
            if module_name in ['__init__', 'utils'] or module_name.startswith('test_'):
                return None
            
            classes = []
            nn_module_classes = []
            callable_classes = []
            imports = []
            torch_nn_imports = False
            has_forward_method = False
            dependencies = []
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                        if 'torch' in alias.name:
                            torch_nn_imports = True
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        if 'torch' in node.module:
                            torch_nn_imports = True
                        dependencies.append(node.module)
                
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    
                    # Check if inherits from nn.Module
                    for base in node.bases:
                        if (isinstance(base, ast.Attribute) and 
                            isinstance(base.value, ast.Name) and 
                            base.value.id == 'nn' and 
                            base.attr == 'Module'):
                            nn_module_classes.append(node.name)
                        elif (isinstance(base, ast.Name) and 
                              base.id in ['Module', 'nn.Module']):
                            nn_module_classes.append(node.name)
                    
                    # Check for forward method
                    for method in node.body:
                        if (isinstance(method, ast.FunctionDef) and 
                            method.name == 'forward'):
                            has_forward_method = True
                    
                    # Check if callable (has __call__ method)
                    for method in node.body:
                        if (isinstance(method, ast.FunctionDef) and 
                            method.name == '__call__'):
                            callable_classes.append(node.name)
            
            # Calculate complexity
            lines_of_code = len(content.splitlines())
            complexity = self._assess_complexity(
                content, lines_of_code, len(nn_module_classes), imports
            )
            
            # Estimate parameter count
            estimated_params = self._estimate_parameters(content, complexity)
            
            # Check if has tests
            has_tests = module_name in self.test_files
            
            # Assign priority (1=highest, 5=lowest)
            priority = self._assign_priority(
                directory, complexity, has_tests, torch_nn_imports
            )
            
            return ModuleInfo(
                file_path=str(file_path),
                relative_path=relative_path,
                directory=directory,
                module_name=module_name,
                classes=classes,
                nn_module_classes=nn_module_classes,
                callable_classes=callable_classes,
                imports=imports,
                torch_nn_imports=torch_nn_imports,
                has_forward_method=has_forward_method,
                complexity=complexity,
                estimated_params=estimated_params,
                has_tests=has_tests,
                priority=priority,
                lines_of_code=lines_of_code,
                dependencies=dependencies
            )
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _assess_complexity(self, content: str, lines: int, nn_modules: int, imports: List[str]) -> str:
        """Assess module complexity"""
        # Simple heuristics
        if lines < 100 and nn_modules <= 1:
            return "simple"
        elif lines < 300 and nn_modules <= 3:
            return "medium"
        else:
            return "complex"
    
    def _estimate_parameters(self, content: str, complexity: str) -> str:
        """Estimate parameter mapping difficulty"""
        # Look for patterns that suggest parameter complexity
        conv_patterns = len(re.findall(r'nn\.Conv\d?d', content))
        linear_patterns = len(re.findall(r'nn\.Linear', content))
        attention_patterns = len(re.findall(r'attention|Attention', content, re.IGNORECASE))
        
        total_patterns = conv_patterns + linear_patterns + attention_patterns
        
        if complexity == "simple" and total_patterns <= 2:
            return "low"
        elif complexity == "medium" or total_patterns <= 5:
            return "medium"
        else:
            return "high"
    
    def _assign_priority(self, directory: str, complexity: str, has_tests: bool, torch_nn: bool) -> int:
        """Assign testing priority (1=highest, 5=lowest)"""
        # Priority rules:
        # 1. Simple modules without tests (quick wins)
        # 2. Medium modules without tests 
        # 3. Simple modules with failing tests
        # 4. Complex modules without tests
        # 5. Complex modules with tests (likely working)
        
        if not torch_nn:
            return 5  # Utility modules, lowest priority
        
        if complexity == "simple":
            return 1 if not has_tests else 3
        elif complexity == "medium":
            return 2 if not has_tests else 4
        else:  # complex
            return 4 if not has_tests else 5
    
    def analyze_all_modules(self) -> Dict[str, Any]:
        """Analyze all modules in the codebase"""
        
        # Define module directories to analyze
        module_dirs = [
            "modules",
            "audio-ml-extensions", 
            "crossfade",
            "candidates"
        ]
        
        for dir_name in module_dirs:
            dir_path = self.root_dir / dir_name
            if dir_path.exists():
                print(f"Analyzing {dir_name}...")
                for py_file in dir_path.rglob("*.py"):
                    if py_file.is_file():
                        module_info = self.analyze_file(py_file)
                        if module_info:
                            self.module_infos.append(module_info)
        
        # Generate summary statistics
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary"""
        
        # Group by directory
        by_directory = defaultdict(list)
        by_complexity = defaultdict(list)
        by_priority = defaultdict(list)
        
        tested_modules = []
        untested_modules = []
        
        for module in self.module_infos:
            by_directory[module.directory].append(module)
            by_complexity[module.complexity].append(module)
            by_priority[module.priority].append(module)
            
            if module.has_tests:
                tested_modules.append(module)
            else:
                untested_modules.append(module)
        
        # Sort untested by priority
        untested_modules.sort(key=lambda x: (x.priority, x.complexity))
        
        summary = {
            "total_modules": len(self.module_infos),
            "tested_modules": len(tested_modules),
            "untested_modules": len(untested_modules),
            "coverage_percentage": (len(tested_modules) / len(self.module_infos)) * 100 if self.module_infos else 0,
            
            "by_directory": {
                dir_name: {
                    "count": len(modules),
                    "tested": len([m for m in modules if m.has_tests]),
                    "untested": len([m for m in modules if not m.has_tests]),
                    "modules": [asdict(m) for m in modules]
                }
                for dir_name, modules in by_directory.items()
            },
            
            "by_complexity": {
                complexity: {
                    "count": len(modules),
                    "tested": len([m for m in modules if m.has_tests]),
                    "untested": len([m for m in modules if not m.has_tests])
                }
                for complexity, modules in by_complexity.items()
            },
            
            "by_priority": {
                str(priority): {
                    "count": len(modules),
                    "tested": len([m for m in modules if m.has_tests]),
                    "untested": len([m for m in modules if not m.has_tests]),
                    "modules": [asdict(m) for m in modules]
                }
                for priority, modules in by_priority.items()
            },
            
            "untested_priority_order": [asdict(m) for m in untested_modules],
            
            "statistics": {
                "total_classes": sum(len(m.classes) for m in self.module_infos),
                "total_nn_modules": sum(len(m.nn_module_classes) for m in self.module_infos),
                "total_lines": sum(m.lines_of_code for m in self.module_infos),
                "avg_complexity_score": {
                    "simple": len(by_complexity["simple"]),
                    "medium": len(by_complexity["medium"]),
                    "complex": len(by_complexity["complex"])
                }
            },
            
            "quick_wins": [
                asdict(m) for m in untested_modules 
                if m.priority <= 2 and m.complexity in ["simple", "medium"]
            ][:10],  # Top 10 quick wins
            
            "testing_roadmap": self._generate_testing_roadmap(untested_modules)
        }
        
        return summary
    
    def _generate_testing_roadmap(self, untested_modules: List[ModuleInfo]) -> Dict[str, List[Dict]]:
        """Generate a testing roadmap organized by phases"""
        
        roadmap = {
            "phase_1_infrastructure": [],  # Simple, foundational modules
            "phase_2_core_ml": [],          # Medium complexity ML modules  
            "phase_3_advanced": [],         # Complex modules
            "phase_4_integration": [],      # Multi-modal, advanced features
            "phase_5_experimental": []      # Candidates and experimental
        }
        
        for module in untested_modules:
            module_dict = asdict(module)
            
            # Infrastructure phase: simple modules, data handling
            if (module.complexity == "simple" or 
                "data_" in module.module_name or
                "util" in module.module_name.lower()):
                roadmap["phase_1_infrastructure"].append(module_dict)
            
            # Core ML phase: medium complexity, core audio/ml modules
            elif (module.complexity == "medium" and 
                  module.priority <= 3):
                roadmap["phase_2_core_ml"].append(module_dict)
            
            # Advanced phase: complex modules
            elif module.complexity == "complex":
                roadmap["phase_3_advanced"].append(module_dict)
            
            # Integration phase: multi-modal, crossfade
            elif ("crossfade" in module.directory or 
                  "multi" in module.module_name.lower() or
                  "fusion" in module.module_name.lower()):
                roadmap["phase_4_integration"].append(module_dict)
            
            # Experimental: candidates directory
            elif "candidates" in module.directory:
                roadmap["phase_5_experimental"].append(module_dict)
            
            else:
                # Default to core ML
                roadmap["phase_2_core_ml"].append(module_dict)
        
        return roadmap


def main():
    """Generate complete module inventory"""
    
    root_dir = "/Users/jtnt/Play/agent-vomit"
    analyzer = ModuleAnalyzer(root_dir)
    
    print("Starting comprehensive module analysis...")
    summary = analyzer.analyze_all_modules()
    
    # Save detailed inventory
    inventory_file = Path(root_dir) / "complete_module_inventory.json"
    with open(inventory_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== COMPLETE MODULE INVENTORY ===")
    print(f"Total modules found: {summary['total_modules']}")
    print(f"Tested modules: {summary['tested_modules']}")
    print(f"Untested modules: {summary['untested_modules']}")
    print(f"Coverage: {summary['coverage_percentage']:.1f}%")
    
    print(f"\n=== BY DIRECTORY ===")
    for dir_name, info in summary['by_directory'].items():
        print(f"{dir_name}: {info['count']} total, {info['untested']} untested")
    
    print(f"\n=== BY COMPLEXITY ===")
    for complexity, info in summary['by_complexity'].items():
        print(f"{complexity}: {info['count']} total, {info['untested']} untested")
    
    print(f"\n=== TOP 10 QUICK WINS ===")
    for i, module in enumerate(summary['quick_wins'], 1):
        print(f"{i:2d}. {module['module_name']} ({module['complexity']}, priority {module['priority']})")
    
    print(f"\n=== TESTING ROADMAP ===")
    for phase, modules in summary['testing_roadmap'].items():
        if modules:
            print(f"{phase}: {len(modules)} modules")
    
    print(f"\n=== 34 UNTESTED MODULES (Priority Order) ===")
    for i, module in enumerate(summary['untested_priority_order'][:34], 1):
        print(f"{i:2d}. {module['module_name']:30} {module['complexity']:8} P{module['priority']} {module['directory']}")
    
    print(f"\nComplete inventory saved to: {inventory_file}")
    
    return summary


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Untested Modules Priority Analysis

Identifies the 34 highest-priority untested modules and provides specific
testing strategies for each to achieve 100% coverage systematically.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


class UntestedModuleAnalyzer:
    """Analyzes untested modules to create actionable testing plan"""
    
    def __init__(self, inventory_file: str):
        with open(inventory_file, 'r') as f:
            self.inventory = json.load(f)
    
    def get_priority_untested_modules(self, limit: int = 34) -> List[Dict]:
        """Get top priority untested modules"""
        return self.inventory['untested_priority_order'][:limit]
    
    def analyze_testing_difficulty(self, module: Dict) -> str:
        """Analyze testing difficulty for a module"""
        
        # Simple modules with basic nn.Module classes
        if (module['complexity'] == 'simple' and 
            len(module['nn_module_classes']) <= 2 and
            module['estimated_params'] == 'low'):
            return "EASY"
        
        # Medium complexity with standard patterns
        elif (module['complexity'] == 'medium' and 
              module['estimated_params'] in ['low', 'medium']):
            return "MODERATE"
        
        # Complex modules or high parameter count
        elif (module['complexity'] == 'complex' or 
              module['estimated_params'] == 'high'):
            return "HARD"
        
        # Audio analysis and advanced modules
        elif ('audio_analysis' in module['directory'] or 
              'advanced_modules' in module['directory']):
            return "VERY_HARD"
        
        return "MODERATE"
    
    def generate_testing_strategy(self, module: Dict) -> Dict[str, Any]:
        """Generate specific testing strategy for a module"""
        
        difficulty = self.analyze_testing_difficulty(module)
        
        strategy = {
            "module_name": module['module_name'],
            "file_path": module['file_path'],
            "difficulty": difficulty,
            "estimated_hours": self._estimate_testing_hours(difficulty),
            "testing_approach": self._get_testing_approach(module, difficulty),
            "parameter_strategy": self._get_parameter_strategy(module),
            "dependencies": module.get('dependencies', []),
            "blockers": self._identify_blockers(module),
            "success_criteria": self._define_success_criteria(module)
        }
        
        return strategy
    
    def _estimate_testing_hours(self, difficulty: str) -> float:
        """Estimate hours needed for testing"""
        hours_map = {
            "EASY": 0.5,
            "MODERATE": 1.0,
            "HARD": 2.0,
            "VERY_HARD": 4.0
        }
        return hours_map.get(difficulty, 1.0)
    
    def _get_testing_approach(self, module: Dict, difficulty: str) -> str:
        """Get specific testing approach"""
        
        if difficulty == "EASY":
            return "Standard parameter mapping with minimal inputs"
        
        elif difficulty == "MODERATE":
            return "Multi-dimensional inputs with parameter validation"
        
        elif difficulty == "HARD":
            return "Complex input generation with mock dependencies"
        
        elif difficulty == "VERY_HARD":
            return "Extensive mocking and staged integration testing"
        
        return "Standard testing approach"
    
    def _get_parameter_strategy(self, module: Dict) -> str:
        """Get parameter mapping strategy"""
        
        if module['estimated_params'] == 'low':
            return "Direct parameter mapping"
        
        elif module['estimated_params'] == 'medium':
            return "Structured parameter mapping with validation"
        
        else:  # high
            return "Hierarchical parameter mapping with constraints"
    
    def _identify_blockers(self, module: Dict) -> List[str]:
        """Identify potential testing blockers"""
        blockers = []
        
        # Check for external dependencies
        external_deps = ['librosa', 'scipy', 'sklearn', 'transformers']
        for dep in module.get('dependencies', []):
            if any(ext in dep for ext in external_deps):
                blockers.append(f"External dependency: {dep}")
        
        # Check for complex audio processing
        if 'audio' in module['module_name'].lower():
            blockers.append("Audio processing requirements")
        
        # Check for GPU requirements
        if any(keyword in module['module_name'].lower() 
               for keyword in ['gan', 'transformer', 'attention']):
            blockers.append("Potential GPU requirement")
        
        return blockers
    
    def _define_success_criteria(self, module: Dict) -> List[str]:
        """Define success criteria for module testing"""
        criteria = [
            "Module instantiates without errors",
            "Forward pass completes successfully",
            "Output tensor shapes are correct"
        ]
        
        if module['complexity'] == 'complex':
            criteria.extend([
                "All major code paths are exercised",
                "Error handling is validated"
            ])
        
        return criteria
    
    def generate_implementation_plan(self) -> Dict[str, Any]:
        """Generate complete implementation plan for 34 modules"""
        
        priority_modules = self.get_priority_untested_modules(34)
        
        # Group modules by implementation waves
        waves = {
            "wave_1_quick_wins": [],      # 1-6: Simplest modules (3 hours)
            "wave_2_medium_effort": [],   # 7-14: Medium complexity (8 hours)  
            "wave_3_core_audio": [],      # 15-22: Core audio modules (16 hours)
            "wave_4_advanced": [],        # 23-28: Advanced ML modules (24 hours)
            "wave_5_integration": []      # 29-34: Integration modules (24 hours)
        }
        
        total_hours = 0
        
        for i, module in enumerate(priority_modules, 1):
            strategy = self.generate_testing_strategy(module)
            total_hours += strategy['estimated_hours']
            
            # Assign to waves
            if i <= 6:
                waves["wave_1_quick_wins"].append(strategy)
            elif i <= 14:
                waves["wave_2_medium_effort"].append(strategy)
            elif i <= 22:
                waves["wave_3_core_audio"].append(strategy)
            elif i <= 28:
                waves["wave_4_advanced"].append(strategy)
            else:
                waves["wave_5_integration"].append(strategy)
        
        # Calculate wave statistics
        wave_stats = {}
        for wave_name, modules in waves.items():
            wave_hours = sum(m['estimated_hours'] for m in modules)
            wave_stats[wave_name] = {
                "module_count": len(modules),
                "estimated_hours": wave_hours,
                "difficulty_breakdown": self._get_difficulty_breakdown(modules),
                "modules": modules
            }
        
        return {
            "total_modules": len(priority_modules),
            "total_estimated_hours": total_hours,
            "implementation_waves": wave_stats,
            "success_metrics": {
                "target_coverage": "100%",
                "modules_to_test": 34,
                "expected_timeline": f"{total_hours} hours over 5 waves"
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _get_difficulty_breakdown(self, modules: List[Dict]) -> Dict[str, int]:
        """Get difficulty breakdown for a group of modules"""
        breakdown = {"EASY": 0, "MODERATE": 0, "HARD": 0, "VERY_HARD": 0}
        for module in modules:
            breakdown[module['difficulty']] += 1
        return breakdown
    
    def _generate_recommendations(self) -> List[str]:
        """Generate implementation recommendations"""
        return [
            "Start with Wave 1 (quick wins) to build momentum and establish patterns",
            "Use successful Wave 1 tests as templates for subsequent waves",
            "Implement robust parameter generation utilities early",
            "Create reusable test fixtures for common audio/ML patterns",
            "Focus on one wave at a time to maintain quality and consistency",
            "Track success rate and adjust parameter strategies as needed",
            "Consider parallel testing for independent modules within waves",
            "Build comprehensive error handling for complex modules",
            "Document successful patterns for future module testing",
            "Validate all tests pass consistently before moving to next wave"
        ]


def main():
    """Generate complete untested modules analysis"""
    
    inventory_file = "/Users/jtnt/Play/agent-vomit/complete_module_inventory.json"
    analyzer = UntestedModuleAnalyzer(inventory_file)
    
    # Generate implementation plan
    plan = analyzer.generate_implementation_plan()
    
    # Save plan
    plan_file = Path("/Users/jtnt/Play/agent-vomit/untested_modules_implementation_plan.json")
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=2)
    
    print("=== UNTESTED MODULES IMPLEMENTATION PLAN ===")
    print(f"Total modules to test: {plan['total_modules']}")
    print(f"Estimated total effort: {plan['total_estimated_hours']} hours")
    print()
    
    print("=== IMPLEMENTATION WAVES ===")
    for wave_name, wave_info in plan['implementation_waves'].items():
        print(f"\n{wave_name.upper().replace('_', ' ')}")
        print(f"  Modules: {wave_info['module_count']}")
        print(f"  Hours: {wave_info['estimated_hours']}")
        print(f"  Difficulty: {wave_info['difficulty_breakdown']}")
        
        print(f"  Modules in this wave:")
        for i, module in enumerate(wave_info['modules'], 1):
            print(f"    {i:2d}. {module['module_name']:30} {module['difficulty']:10} ({module['estimated_hours']}h)")
    
    print(f"\n=== TOP RECOMMENDATIONS ===")
    for i, rec in enumerate(plan['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    print(f"\nComplete plan saved to: {plan_file}")
    
    # Print first wave details for immediate action
    print(f"\n=== WAVE 1 QUICK WINS (START HERE) ===")
    wave1 = plan['implementation_waves']['wave_1_quick_wins']['modules']
    for i, module in enumerate(wave1, 1):
        print(f"\n{i}. {module['module_name']} ({module['difficulty']}, {module['estimated_hours']}h)")
        print(f"   File: {module['file_path']}")
        print(f"   Strategy: {module['testing_approach']}")
        if module['blockers']:
            print(f"   Blockers: {', '.join(module['blockers'])}")
    
    return plan


if __name__ == "__main__":
    main()
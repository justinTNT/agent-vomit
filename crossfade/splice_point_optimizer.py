"""
SplicePointOptimizer - Find globally optimal splice point combination

This module performs multi-objective optimization to find the best
A exit + B entry combination, balancing musical quality vs processing requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .exit_point_analyzer import ExitCandidates, ExitCandidate
from .entry_point_analyzer import EntryCandidates, EntryCandidate
from .harmonic_compatibility_analyzer import HarmonicMatch
from .rhythmic_compatibility_analyzer import RhythmicMatch
from .energy_compatibility_analyzer import EnergyMatch


@dataclass
class OptimalSplice:
    """Optimal splice point selection results"""
    a_exit_sample: int              # Best exit sample in Track A
    b_entry_sample: int             # Best entry sample in Track B
    quality_score: float            # Combined quality score [0,1]
    musical_compatibility: float    # Musical quality score [0,1]
    processing_quality: float       # Expected processing quality [0,1]
    confidence: float              # Confidence in selection [0,1]
    optimization_details: dict      # Detailed scoring breakdown
    fallback_strategy: Optional[str]  # Alternative if optimal fails


class SplicePointOptimizer(nn.Module):
    """
    Find globally optimal splice point combination through multi-objective optimization.
    
    Evaluates all combinations of exit and entry candidates to find the best
    balance between musical compatibility, processing quality, and timing alignment.
    """
    
    def __init__(self, 
                 max_combinations: int = 50,
                 musical_weight: float = 0.4,
                 processing_weight: float = 0.3,
                 timing_weight: float = 0.3,
                 min_quality_threshold: float = 0.3):
        super().__init__()
        
        self.max_combinations = max_combinations
        self.musical_weight = musical_weight
        self.processing_weight = processing_weight
        self.timing_weight = timing_weight
        self.min_quality_threshold = min_quality_threshold
        
        # Neural network for global optimization
        self.optimization_network = nn.Sequential(
            nn.Linear(12, 64),  # Multiple compatibility scores
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation network
        self.confidence_estimator = nn.Sequential(
            nn.Linear(8, 32),  # Confidence factors
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                exit_candidates: ExitCandidates,
                entry_candidates: EntryCandidates,
                harmonic_analysis_func,
                rhythmic_analysis_func,
                energy_analysis_func) -> OptimalSplice:
        """
        Find optimal splice point combination.
        
        Args:
            exit_candidates: Track A exit point options
            entry_candidates: Track B entry point options
            harmonic_analysis_func: Function to analyze harmonic compatibility
            rhythmic_analysis_func: Function to analyze rhythmic compatibility  
            energy_analysis_func: Function to analyze energy compatibility
            
        Returns:
            OptimalSplice with best combination and quality scores
        """
        
        # Limit combinations to prevent computational explosion
        exit_options = exit_candidates.positions[:min(10, len(exit_candidates.positions))]
        entry_options = entry_candidates.positions[:min(10, len(entry_candidates.positions))]
        
        if not exit_options or not entry_options:
            return self._create_fallback_splice(exit_candidates, entry_candidates)
        
        # Evaluate all combinations
        combinations = []
        for exit_candidate in exit_options:
            for entry_candidate in entry_options:
                combination_score = self._evaluate_combination(
                    exit_candidate, entry_candidate,
                    harmonic_analysis_func, rhythmic_analysis_func, energy_analysis_func
                )
                
                if combination_score['overall_score'] >= self.min_quality_threshold:
                    combinations.append({
                        'exit': exit_candidate,
                        'entry': entry_candidate,
                        **combination_score
                    })
        
        if not combinations:
            return self._create_fallback_splice(exit_candidates, entry_candidates)
        
        # Sort by overall score and select best
        combinations.sort(key=lambda x: x['overall_score'], reverse=True)
        best_combination = combinations[0]
        
        # Create optimal splice result
        optimal_splice = OptimalSplice(
            a_exit_sample=best_combination['exit'].sample_position,
            b_entry_sample=best_combination['entry'].sample_position,
            quality_score=best_combination['overall_score'],
            musical_compatibility=best_combination['musical_score'],
            processing_quality=best_combination['processing_score'],
            confidence=best_combination['confidence'],
            optimization_details=self._create_optimization_details(combinations[:5]),
            fallback_strategy=self._determine_fallback_strategy(combinations)
        )
        
        return optimal_splice
    
    def _evaluate_combination(self, 
                            exit_candidate: ExitCandidate,
                            entry_candidate: EntryCandidate,
                            harmonic_analysis_func,
                            rhythmic_analysis_func,
                            energy_analysis_func) -> dict:
        """Evaluate a specific exit/entry combination"""
        
        try:
            # Get compatibility analyses for this combination
            harmonic_match = harmonic_analysis_func(exit_candidate, entry_candidate)
            rhythmic_match = rhythmic_analysis_func(exit_candidate, entry_candidate)
            energy_match = energy_analysis_func(
                exit_candidate.sample_position, entry_candidate.sample_position
            )
            
            # Calculate component scores
            musical_score = self._calculate_musical_score(
                exit_candidate, entry_candidate, harmonic_match, rhythmic_match
            )
            
            processing_score = self._calculate_processing_score(
                harmonic_match, rhythmic_match, energy_match
            )
            
            timing_score = self._calculate_timing_score(
                exit_candidate, entry_candidate, energy_match
            )
            
            # Use neural network for global optimization
            overall_score = self._calculate_neural_optimization_score(
                exit_candidate, entry_candidate, harmonic_match, rhythmic_match, energy_match
            )
            
            # Calculate confidence
            confidence = self._calculate_combination_confidence(
                exit_candidate, entry_candidate, harmonic_match, rhythmic_match, energy_match
            )
            
        except Exception as e:
            # Fallback to heuristic scoring if analysis fails
            musical_score = (exit_candidate.musical_score + entry_candidate.overall_score) / 2.0
            processing_score = 0.6  # Assume moderate processing quality
            timing_score = (exit_candidate.beat_alignment + entry_candidate.beat_alignment) / 2.0
            overall_score = (musical_score + processing_score + timing_score) / 3.0
            confidence = 0.4  # Low confidence due to analysis failure
        
        return {
            'musical_score': musical_score,
            'processing_score': processing_score,
            'timing_score': timing_score,
            'overall_score': overall_score,
            'confidence': confidence
        }
    
    def _calculate_musical_score(self, 
                               exit_candidate: ExitCandidate,
                               entry_candidate: EntryCandidate,
                               harmonic_match: HarmonicMatch,
                               rhythmic_match: RhythmicMatch) -> float:
        """Calculate musical compatibility score"""
        
        # Base musical quality from candidates
        exit_musical = exit_candidate.musical_score
        entry_musical = entry_candidate.overall_score
        
        # Harmonic compatibility
        harmonic_score = harmonic_match.compatibility_score
        
        # Rhythmic compatibility  
        rhythmic_score = rhythmic_match.compatibility_score
        
        # "Drop on the 1" bonus
        drop_bonus = entry_candidate.drop_score * 0.1
        
        # Phrase completion bonus
        phrase_bonus = (exit_candidate.phrase_completion + entry_candidate.phrase_start_score) * 0.05
        
        # Combined musical score
        musical_score = (
            exit_musical * 0.2 +
            entry_musical * 0.2 +
            harmonic_score * 0.3 +
            rhythmic_score * 0.3 +
            drop_bonus +
            phrase_bonus
        )
        
        return min(1.0, max(0.0, musical_score))
    
    def _calculate_processing_score(self, 
                                  harmonic_match: HarmonicMatch,
                                  rhythmic_match: RhythmicMatch,
                                  energy_match: EnergyMatch) -> float:
        """Calculate processing quality score"""
        
        # Harmonic processing quality
        harmonic_processing = 1.0 - harmonic_match.dissonance_risk
        
        # Rhythmic processing quality  
        rhythmic_processing = 1.0 - rhythmic_match.correction_difficulty
        
        # Energy matching quality
        energy_processing = energy_match.compatibility_score
        
        # Processing requirements penalty
        processing_penalty = 0.0
        
        # Penalize large pitch shifts
        if abs(harmonic_match.required_pitch_shift) > 1.0:
            processing_penalty += 0.1
        
        # Penalize large tempo changes
        if abs(rhythmic_match.required_rate_change - 1.0) > 0.05:
            processing_penalty += 0.1
        
        # Combined processing score
        processing_score = (
            harmonic_processing * 0.35 +
            rhythmic_processing * 0.35 +
            energy_processing * 0.3 -
            processing_penalty
        )
        
        return min(1.0, max(0.0, processing_score))
    
    def _calculate_timing_score(self, 
                              exit_candidate: ExitCandidate,
                              entry_candidate: EntryCandidate,
                              energy_match: EnergyMatch) -> float:
        """Calculate timing alignment score"""
        
        # Beat alignment scores
        exit_beat_score = exit_candidate.beat_alignment
        entry_beat_score = entry_candidate.beat_alignment
        
        # Energy flow score
        energy_flow_score = energy_match.energy_flow_score
        
        # Combined timing score
        timing_score = (
            exit_beat_score * 0.3 +
            entry_beat_score * 0.4 +
            energy_flow_score * 0.3
        )
        
        return timing_score
    
    def _calculate_neural_optimization_score(self, 
                                           exit_candidate: ExitCandidate,
                                           entry_candidate: EntryCandidate,
                                           harmonic_match: HarmonicMatch,
                                           rhythmic_match: RhythmicMatch,
                                           energy_match: EnergyMatch) -> float:
        """Use neural network for global optimization scoring"""
        
        features = torch.tensor([
            exit_candidate.overall_score,
            entry_candidate.overall_score,
            exit_candidate.beat_alignment,
            entry_candidate.beat_alignment,
            entry_candidate.drop_score,
            harmonic_match.compatibility_score,
            rhythmic_match.compatibility_score,
            energy_match.compatibility_score,
            1.0 - harmonic_match.dissonance_risk,
            1.0 - rhythmic_match.correction_difficulty,
            energy_match.energy_flow_score,
            harmonic_match.correction_confidence
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                score = self.optimization_network(features.unsqueeze(0))
                score = float(score.squeeze())
        except:
            # Fallback to weighted combination
            score = (
                self.musical_weight * (harmonic_match.compatibility_score + rhythmic_match.compatibility_score) / 2.0 +
                self.processing_weight * (1.0 - (harmonic_match.dissonance_risk + rhythmic_match.correction_difficulty) / 2.0) +
                self.timing_weight * (exit_candidate.beat_alignment + entry_candidate.beat_alignment) / 2.0
            )
        
        return score
    
    def _calculate_combination_confidence(self, 
                                        exit_candidate: ExitCandidate,
                                        entry_candidate: EntryCandidate,
                                        harmonic_match: HarmonicMatch,
                                        rhythmic_match: RhythmicMatch,
                                        energy_match: EnergyMatch) -> float:
        """Calculate confidence in the combination selection"""
        
        features = torch.tensor([
            exit_candidate.overall_score,
            entry_candidate.overall_score,
            harmonic_match.correction_confidence,
            rhythmic_match.tempo_stability_factor,
            energy_match.compatibility_score,
            1.0 - harmonic_match.dissonance_risk,
            1.0 - rhythmic_match.correction_difficulty,
            energy_match.energy_flow_score
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                confidence = self.confidence_estimator(features.unsqueeze(0))
                confidence = float(confidence.squeeze())
        except:
            # Fallback heuristic
            confidence = (
                exit_candidate.overall_score * 0.15 +
                entry_candidate.overall_score * 0.15 +
                harmonic_match.correction_confidence * 0.25 +
                rhythmic_match.tempo_stability_factor * 0.2 +
                energy_match.compatibility_score * 0.25
            )
        
        return confidence
    
    def _create_optimization_details(self, top_combinations: List[dict]) -> dict:
        """Create detailed optimization results"""
        
        return {
            'total_combinations_evaluated': len(top_combinations),
            'top_5_scores': [combo['overall_score'] for combo in top_combinations],
            'score_distribution': {
                'max': max(combo['overall_score'] for combo in top_combinations),
                'min': min(combo['overall_score'] for combo in top_combinations),
                'mean': sum(combo['overall_score'] for combo in top_combinations) / len(top_combinations)
            },
            'optimization_strategy': 'neural_multi_objective'
        }
    
    def _determine_fallback_strategy(self, combinations: List[dict]) -> Optional[str]:
        """Determine fallback strategy if optimal fails"""
        
        if not combinations:
            return "hard_cut_timing_optimized"
        
        best = combinations[0]
        
        # If processing quality is low, suggest hard cut
        if best['processing_score'] < 0.5:
            return "hard_cut_beat_aligned"
        
        # If musical compatibility is low, try timing-only approach
        if best['musical_score'] < 0.4:
            return "timing_only_no_processing"
        
        # If overall quality is marginal, suggest conservative approach
        if best['overall_score'] < 0.6:
            return "conservative_minimal_processing"
        
        return None  # No fallback needed
    
    def _create_fallback_splice(self, 
                              exit_candidates: ExitCandidates,
                              entry_candidates: EntryCandidates) -> OptimalSplice:
        """Create fallback splice when optimization fails"""
        
        # Use best individual candidates if available
        exit_sample = (exit_candidates.best_exit.sample_position 
                      if exit_candidates.best_exit 
                      else 0)
        
        entry_sample = (entry_candidates.best_entry.sample_position
                       if entry_candidates.best_entry
                       else 0)
        
        return OptimalSplice(
            a_exit_sample=exit_sample,
            b_entry_sample=entry_sample,
            quality_score=0.3,  # Low quality fallback
            musical_compatibility=0.3,
            processing_quality=0.5,
            confidence=0.2,
            optimization_details={'status': 'fallback_used'},
            fallback_strategy="hard_cut_timing_optimized"
        )
    
    def get_alternative_splices(self, 
                              exit_candidates: ExitCandidates,
                              entry_candidates: EntryCandidates,
                              top_n: int = 3) -> List[OptimalSplice]:
        """Get alternative splice options beyond the optimal"""
        
        # This would implement similar logic to forward() but return multiple options
        # For now, return a placeholder structure
        alternatives = []
        
        # Could return second-best, different style preferences, etc.
        # Implementation would depend on specific use case requirements
        
        return alternatives
    
    def explain_optimization_decision(self, optimal_splice: OptimalSplice) -> dict:
        """Explain why this splice was chosen"""
        
        return {
            'primary_factors': [
                f"Musical compatibility: {optimal_splice.musical_compatibility:.2f}",
                f"Processing quality: {optimal_splice.processing_quality:.2f}",
                f"Overall confidence: {optimal_splice.confidence:.2f}"
            ],
            'decision_rationale': self._generate_decision_rationale(optimal_splice),
            'potential_concerns': self._identify_potential_concerns(optimal_splice),
            'optimization_method': 'multi_objective_neural_optimization'
        }
    
    def _generate_decision_rationale(self, optimal_splice: OptimalSplice) -> str:
        """Generate human-readable decision rationale"""
        
        if optimal_splice.quality_score > 0.8:
            return "High quality match found with excellent musical and processing compatibility"
        elif optimal_splice.musical_compatibility > 0.7:
            return "Good musical match selected, processing quality acceptable"
        elif optimal_splice.processing_quality > 0.7:
            return "Clean processing possible, musical compatibility reasonable"
        else:
            return "Best available option selected, may require fallback strategy"
    
    def _identify_potential_concerns(self, optimal_splice: OptimalSplice) -> List[str]:
        """Identify potential concerns with the selected splice"""
        
        concerns = []
        
        if optimal_splice.quality_score < 0.5:
            concerns.append("Overall quality below preferred threshold")
        
        if optimal_splice.musical_compatibility < 0.4:
            concerns.append("Musical compatibility may be noticeable")
        
        if optimal_splice.processing_quality < 0.4:
            concerns.append("Processing artifacts may be audible")
        
        if optimal_splice.confidence < 0.5:
            concerns.append("Low confidence in splice success")
        
        if optimal_splice.fallback_strategy:
            concerns.append(f"Fallback strategy recommended: {optimal_splice.fallback_strategy}")
        
        return concerns


if __name__ == "__main__":
    # Test with mock data
    from .exit_point_analyzer import ExitCandidate, ExitCandidates
    from .entry_point_analyzer import EntryCandidate, EntryCandidates
    from .harmonic_compatibility_analyzer import HarmonicMatch, CompatibilityLevel
    from .rhythmic_compatibility_analyzer import RhythmicMatch, TempoCompatibility
    from .energy_compatibility_analyzer import EnergyMatch, EnergyCompatibilityLevel
    
    # Create mock candidates
    exit_candidate = ExitCandidate(
        sample_position=100000,
        musical_score=0.8,
        energy_level=0.7,
        beat_alignment=0.9,
        phrase_completion=0.6,
        overall_score=0.75
    )
    
    entry_candidate = EntryCandidate(
        sample_position=50000,
        drop_score=0.8,
        energy_level=0.7,
        buildup_score=0.6,
        phrase_start_score=0.7,
        beat_alignment=0.85,
        overall_score=0.74
    )
    
    exit_candidates = ExitCandidates(
        positions=[exit_candidate],
        best_exit=exit_candidate,
        musical_context={}
    )
    
    entry_candidates = EntryCandidates(
        positions=[entry_candidate],
        best_entry=entry_candidate,
        drop_opportunities=[entry_candidate],
        intro_character="building"
    )
    
    # Mock analysis functions
    def mock_harmonic_analysis(exit, entry):
        return HarmonicMatch(
            compatibility_score=0.7,
            compatibility_level=CompatibilityLevel.GOOD,
            required_pitch_shift=1.0,
            dissonance_risk=0.2,
            key_relationship="perfect_fifth_related",
            correction_confidence=0.8,
            harmonic_tension=0.3,
            processing_recommendation="light_correction_+1.0_semitones"
        )
    
    def mock_rhythmic_analysis(exit, entry):
        return RhythmicMatch(
            compatibility_score=0.8,
            compatibility_level=TempoCompatibility.EXCELLENT,
            required_rate_change=1.02,
            phase_offset=0.1,
            bpm_difference=2.0,
            tempo_stability_factor=0.9,
            correction_difficulty=0.2,
            processing_recommendation="light_tempo_correction_+2.0_percent"
        )
    
    def mock_energy_analysis(exit_pos, entry_pos):
        return EnergyMatch(
            compatibility_score=0.75,
            compatibility_level=EnergyCompatibilityLevel.GOOD,
            level_difference_db=-2.0,
            spectral_balance_score=0.8,
            loudness_adjustment_db=-1.5,
            dynamic_range_compatibility=0.7,
            energy_flow_score=0.8,
            processing_recommendation="moderate_level_adjustment_-2.0_db"
        )
    
    # Test optimizer
    optimizer = SplicePointOptimizer()
    
    optimal_splice = optimizer(
        exit_candidates,
        entry_candidates,
        mock_harmonic_analysis,
        mock_rhythmic_analysis,
        mock_energy_analysis
    )
    
    print("=== Optimal Splice Results ===")
    print(f"A exit sample: {optimal_splice.a_exit_sample}")
    print(f"B entry sample: {optimal_splice.b_entry_sample}")
    print(f"Quality score: {optimal_splice.quality_score:.3f}")
    print(f"Musical compatibility: {optimal_splice.musical_compatibility:.3f}")
    print(f"Processing quality: {optimal_splice.processing_quality:.3f}")
    print(f"Confidence: {optimal_splice.confidence:.3f}")
    
    if optimal_splice.fallback_strategy:
        print(f"Fallback strategy: {optimal_splice.fallback_strategy}")
    
    # Test explanation
    explanation = optimizer.explain_optimization_decision(optimal_splice)
    print(f"\nDecision rationale: {explanation['decision_rationale']}")
    if explanation['potential_concerns']:
        print(f"Concerns: {explanation['potential_concerns']}")
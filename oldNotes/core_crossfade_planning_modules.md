# Core Crossfade Planning Modules

## System Overview

**Pure Functional Interface**:
```python
def plan_crossfade(track_a_audio, track_b_audio):
    """
    Input: Two 30s audio segments (A outro, B intro)
    Output: CrossfadePlan(translation, envelope, pitch_shift, rate_change) OR None
    """
```

**Core Parameters to Extract & Optimize**:
- **Translation**: Where in A to exit, where in B to enter (sample positions)
- **Envelope**: Crossfade curve shape and timing
- **Pitch**: Semitone adjustment for B (0 = no change)
- **Rate**: Tempo adjustment for B (1.0 = no change)

---

## Audio Analysis Modules

### **Tier 1: Musical Feature Extraction**

1. **BeatGridExtractor** - Sig: 10, Conf: 7
   - Extract precise beat/bar/phrase grid from audio
   - Tempo stability analysis
   - Beat confidence scoring per time position
   - **Output**: BeatGrid(bpm, beat_times, bar_times, confidence_curve)

2. **MusicalKeyExtractor** - Sig: 10, Conf: 6
   - Chromatic key detection with confidence
   - Key stability over time (detect modulations)
   - Tonal vs atonal classification
   - **Output**: KeyProfile(key, mode, confidence, stability_zones)

3. **EnergyProfileExtractor** - Sig: 9, Conf: 9
   - RMS energy curve over time
   - Spectral energy distribution (low/mid/high)
   - Dynamic range analysis
   - Energy slope detection (building/falling)
   - **Output**: EnergyProfile(rms_curve, spectral_bands, dynamics)

4. **StructuralFeatureExtractor** - Sig: 8, Conf: 7
   - Detect drops, breaks, build-ups
   - Vocal vs instrumental sections
   - Phrase boundary detection
   - Musical "events" for alignment targets
   - **Output**: StructuralEvents(drops, breaks, phrase_boundaries, vocal_regions)

### **Tier 2: Splice Point Analysis**

5. **ExitPointAnalyzer** - Sig: 9, Conf: 8
   - Find optimal exit points in Track A outro
   - Musical phrase completion
   - Energy-appropriate exit moments
   - Zero-crossing alignment for clean cuts
   - **Output**: ExitCandidates(positions, musical_scores, energy_levels)

6. **EntryPointAnalyzer** - Sig: 9, Conf: 8
   - Find optimal entry points in Track B intro
   - Detect "drop on the 1" opportunities
   - Build-up detection for energy matching
   - Musical phrase beginnings
   - **Output**: EntryCandidates(positions, drop_scores, energy_levels, phrase_starts)

---

## Compatibility Analysis Modules

### **Tier 3: Musical Compatibility**

7. **HarmonicCompatibilityAnalyzer** - Sig: 9, Conf: 7
   - Key compatibility matrix (circle of fifths)
   - Required pitch shift calculation
   - Harmonic dissonance prediction
   - Genre-aware harmonic tolerance
   - **Output**: HarmonicMatch(compatibility_score, required_pitch_shift, dissonance_risk)

8. **RhythmicCompatibilityAnalyzer** - Sig: 8, Conf: 8
   - BPM compatibility analysis
   - Required tempo adjustment calculation
   - Polyrhythm detection and handling
   - Beat phase alignment requirements
   - **Output**: RhythmicMatch(compatibility_score, required_rate_change, phase_offset)

9. **EnergyCompatibilityAnalyzer** - Sig: 8, Conf: 9
   - Energy level matching between A exit and B entry
   - Dynamic range compatibility
   - Spectral energy balance
   - Perceptual loudness matching
   - **Output**: EnergyMatch(level_difference, spectral_balance, loudness_adjustment)

### **Tier 4: Optimization Engine**

10. **SplicePointOptimizer** - Sig: 10, Conf: 8
    - Multi-objective optimization across all compatibility scores
    - Find globally optimal A exit + B entry combination
    - Balance musical quality vs processing requirements
    - "Drop on the 1" preference weighting
    - **Output**: OptimalSplice(a_exit_sample, b_entry_sample, quality_score)

11. **ProcessingParameterCalculator** - Sig: 9, Conf: 8
    - Calculate exact pitch shift (semitones)
    - Calculate exact rate change (ratio)
    - Predict processing artifacts
    - Quality vs modification trade-offs
    - **Output**: ProcessingParams(pitch_shift, rate_change, artifact_prediction, quality_score)

12. **CrossfadeEnvelopeDesigner** - Sig: 9, Conf: 9
    - Design optimal crossfade curve for musical content
    - Energy-matched transitions
    - Frequency-selective crossfading decisions
    - Curve shape optimization (linear, equal-power, S-curve, custom)
    - **Output**: CrossfadeEnvelope(curve_type, duration, frequency_weights)

---

## Core Data Structures

### **Analysis Results**:
```python
@dataclass
class TrackAnalysis:
    beat_grid: BeatGrid
    key_profile: KeyProfile  
    energy_profile: EnergyProfile
    structural_events: StructuralEvents
    exit_candidates: ExitCandidates  # For Track A
    entry_candidates: EntryCandidates  # For Track B

@dataclass
class CrossfadePlan:
    # Translation parameters
    a_exit_sample: int          # Sample position in A to exit
    b_entry_sample: int         # Sample position in B to enter
    
    # B transformation parameters
    pitch_shift: float          # Semitones (+/- 12)
    rate_change: float          # Tempo ratio (0.5 = half speed, 2.0 = double)
    
    # Crossfade parameters
    envelope: CrossfadeEnvelope # Curve shape and timing
    
    # Quality metrics
    musical_quality: float      # 0-1 musical compatibility score
    processing_quality: float   # 0-1 expected audio quality after processing
    overall_score: float        # Combined quality metric
```

---

## System Architecture

### **Processing Pipeline**:
```
Track A (30s) ──┬─→ [BeatGridExtractor] ──┐
                ├─→ [MusicalKeyExtractor] ─┤
                ├─→ [EnergyProfileExtractor] ─┤──→ TrackA_Analysis
                ├─→ [StructuralFeatureExtractor] ─┤
                └─→ [ExitPointAnalyzer] ──┘

Track B (30s) ──┬─→ [Same Analysis Pipeline] ──→ TrackB_Analysis
                └─→ [EntryPointAnalyzer]

TrackA_Analysis + TrackB_Analysis ──┬─→ [HarmonicCompatibilityAnalyzer] ──┐
                                    ├─→ [RhythmicCompatibilityAnalyzer] ─┤
                                    └─→ [EnergyCompatibilityAnalyzer] ──┘
                                                        │
                                                        ▼
                                            [SplicePointOptimizer]
                                                        │
                                                        ▼
                                    [ProcessingParameterCalculator]
                                                        │
                                                        ▼
                                     [CrossfadeEnvelopeDesigner]
                                                        │
                                                        ▼
                                                CrossfadePlan
```

---

## Module Interface Specifications

### **BeatGridExtractor**:
```python
class BeatGridExtractor(nn.Module):
    def forward(self, audio: torch.Tensor) -> BeatGrid:
        """
        Extract beat grid from 30s audio segment.
        
        Args:
            audio: (samples,) mono audio at any sample rate
            
        Returns:
            BeatGrid with sample-accurate beat positions
        """
```

### **MusicalKeyExtractor**:
```python
class MusicalKeyExtractor(nn.Module):
    def forward(self, audio: torch.Tensor) -> KeyProfile:
        """
        Extract musical key with confidence and stability.
        
        Returns:
            KeyProfile(key='C', mode='major', confidence=0.85, modulations=[...])
        """
```

### **SplicePointOptimizer**:
```python
class SplicePointOptimizer(nn.Module):
    def forward(self, 
                track_a_analysis: TrackAnalysis,
                track_b_analysis: TrackAnalysis) -> Optional[OptimalSplice]:
        """
        Find globally optimal splice point or return None if impossible.
        
        Multi-objective optimization:
        - Maximize musical compatibility
        - Minimize processing artifacts  
        - Prefer "drop on the 1"
        - Respect phrase boundaries
        """
```

### **ProcessingParameterCalculator**:
```python
class ProcessingParameterCalculator(nn.Module):
    def forward(self, 
                optimal_splice: OptimalSplice,
                harmonic_match: HarmonicMatch,
                rhythmic_match: RhythmicMatch) -> ProcessingParams:
        """
        Calculate exact processing parameters for Track B.
        
        Returns:
            ProcessingParams(
                pitch_shift=2.0,     # +2 semitones
                rate_change=1.05,    # 5% faster
                artifact_prediction=0.15,  # Low artifacts expected
                quality_score=0.92   # High quality expected
            )
        """
```

---

## Success Criteria

### **Functional Requirements**:
- ✅ **Sample-accurate** beat and splice point detection
- ✅ **Reliable key detection** across genres (electronic, jazz, classical)
- ✅ **Robust optimization** - finds best solution or correctly rejects
- ✅ **Artifact prediction** - accurate quality assessment

### **Musical Requirements**:
- ✅ **"Drop on the 1"** - detects and prioritizes energy peaks on strong beats
- ✅ **Harmonic compatibility** - avoids dissonant key clashes
- ✅ **Energy continuity** - smooth energy flow across transition
- ✅ **Phrase respect** - doesn't cut mid-phrase unless necessary

### **Quality Requirements**:
- ✅ **Minimal processing** - only pitch/tempo adjust when necessary
- ✅ **Accurate predictions** - processing quality estimates match reality
- ✅ **Genre agnostic** - works across musical styles
- ✅ **Graceful failure** - rejects impossible splices cleanly

---

## Implementation Priority

### **Phase 1: Foundation Analysis**
1. **BeatGridExtractor** - Essential for everything
2. **MusicalKeyExtractor** - Core harmonic intelligence
3. **EnergyProfileExtractor** - Basic compatibility
4. **ExitPointAnalyzer** + **EntryPointAnalyzer** - Splice candidates

### **Phase 2: Compatibility Intelligence**
5. **HarmonicCompatibilityAnalyzer** - Key matching logic
6. **RhythmicCompatibilityAnalyzer** - Tempo matching logic
7. **EnergyCompatibilityAnalyzer** - Energy matching
8. **SplicePointOptimizer** - Core decision engine

### **Phase 3: Parameter Calculation**
9. **ProcessingParameterCalculator** - Exact B transformations
10. **CrossfadeEnvelopeDesigner** - Transition smoothing
11. **StructuralFeatureExtractor** - Advanced musical intelligence

This creates a **pure functional crossfade planning system** that takes two audio segments and outputs exact transformation parameters - completely independent of any user interaction or caching concerns.
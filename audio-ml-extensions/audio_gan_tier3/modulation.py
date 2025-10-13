"""
Modulation module for audio synthesis effects.

This module provides various modulation effects commonly used in audio
synthesis and processing, including amplitude modulation, frequency
modulation, and ring modulation effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, List


class AmplitudeModulation(nn.Module):
    """
    Amplitude Modulation (AM) effect.
    
    Modulates the amplitude of the input signal with a low-frequency
    oscillator (LFO), creating tremolo and related effects.
    
    Args:
        sample_rate: Audio sample rate
        mod_freq: Modulation frequency in Hz
        mod_depth: Modulation depth (0-1)
        mod_type: Modulation waveform ('sine', 'triangle', 'square', 'sawtooth')
        sync: Whether to sync modulation phase across batches
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        sample_rate: float = 22050,
        mod_freq: float = 5.0,
        mod_depth: float = 0.5,
        mod_type: str = 'sine',
        sync: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.sample_rate = sample_rate
        self.mod_freq = mod_freq
        self.mod_depth = mod_depth
        self.mod_type = mod_type
        self.sync = sync
        
        # Phase accumulator for modulation
        self.register_buffer('phase', torch.tensor(0.0))
    
    def forward(
        self,
        audio: torch.Tensor,
        mod_freq: Optional[float] = None,
        mod_depth: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply amplitude modulation.
        
        Args:
            audio: Input audio (batch, time) or (batch, channels, time)
            mod_freq: Override modulation frequency
            mod_depth: Override modulation depth
            
        Returns:
            Amplitude-modulated audio
        """
        if mod_freq is None:
            mod_freq = self.mod_freq
        if mod_depth is None:
            mod_depth = self.mod_depth
        
        # Handle input shape
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        batch_size, channels, length = audio.shape
        
        # Generate modulation signal
        mod_signal = self._generate_modulation(length, mod_freq, mod_depth)
        
        # Apply modulation
        modulated = audio * mod_signal.view(1, 1, -1)
        
        # Restore original shape
        if len(original_shape) == 2:
            modulated = modulated.squeeze(1)
        
        return modulated
    
    def _generate_modulation(
        self,
        length: int,
        mod_freq: float,
        mod_depth: float
    ) -> torch.Tensor:
        """Generate modulation waveform."""
        
        # Time vector
        t = torch.arange(length, dtype=torch.float32, device=self.phase.device) / self.sample_rate
        
        # Add current phase offset
        if not self.sync:
            t = t + self.phase
        
        # Generate modulation waveform
        if self.mod_type == 'sine':
            mod_wave = torch.sin(2 * np.pi * mod_freq * t)
        elif self.mod_type == 'triangle':
            mod_wave = 2 * torch.abs(2 * ((mod_freq * t) % 1.0) - 1) - 1
        elif self.mod_type == 'square':
            mod_wave = torch.sign(torch.sin(2 * np.pi * mod_freq * t))
        elif self.mod_type == 'sawtooth':
            mod_wave = 2 * ((mod_freq * t) % 1.0) - 1
        else:
            raise ValueError(f"Unknown modulation type: {self.mod_type}")
        
        # Apply depth and offset
        mod_signal = 1.0 + mod_depth * mod_wave
        
        # Update phase for next call (if not syncing)
        if not self.sync:
            phase_increment = 2 * np.pi * mod_freq * length / self.sample_rate
            self.phase = (self.phase + phase_increment) % (2 * np.pi)
        
        return mod_signal


class FrequencyModulation(nn.Module):
    """
    Frequency Modulation (FM) synthesis.
    
    Implements classic FM synthesis with carrier and modulator oscillators.
    Can be used for vibrato effects or full FM synthesis.
    
    Args:
        sample_rate: Audio sample rate
        carrier_freq: Carrier frequency in Hz
        mod_freq: Modulator frequency in Hz
        mod_index: Modulation index (depth)
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        sample_rate: float = 22050,
        carrier_freq: float = 440.0,
        mod_freq: float = 5.0,
        mod_index: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.mod_freq = mod_freq
        self.mod_index = mod_index
        
        # Phase accumulators
        self.register_buffer('carrier_phase', torch.tensor(0.0))
        self.register_buffer('mod_phase', torch.tensor(0.0))
    
    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        length: Optional[int] = None,
        carrier_freq: Optional[float] = None,
        mod_freq: Optional[float] = None,
        mod_index: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate or process audio with FM.
        
        Args:
            audio: Optional input audio for FM processing
            length: Length of audio to generate (if audio is None)
            carrier_freq: Override carrier frequency
            mod_freq: Override modulator frequency
            mod_index: Override modulation index
            
        Returns:
            FM synthesized or processed audio
        """
        if carrier_freq is None:
            carrier_freq = self.carrier_freq
        if mod_freq is None:
            mod_freq = self.mod_freq
        if mod_index is None:
            mod_index = self.mod_index
        
        if audio is None:
            # Generate FM synthesis
            if length is None:
                raise ValueError("Must provide either audio or length")
            
            batch_size = 1
            fm_audio = self._generate_fm(batch_size, length, carrier_freq, mod_freq, mod_index)
            return fm_audio.squeeze(0)
        else:
            # Apply FM as an effect (vibrato)
            return self._apply_fm_effect(audio, mod_freq, mod_index)
    
    def _generate_fm(
        self,
        batch_size: int,
        length: int,
        carrier_freq: float,
        mod_freq: float,
        mod_index: float
    ) -> torch.Tensor:
        """Generate FM synthesis."""
        
        # Time vector
        t = torch.arange(length, dtype=torch.float32, device=self.carrier_phase.device) / self.sample_rate
        
        # Modulator signal
        mod_signal = torch.sin(2 * np.pi * mod_freq * t + self.mod_phase)
        
        # Carrier with frequency modulation
        instantaneous_freq = carrier_freq + mod_index * mod_freq * mod_signal
        phase = torch.cumsum(2 * np.pi * instantaneous_freq / self.sample_rate, dim=0) + self.carrier_phase
        
        # Generate FM output
        fm_output = torch.sin(phase)
        
        # Update phases
        self.carrier_phase = (phase[-1] % (2 * np.pi)).detach()
        mod_phase_increment = 2 * np.pi * mod_freq * length / self.sample_rate
        self.mod_phase = (self.mod_phase + mod_phase_increment) % (2 * np.pi)
        
        return fm_output.unsqueeze(0).expand(batch_size, -1)
    
    def _apply_fm_effect(
        self,
        audio: torch.Tensor,
        mod_freq: float,
        mod_index: float
    ) -> torch.Tensor:
        """Apply FM as vibrato effect."""
        
        # Handle input shape
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        batch_size, channels, length = audio.shape
        
        # Generate modulation for pitch shifting
        t = torch.arange(length, dtype=torch.float32, device=audio.device) / self.sample_rate
        mod_signal = mod_index * torch.sin(2 * np.pi * mod_freq * t + self.mod_phase)
        
        # Apply as time-varying delay (simplified vibrato)
        delay_samples = mod_signal * self.sample_rate / 1000  # Convert to samples
        
        # Create fractional delay using interpolation
        output = torch.zeros_like(audio)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(length):
                    delay = delay_samples[i].item()
                    read_pos = i - delay
                    
                    if 0 <= read_pos < length - 1:
                        # Linear interpolation
                        pos_int = int(read_pos)
                        pos_frac = read_pos - pos_int
                        
                        val1 = audio[b, c, pos_int]
                        val2 = audio[b, c, pos_int + 1] if pos_int + 1 < length else 0
                        
                        output[b, c, i] = val1 * (1 - pos_frac) + val2 * pos_frac
        
        # Restore original shape
        if len(original_shape) == 2:
            output = output.squeeze(1)
        
        return output


class RingModulation(nn.Module):
    """
    Ring Modulation effect.
    
    Multiplies the input signal with a carrier wave, creating
    sidebands and inharmonic content for metallic sounds.
    
    Args:
        sample_rate: Audio sample rate
        carrier_freq: Carrier frequency in Hz
        mix: Dry/wet mix (0=dry, 1=wet)
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        sample_rate: float = 22050,
        carrier_freq: float = 440.0,
        mix: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.mix = mix
        
        # Phase accumulator
        self.register_buffer('phase', torch.tensor(0.0))
    
    def forward(
        self,
        audio: torch.Tensor,
        carrier_freq: Optional[float] = None,
        mix: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply ring modulation.
        
        Args:
            audio: Input audio
            carrier_freq: Override carrier frequency
            mix: Override mix amount
            
        Returns:
            Ring-modulated audio
        """
        if carrier_freq is None:
            carrier_freq = self.carrier_freq
        if mix is None:
            mix = self.mix
        
        # Handle input shape
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        batch_size, channels, length = audio.shape
        
        # Generate carrier wave
        t = torch.arange(length, dtype=torch.float32, device=audio.device) / self.sample_rate
        carrier = torch.sin(2 * np.pi * carrier_freq * t + self.phase)
        
        # Apply ring modulation
        modulated = audio * carrier.view(1, 1, -1)
        
        # Mix with dry signal
        output = audio * (1 - mix) + modulated * mix
        
        # Update phase
        phase_increment = 2 * np.pi * carrier_freq * length / self.sample_rate
        self.phase = (self.phase + phase_increment) % (2 * np.pi)
        
        # Restore original shape
        if len(original_shape) == 2:
            output = output.squeeze(1)
        
        return output


class Tremolo(nn.Module):
    """
    Tremolo effect (amplitude modulation with LFO).
    
    Classic guitar/synth effect that modulates volume rhythmically.
    
    Args:
        sample_rate: Audio sample rate
        rate: Tremolo rate in Hz
        depth: Tremolo depth (0-1)
        waveform: LFO waveform ('sine', 'triangle', 'square')
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        sample_rate: float = 22050,
        rate: float = 6.0,
        depth: float = 0.5,
        waveform: str = 'sine',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.am = AmplitudeModulation(
            sample_rate=sample_rate,
            mod_freq=rate,
            mod_depth=depth,
            mod_type=waveform,
            sync=False,
            **kwargs
        )
    
    def forward(
        self,
        audio: torch.Tensor,
        rate: Optional[float] = None,
        depth: Optional[float] = None
    ) -> torch.Tensor:
        """Apply tremolo effect."""
        return self.am(audio, mod_freq=rate, mod_depth=depth)


class Vibrato(nn.Module):
    """
    Vibrato effect (frequency modulation with LFO).
    
    Modulates pitch slightly for expressive musical effects.
    
    Args:
        sample_rate: Audio sample rate
        rate: Vibrato rate in Hz
        depth: Vibrato depth in cents (100 cents = 1 semitone)
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        sample_rate: float = 22050,
        rate: float = 5.0,
        depth: float = 50.0,  # cents
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.sample_rate = sample_rate
        self.rate = rate
        self.depth = depth
        
        # Phase accumulator
        self.register_buffer('phase', torch.tensor(0.0))
    
    def forward(
        self,
        audio: torch.Tensor,
        rate: Optional[float] = None,
        depth: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply vibrato effect.
        
        Args:
            audio: Input audio
            rate: Override vibrato rate
            depth: Override vibrato depth in cents
            
        Returns:
            Audio with vibrato applied
        """
        if rate is None:
            rate = self.rate
        if depth is None:
            depth = self.depth
        
        # Convert depth from cents to frequency ratio modulation depth
        mod_depth = depth / 1200.0  # Convert cents to octaves
        
        # Handle input shape
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        batch_size, channels, length = audio.shape
        
        # Generate vibrato LFO
        t = torch.arange(length, dtype=torch.float32, device=audio.device) / self.sample_rate
        lfo = torch.sin(2 * np.pi * rate * t + self.phase)
        
        # Convert to time-varying delay
        # Vibrato is typically implemented as a small time-varying delay
        max_delay_ms = 5.0  # Maximum delay in milliseconds
        delay_ms = max_delay_ms * mod_depth * lfo
        delay_samples = delay_ms * self.sample_rate / 1000.0
        
        # Apply fractional delay
        output = torch.zeros_like(audio)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(length):
                    delay = delay_samples[i].item()
                    read_pos = i - delay
                    
                    if 0 <= read_pos < length - 1:
                        # Linear interpolation
                        pos_int = int(read_pos)
                        pos_frac = read_pos - pos_int
                        
                        val1 = audio[b, c, pos_int]
                        val2 = audio[b, c, pos_int + 1] if pos_int + 1 < length else val1
                        
                        output[b, c, i] = val1 * (1 - pos_frac) + val2 * pos_frac
                    else:
                        output[b, c, i] = audio[b, c, i]
        
        # Update phase
        phase_increment = 2 * np.pi * rate * length / self.sample_rate
        self.phase = (self.phase + phase_increment) % (2 * np.pi)
        
        # Restore original shape
        if len(original_shape) == 2:
            output = output.squeeze(1)
        
        return output


class ModulationMatrix(nn.Module):
    """
    Modulation matrix for complex modulation routing.
    
    Allows multiple modulation sources to control multiple parameters
    with configurable routing and scaling.
    
    Args:
        num_sources: Number of modulation sources
        num_targets: Number of modulation targets
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_sources: int,
        num_targets: int,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_sources = num_sources
        self.num_targets = num_targets
        
        # Modulation matrix (learnable routing weights)
        self.mod_matrix = nn.Parameter(torch.zeros(num_sources, num_targets))
        
        # Source scaling
        self.source_scale = nn.Parameter(torch.ones(num_sources))
        
        # Target offset and scale
        self.target_offset = nn.Parameter(torch.zeros(num_targets))
        self.target_scale = nn.Parameter(torch.ones(num_targets))
    
    def forward(
        self,
        sources: torch.Tensor,
        base_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply modulation matrix.
        
        Args:
            sources: Modulation sources (batch, time, num_sources)
            base_values: Base values for targets (batch, time, num_targets)
            
        Returns:
            Modulated target values (batch, time, num_targets)
        """
        batch_size, time_steps, _ = sources.shape
        
        # Scale sources
        scaled_sources = sources * self.source_scale.view(1, 1, -1)
        
        # Apply modulation matrix
        modulation = torch.matmul(scaled_sources, self.mod_matrix)
        
        # Apply target scaling and offset
        modulation = modulation * self.target_scale.view(1, 1, -1) + self.target_offset.view(1, 1, -1)
        
        # Add to base values if provided
        if base_values is not None:
            modulation = modulation + base_values
        
        return modulation
    
    def set_routing(self, source_idx: int, target_idx: int, amount: float):
        """Set specific routing amount."""
        with torch.no_grad():
            self.mod_matrix[source_idx, target_idx] = amount
    
    def clear_routing(self):
        """Clear all routing."""
        with torch.no_grad():
            self.mod_matrix.zero_()


class LFOBank(nn.Module):
    """
    Bank of Low Frequency Oscillators.
    
    Provides multiple synchronized or independent LFOs for modulation.
    
    Args:
        num_lfos: Number of LFOs
        sample_rate: Audio sample rate
        sync: Whether to sync all LFOs
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_lfos: int = 4,
        sample_rate: float = 22050,
        sync: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_lfos = num_lfos
        self.sample_rate = sample_rate
        self.sync = sync
        
        # LFO parameters (learnable)
        self.frequencies = nn.Parameter(torch.ones(num_lfos) * 2.0)  # Default 2 Hz
        self.phases = nn.Parameter(torch.zeros(num_lfos))
        self.amplitudes = nn.Parameter(torch.ones(num_lfos))
        
        # Phase accumulators
        self.register_buffer('phase_accum', torch.zeros(num_lfos))
    
    def forward(self, length: int, waveform: str = 'sine') -> torch.Tensor:
        """
        Generate LFO bank output.
        
        Args:
            length: Number of samples to generate
            waveform: Waveform type for all LFOs
            
        Returns:
            LFO outputs (length, num_lfos)
        """
        # Time vector
        t = torch.arange(length, dtype=torch.float32, device=self.frequencies.device) / self.sample_rate
        
        # Generate all LFOs
        lfos = []
        for i in range(self.num_lfos):
            freq = self.frequencies[i]
            phase = self.phases[i] if self.sync else self.phase_accum[i]
            amp = self.amplitudes[i]
            
            # Generate waveform
            if waveform == 'sine':
                lfo = torch.sin(2 * np.pi * freq * t + phase)
            elif waveform == 'triangle':
                lfo = 2 * torch.abs(2 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1) - 1
            elif waveform == 'square':
                lfo = torch.sign(torch.sin(2 * np.pi * freq * t + phase))
            elif waveform == 'sawtooth':
                lfo = 2 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1
            else:
                raise ValueError(f"Unknown waveform: {waveform}")
            
            lfo = lfo * amp
            lfos.append(lfo)
            
            # Update phase accumulator
            if not self.sync:
                phase_increment = 2 * np.pi * freq * length / self.sample_rate
                self.phase_accum[i] = (self.phase_accum[i] + phase_increment) % (2 * np.pi)
        
        return torch.stack(lfos, dim=1)
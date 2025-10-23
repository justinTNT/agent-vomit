import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class AudioEventDetector(nn.Module):
    """
    Multi-label audio event detection with temporal localization.
    
    Designed for:
    - Environmental sound classification (ESC-50, AudioSet)
    - Music event detection (onset, offset, note events)
    - Speech event detection (voice activity, keywords)
    - Multi-label classification with temporal boundaries
    """
    
    def __init__(
        self,
        num_classes: int = 50,  # ESC-50 classes by default
        sample_rate: int = 16000,
        window_size: float = 1.0,  # seconds
        hop_size: float = 0.5,     # seconds
        n_mels: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_attention: bool = True,
        detection_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_samples = int(window_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)
        self.detection_threshold = detection_threshold
        
        # Audio preprocessing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2
        )
        
        # Feature extraction backbone
        self.feature_extractor = self._build_feature_extractor(
            n_mels, hidden_dim, num_layers
        )
        
        # Temporal modeling
        self.temporal_model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism for multi-scale temporal modeling
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # bidirectional LSTM
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.attention = None
            
        # Multi-scale feature fusion
        self.scale_fusion = nn.ModuleList([
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]  # Different temporal scales
        ])
        
        # Classification heads
        final_dim = hidden_dim * len(self.scale_fusion)
        
        # Frame-level classification
        self.frame_classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Segment-level classification (for short clips)
        self.segment_classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Event boundary detection
        self.boundary_detector = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # onset, offset, continuation
        )
        
    def _build_feature_extractor(self, input_dim: int, hidden_dim: int, num_layers: int):
        """Build CNN feature extraction layers."""
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim // (2 ** (num_layers - i - 1))
            
            layers.extend([
                nn.Conv2d(1 if i == 0 else current_dim, out_dim, 
                         kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),  # Pool only in frequency
                nn.Dropout2d(0.1)
            ])
            current_dim = out_dim
            
        return nn.Sequential(*layers)
        
    def extract_windows(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract overlapping time windows from waveform."""
        batch_size, audio_length = waveform.shape
        
        # Calculate number of windows
        num_windows = max(1, (audio_length - self.window_samples) // self.hop_samples + 1)
        
        # Extract windows
        windows = []
        for i in range(num_windows):
            start = i * self.hop_samples
            end = start + self.window_samples
            if end <= audio_length:
                windows.append(waveform[:, start:end])
            else:
                # Pad the last window if necessary
                padded_window = torch.zeros(batch_size, self.window_samples, device=waveform.device)
                remaining = audio_length - start
                padded_window[:, :remaining] = waveform[:, start:]
                windows.append(padded_window)
                
        if windows:
            return torch.stack(windows, dim=1)  # [batch, num_windows, window_samples]
        else:
            # Handle very short audio
            padded = torch.zeros(batch_size, self.window_samples, device=waveform.device)
            padded[:, :audio_length] = waveform
            return padded.unsqueeze(1)  # [batch, 1, window_samples]
        
    def forward(
        self, 
        waveform: torch.Tensor,
        return_features: bool = False,
        return_boundaries: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for audio event detection.
        
        Args:
            waveform: Input audio [batch, samples]
            return_features: Whether to return intermediate features
            return_boundaries: Whether to return event boundary predictions
            
        Returns:
            Dictionary with detection results
        """
        batch_size = waveform.shape[0]
        
        # Extract overlapping windows
        windows = self.extract_windows(waveform)  # [batch, num_windows, window_samples]
        num_windows = windows.shape[1]
        
        # Reshape for batch processing
        windows_flat = windows.view(-1, self.window_samples)
        
        # Extract mel-spectrograms for each window
        mel_specs = []
        for i in range(0, windows_flat.shape[0], batch_size):
            batch_windows = windows_flat[i:i+batch_size]
            mel_batch = self.mel_transform(batch_windows)
            mel_batch = torch.log(mel_batch.clamp(min=1e-8))
            mel_specs.append(mel_batch)
            
        mel_specs = torch.cat(mel_specs, dim=0)  # [batch*num_windows, n_mels, time]
        
        # CNN feature extraction
        mel_input = mel_specs.unsqueeze(1)  # Add channel dimension
        cnn_features = self.feature_extractor(mel_input)
        
        # Global average pooling over frequency
        cnn_features = F.adaptive_avg_pool2d(cnn_features, (1, cnn_features.size(-1)))
        cnn_features = cnn_features.squeeze(2)  # [batch*num_windows, hidden_dim, time]
        
        # Reshape back to batch format
        cnn_features = cnn_features.view(
            batch_size, num_windows, cnn_features.size(1), -1
        )
        
        # Average pool over time within each window
        window_features = cnn_features.mean(dim=-1)  # [batch, num_windows, hidden_dim]
        
        # Temporal modeling across windows
        lstm_out, _ = self.temporal_model(window_features)  # [batch, num_windows, hidden_dim*2]
        
        # Apply attention if available
        if self.attention is not None:
            attended_features, attention_weights = self.attention(
                lstm_out, lstm_out, lstm_out, need_weights=True
            )
            temporal_features = attended_features + lstm_out  # Residual connection
        else:
            temporal_features = lstm_out
            attention_weights = None
            
        # Multi-scale temporal convolutions
        # Transpose for 1D convolution: [batch, hidden_dim*2, num_windows]
        conv_input = temporal_features.transpose(1, 2)
        
        scale_features = []
        for conv in self.scale_fusion:
            scale_feat = F.relu(conv(conv_input))
            scale_features.append(scale_feat)
            
        # Concatenate multi-scale features
        fused_features = torch.cat(scale_features, dim=1)  # [batch, hidden_dim*4, num_windows]
        fused_features = fused_features.transpose(1, 2)    # [batch, num_windows, hidden_dim*4]
        
        # Frame-level predictions
        frame_logits = self.frame_classifier(fused_features)  # [batch, num_windows, num_classes]
        
        # Segment-level predictions (average over time)
        segment_features = fused_features.mean(dim=1)  # [batch, hidden_dim*4]
        segment_logits = self.segment_classifier(segment_features)  # [batch, num_classes]
        
        result = {
            'frame_logits': frame_logits,
            'segment_logits': segment_logits,
            'frame_probabilities': torch.sigmoid(frame_logits),
            'segment_probabilities': torch.sigmoid(segment_logits)
        }
        
        # Event boundary detection
        if return_boundaries:
            boundary_logits = self.boundary_detector(fused_features)
            result['boundary_logits'] = boundary_logits
            result['boundary_probabilities'] = F.softmax(boundary_logits, dim=-1)
            
        # Return features if requested
        if return_features:
            result['features'] = {
                'mel_spectrograms': mel_specs.view(batch_size, num_windows, *mel_specs.shape[1:]),
                'cnn_features': cnn_features,
                'temporal_features': temporal_features,
                'fused_features': fused_features
            }
            if attention_weights is not None:
                result['attention_weights'] = attention_weights
                
        return result
        
    def detect_events(
        self,
        waveform: torch.Tensor,
        class_names: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Detect audio events with temporal localization.
        
        Args:
            waveform: Input audio [batch, samples]
            class_names: List of class names for interpretation
            confidence_threshold: Detection threshold (uses model default if None)
            
        Returns:
            Dictionary with detected events and timing information
        """
        if confidence_threshold is None:
            confidence_threshold = self.detection_threshold
            
        with torch.no_grad():
            output = self.forward(waveform, return_boundaries=True)
            
        batch_size = waveform.shape[0]
        num_windows = output['frame_logits'].shape[1]
        
        # Convert to numpy for easier processing
        frame_probs = output['frame_probabilities'].cpu().numpy()
        segment_probs = output['segment_probabilities'].cpu().numpy()
        
        detections = []
        
        for b in range(batch_size):
            sample_detections = {
                'segment_events': [],
                'frame_events': [],
                'temporal_events': []
            }
            
            # Segment-level detection
            segment_confident = segment_probs[b] > confidence_threshold
            for class_idx in np.where(segment_confident)[0]:
                event = {
                    'class_idx': int(class_idx),
                    'confidence': float(segment_probs[b, class_idx]),
                    'start_time': 0.0,
                    'end_time': float(num_windows * self.hop_size)
                }
                if class_names:
                    event['class_name'] = class_names[class_idx]
                sample_detections['segment_events'].append(event)
                
            # Frame-level detection with temporal localization
            frame_confident = frame_probs[b] > confidence_threshold
            
            for class_idx in range(self.num_classes):
                if np.any(frame_confident[:, class_idx]):
                    # Find continuous regions
                    active_frames = np.where(frame_confident[:, class_idx])[0]
                    
                    if len(active_frames) > 0:
                        # Group consecutive frames
                        regions = []
                        start = active_frames[0]
                        end = active_frames[0]
                        
                        for frame in active_frames[1:]:
                            if frame == end + 1:
                                end = frame
                            else:
                                regions.append((start, end))
                                start = frame
                                end = frame
                        regions.append((start, end))
                        
                        # Convert frame indices to time
                        for start_frame, end_frame in regions:
                            start_time = start_frame * self.hop_size
                            end_time = (end_frame + 1) * self.hop_size
                            avg_confidence = float(
                                frame_probs[b, start_frame:end_frame+1, class_idx].mean()
                            )
                            
                            event = {
                                'class_idx': int(class_idx),
                                'confidence': avg_confidence,
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': end_time - start_time
                            }
                            if class_names:
                                event['class_name'] = class_names[class_idx]
                            sample_detections['frame_events'].append(event)
                            
            detections.append(sample_detections)
            
        return {
            'detections': detections,
            'window_times': [i * self.hop_size for i in range(num_windows)],
            'total_duration': num_windows * self.hop_size
        }
        
    def get_activation_timeline(
        self,
        waveform: torch.Tensor,
        class_idx: int,
        smoothing_window: int = 3
    ) -> Dict[str, np.ndarray]:
        """Get temporal activation timeline for a specific class."""
        with torch.no_grad():
            output = self.forward(waveform)
            
        # Extract activations for the specified class
        frame_activations = output['frame_probabilities'][0, :, class_idx].cpu().numpy()
        
        # Apply smoothing
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            frame_activations = np.convolve(frame_activations, kernel, mode='same')
            
        # Create time axis
        num_frames = len(frame_activations)
        time_axis = np.array([i * self.hop_size for i in range(num_frames)])
        
        return {
            'time': time_axis,
            'activations': frame_activations,
            'peaks': self._find_peaks(frame_activations, time_axis)
        }
        
    def _find_peaks(self, activations: np.ndarray, time_axis: np.ndarray) -> List[Dict]:
        """Find peaks in activation timeline."""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(
            activations,
            height=self.detection_threshold,
            distance=int(0.5 / self.hop_size)  # Minimum 0.5s between peaks
        )
        
        peak_info = []
        for i, peak_idx in enumerate(peaks):
            peak_info.append({
                'time': time_axis[peak_idx],
                'activation': activations[peak_idx],
                'prominence': properties.get('prominences', [0])[i] if 'prominences' in properties else 0
            })
            
        return peak_info


# Pre-defined event taxonomies
ESC50_CLASSES = [
    'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops',
    'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing',
    'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth',
    'snoring', 'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing',
    'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
    'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn',
    'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
]

MUSIC_EVENT_CLASSES = [
    'note_onset', 'note_offset', 'chord_change', 'beat', 'downbeat', 'phrase_boundary',
    'tempo_change', 'key_change', 'dynamics_change', 'silence', 'applause', 'crowd_noise'
]

SPEECH_EVENT_CLASSES = [
    'speech', 'silence', 'music', 'noise', 'laughter', 'applause', 'cough', 'breath',
    'mouth_sound', 'background_noise', 'overlapping_speech', 'phone_ring'
]


# Factory functions
def create_esc50_detector(**kwargs) -> AudioEventDetector:
    """Create ESC-50 environmental sound detector."""
    return AudioEventDetector(
        num_classes=len(ESC50_CLASSES),
        sample_rate=16000,
        **kwargs
    )

def create_music_event_detector(**kwargs) -> AudioEventDetector:
    """Create music event detector."""
    return AudioEventDetector(
        num_classes=len(MUSIC_EVENT_CLASSES),
        sample_rate=22050,
        window_size=0.5,  # Shorter windows for music events
        hop_size=0.1,
        **kwargs
    )

def create_speech_event_detector(**kwargs) -> AudioEventDetector:
    """Create speech event detector."""
    return AudioEventDetector(
        num_classes=len(SPEECH_EVENT_CLASSES),
        sample_rate=16000,
        window_size=1.0,
        hop_size=0.25,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create detector
    detector = create_esc50_detector()
    
    # Test with dummy audio
    waveform = torch.randn(1, 16000 * 5)  # 5 seconds
    
    # Forward pass
    output = detector(waveform, return_features=True, return_boundaries=True)
    
    print(f"Frame logits shape: {output['frame_logits'].shape}")
    print(f"Segment logits shape: {output['segment_logits'].shape}")
    print(f"Boundary logits shape: {output['boundary_logits'].shape}")
    
    # Detect events
    detections = detector.detect_events(waveform, class_names=ESC50_CLASSES)
    
    print(f"Number of segment events: {len(detections['detections'][0]['segment_events'])}")
    print(f"Number of frame events: {len(detections['detections'][0]['frame_events'])}")
    
    # Get activation timeline for 'dog' class
    timeline = detector.get_activation_timeline(waveform, class_idx=0)
    print(f"Timeline length: {len(timeline['time'])}")
    print(f"Number of peaks: {len(timeline['peaks'])}")
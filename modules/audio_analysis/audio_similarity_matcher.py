import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class AudioSimilarityMatcher(nn.Module):
    """
    Multi-scale audio similarity matching for retrieval and comparison.
    
    Supports:
    - Content-based similarity (spectral, timbral, harmonic)
    - Perceptual similarity using learned embeddings
    - Multi-modal similarity (audio + metadata)
    - Temporal alignment and DTW-based matching
    - Scale-invariant similarity (tempo/pitch independent)
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        embedding_dim: int = 512,
        n_mels: int = 128,
        n_chroma: int = 12,
        similarity_metrics: List[str] = ['spectral', 'harmonic', 'temporal', 'learned'],
        pooling_strategy: str = 'attention',  # 'mean', 'max', 'attention'
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.similarity_metrics = similarity_metrics
        self.pooling_strategy = pooling_strategy
        self.temperature = temperature
        
        # Feature extractors
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': n_mels}
        )
        
        # Chroma features
        self.chroma_transform = self._build_chroma_transform(n_chroma)
        
        # Learned embedding network
        if 'learned' in similarity_metrics:
            self.embedding_network = self._build_embedding_network(
                n_mels, embedding_dim
            )
            
        # Attention pooling
        if pooling_strategy == 'attention':
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=8,
                batch_first=True
            )
            
        # Similarity fusion network
        self.similarity_fusion = self._build_fusion_network()
        
        # Distance metrics
        self.distance_functions = {
            'cosine': self._cosine_distance,
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'learned': self._learned_distance
        }
        
    def _build_chroma_transform(self, n_chroma: int):
        """Build chroma feature transform."""
        import librosa
        chroma_fb = librosa.filters.chroma(
            sr=self.sample_rate,
            n_fft=2048,
            n_chroma=n_chroma
        )
        chroma_fb = torch.from_numpy(chroma_fb).float()
        self.register_buffer('chroma_filterbank', chroma_fb)
        
        return lambda spec: torch.matmul(self.chroma_filterbank, spec)
        
    def _build_embedding_network(self, input_dim: int, output_dim: int):
        """Build learned embedding network."""
        return nn.Sequential(
            # Convolutional feature extraction
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            # Dense layers
            nn.Flatten(),
            nn.Linear(256 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.L2Norm(dim=1)  # L2 normalize embeddings
        )
        
    def _build_fusion_network(self):
        """Build network to fuse multiple similarity scores."""
        n_metrics = len(self.similarity_metrics)
        return nn.Sequential(
            nn.Linear(n_metrics, n_metrics * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_metrics * 2, 1),
            nn.Sigmoid()
        )
        
    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features for similarity comparison."""
        # Ensure mono audio
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
            
        # Basic spectral features
        stft = torch.stft(
            waveform, n_fft=2048, hop_length=512, 
            return_complex=True, window=torch.hann_window(2048, device=waveform.device)
        )
        magnitude = torch.abs(stft)
        
        # Mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-8)
        
        # MFCCs
        mfcc = self.mfcc_transform(waveform)
        
        # Chroma
        chroma = self.chroma_transform(magnitude)
        chroma = F.normalize(chroma, p=2, dim=1)
        
        features = {
            'magnitude': magnitude,
            'mel_spectrogram': mel_spec_db,
            'mfcc': mfcc,
            'chroma': chroma
        }
        
        # Learned embeddings
        if 'learned' in self.similarity_metrics:
            mel_input = mel_spec_db.unsqueeze(1)  # Add channel dimension
            learned_embedding = self.embedding_network(mel_input)
            features['learned_embedding'] = learned_embedding
            
        return features
        
    def compute_spectral_similarity(
        self, 
        features1: Dict[str, torch.Tensor], 
        features2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute spectral similarity using mel-spectrograms and MFCCs."""
        # Mel-spectrogram similarity
        mel1 = features1['mel_spectrogram']
        mel2 = features2['mel_spectrogram']
        
        # Pool temporal dimension
        mel1_pooled = self._pool_temporal(mel1)
        mel2_pooled = self._pool_temporal(mel2)
        
        mel_similarity = F.cosine_similarity(mel1_pooled, mel2_pooled, dim=1)
        
        # MFCC similarity
        mfcc1 = features1['mfcc']
        mfcc2 = features2['mfcc']
        
        mfcc1_pooled = self._pool_temporal(mfcc1)
        mfcc2_pooled = self._pool_temporal(mfcc2)
        
        mfcc_similarity = F.cosine_similarity(mfcc1_pooled, mfcc2_pooled, dim=1)
        
        # Combine spectral similarities
        spectral_similarity = (mel_similarity + mfcc_similarity) / 2
        
        return spectral_similarity
        
    def compute_harmonic_similarity(
        self, 
        features1: Dict[str, torch.Tensor], 
        features2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute harmonic similarity using chroma features."""
        chroma1 = features1['chroma']
        chroma2 = features2['chroma']
        
        # Pool temporal dimension
        chroma1_pooled = self._pool_temporal(chroma1)
        chroma2_pooled = self._pool_temporal(chroma2)
        
        # Cosine similarity in chroma space
        harmonic_similarity = F.cosine_similarity(chroma1_pooled, chroma2_pooled, dim=1)
        
        return harmonic_similarity
        
    def compute_temporal_similarity(
        self, 
        features1: Dict[str, torch.Tensor], 
        features2: Dict[str, torch.Tensor],
        use_dtw: bool = False
    ) -> torch.Tensor:
        """Compute temporal similarity considering temporal dynamics."""
        # Use mel-spectrograms for temporal analysis
        mel1 = features1['mel_spectrogram']  # [batch, n_mels, time]
        mel2 = features2['mel_spectrogram']
        
        if use_dtw:
            # Dynamic Time Warping similarity (simplified)
            temporal_similarity = self._compute_dtw_similarity(mel1, mel2)
        else:
            # Cross-correlation based similarity
            temporal_similarity = self._compute_correlation_similarity(mel1, mel2)
            
        return temporal_similarity
        
    def compute_learned_similarity(
        self, 
        features1: Dict[str, torch.Tensor], 
        features2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute similarity using learned embeddings."""
        emb1 = features1['learned_embedding']
        emb2 = features2['learned_embedding']
        
        # Cosine similarity in learned space
        learned_similarity = F.cosine_similarity(emb1, emb2, dim=1)
        
        return learned_similarity
        
    def _pool_temporal(self, features: torch.Tensor) -> torch.Tensor:
        """Pool features across temporal dimension."""
        if self.pooling_strategy == 'mean':
            return features.mean(dim=-1)
        elif self.pooling_strategy == 'max':
            return features.max(dim=-1)[0]
        elif self.pooling_strategy == 'attention':
            # Reshape for attention: [batch, time, features]
            batch_size, n_features, time_steps = features.shape
            features_reshaped = features.transpose(1, 2)  # [batch, time, features]
            
            # Project to embedding dimension if needed
            if n_features != self.embedding_dim:
                features_reshaped = features_reshaped.reshape(-1, n_features)
                proj = nn.Linear(n_features, self.embedding_dim).to(features.device)
                features_reshaped = proj(features_reshaped)
                features_reshaped = features_reshaped.reshape(batch_size, time_steps, self.embedding_dim)
            
            # Apply attention pooling
            attended, _ = self.attention_pooling(
                features_reshaped, features_reshaped, features_reshaped
            )
            
            # Pool attended features
            pooled = attended.mean(dim=1)  # [batch, embedding_dim]
            return pooled
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
    def _compute_dtw_similarity(self, seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
        """Compute DTW-based similarity (simplified version)."""
        batch_size = seq1.shape[0]
        similarities = []
        
        for i in range(batch_size):
            s1 = seq1[i].transpose(0, 1)  # [time, features]
            s2 = seq2[i].transpose(0, 1)
            
            # Simplified DTW using cosine distance
            distance_matrix = 1 - F.cosine_similarity(
                s1.unsqueeze(1), s2.unsqueeze(0), dim=2
            )
            
            # Simplified DTW path (diagonal only for efficiency)
            min_len = min(s1.shape[0], s2.shape[0])
            dtw_distance = torch.diagonal(distance_matrix[:min_len, :min_len]).mean()
            dtw_similarity = 1 - dtw_distance
            
            similarities.append(dtw_similarity)
            
        return torch.stack(similarities)
        
    def _compute_correlation_similarity(self, seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
        """Compute cross-correlation based similarity."""
        # Pool frequency dimension first
        seq1_pooled = seq1.mean(dim=1)  # [batch, time]
        seq2_pooled = seq2.mean(dim=1)
        
        # Normalize sequences
        seq1_norm = F.normalize(seq1_pooled, p=2, dim=1)
        seq2_norm = F.normalize(seq2_pooled, p=2, dim=1)
        
        # Compute cross-correlation
        batch_size = seq1.shape[0]
        correlations = []
        
        for i in range(batch_size):
            correlation = F.conv1d(
                seq1_norm[i:i+1].unsqueeze(0),  # [1, 1, time]
                seq2_norm[i:i+1].unsqueeze(0).flip(-1),  # [1, 1, time] flipped
                padding=seq2_norm.shape[1] - 1
            ).squeeze()
            
            max_correlation = torch.max(correlation)
            correlations.append(max_correlation)
            
        return torch.stack(correlations)
        
    def forward(
        self, 
        waveform1: torch.Tensor, 
        waveform2: torch.Tensor,
        return_individual_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale audio similarity.
        
        Args:
            waveform1: First audio batch [batch, samples]
            waveform2: Second audio batch [batch, samples]
            return_individual_scores: Whether to return individual metric scores
            
        Returns:
            Dictionary with similarity scores
        """
        # Extract features
        features1 = self.extract_features(waveform1)
        features2 = self.extract_features(waveform2)
        
        # Compute individual similarities
        similarities = {}
        
        if 'spectral' in self.similarity_metrics:
            similarities['spectral'] = self.compute_spectral_similarity(features1, features2)
            
        if 'harmonic' in self.similarity_metrics:
            similarities['harmonic'] = self.compute_harmonic_similarity(features1, features2)
            
        if 'temporal' in self.similarity_metrics:
            similarities['temporal'] = self.compute_temporal_similarity(features1, features2)
            
        if 'learned' in self.similarity_metrics:
            similarities['learned'] = self.compute_learned_similarity(features1, features2)
            
        # Fuse similarities
        if len(similarities) > 1:
            # Stack similarities for fusion
            similarity_stack = torch.stack(list(similarities.values()), dim=1)
            fused_similarity = self.similarity_fusion(similarity_stack).squeeze(-1)
        else:
            fused_similarity = list(similarities.values())[0]
            
        result = {
            'similarity': fused_similarity,
            'features1': features1,
            'features2': features2
        }
        
        if return_individual_scores:
            result['individual_scores'] = similarities
            
        return result
        
    def batch_similarity_matrix(
        self, 
        waveforms: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pairwise similarity matrix for a batch of audio samples.
        
        Args:
            waveforms: Audio batch [batch, samples]
            return_features: Whether to return extracted features
            
        Returns:
            Dictionary with similarity matrix and optionally features
        """
        batch_size = waveforms.shape[0]
        
        # Extract features for all samples
        all_features = self.extract_features(waveforms)
        
        # Compute pairwise similarities
        similarity_matrix = torch.zeros(batch_size, batch_size, device=waveforms.device)
        
        for i in range(batch_size):
            for j in range(i, batch_size):
                # Extract features for pair
                features_i = {k: v[i:i+1] for k, v in all_features.items()}
                features_j = {k: v[j:j+1] for k, v in all_features.items()}
                
                # Compute similarity
                similarities = {}
                
                if 'spectral' in self.similarity_metrics:
                    similarities['spectral'] = self.compute_spectral_similarity(features_i, features_j)
                    
                if 'harmonic' in self.similarity_metrics:
                    similarities['harmonic'] = self.compute_harmonic_similarity(features_i, features_j)
                    
                if 'temporal' in self.similarity_metrics:
                    similarities['temporal'] = self.compute_temporal_similarity(features_i, features_j)
                    
                if 'learned' in self.similarity_metrics:
                    similarities['learned'] = self.compute_learned_similarity(features_i, features_j)
                
                # Fuse similarities
                if len(similarities) > 1:
                    similarity_stack = torch.stack(list(similarities.values()), dim=1)
                    fused_similarity = self.similarity_fusion(similarity_stack).squeeze(-1)
                else:
                    fused_similarity = list(similarities.values())[0]
                
                similarity_matrix[i, j] = fused_similarity.item()
                similarity_matrix[j, i] = fused_similarity.item()  # Symmetric
                
        result = {'similarity_matrix': similarity_matrix}
        
        if return_features:
            result['features'] = all_features
            
        return result
        
    def find_most_similar(
        self, 
        query_waveform: torch.Tensor, 
        database_waveforms: torch.Tensor,
        top_k: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Find most similar audio samples in database.
        
        Args:
            query_waveform: Query audio [1, samples]
            database_waveforms: Database audio [n_samples, samples]
            top_k: Number of most similar samples to return
            
        Returns:
            Dictionary with indices and similarity scores of top matches
        """
        n_database = database_waveforms.shape[0]
        
        # Compute similarities with all database samples
        similarities = []
        
        for i in range(n_database):
            db_sample = database_waveforms[i:i+1]
            similarity_result = self.forward(query_waveform, db_sample)
            similarities.append(similarity_result['similarity'])
            
        similarities = torch.cat(similarities)
        
        # Find top-k most similar
        top_similarities, top_indices = torch.topk(similarities, k=min(top_k, n_database))
        
        return {
            'top_indices': top_indices,
            'top_similarities': top_similarities,
            'all_similarities': similarities
        }
        
    # Distance metric functions
    def _cosine_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(x, y, dim=1)
        
    def _euclidean_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x - y, p=2, dim=1)
        
    def _manhattan_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x - y, p=1, dim=1)
        
    def _learned_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Use learned embedding network for distance computation
        combined = torch.cat([x, y, torch.abs(x - y)], dim=1)
        
        # Simple learned distance network
        distance_net = nn.Sequential(
            nn.Linear(combined.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(x.device)
        
        return distance_net(combined).squeeze(-1)


class L2Norm(nn.Module):
    """L2 normalization layer."""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim)


# Factory functions for different use cases
def create_spectral_similarity_matcher(**kwargs) -> AudioSimilarityMatcher:
    """Create matcher focused on spectral similarity."""
    return AudioSimilarityMatcher(
        similarity_metrics=['spectral'],
        **kwargs
    )

def create_harmonic_similarity_matcher(**kwargs) -> AudioSimilarityMatcher:
    """Create matcher focused on harmonic similarity."""
    return AudioSimilarityMatcher(
        similarity_metrics=['harmonic'],
        **kwargs
    )

def create_comprehensive_similarity_matcher(**kwargs) -> AudioSimilarityMatcher:
    """Create matcher using all similarity metrics."""
    return AudioSimilarityMatcher(
        similarity_metrics=['spectral', 'harmonic', 'temporal', 'learned'],
        **kwargs
    )

def create_learned_similarity_matcher(**kwargs) -> AudioSimilarityMatcher:
    """Create matcher using only learned embeddings."""
    return AudioSimilarityMatcher(
        similarity_metrics=['learned'],
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create similarity matcher
    matcher = create_comprehensive_similarity_matcher()
    
    # Test with dummy audio
    batch_size = 4
    audio_length = 22050 * 5  # 5 seconds
    
    waveform1 = torch.randn(batch_size, audio_length)
    waveform2 = torch.randn(batch_size, audio_length)
    
    # Compute pairwise similarity
    result = matcher(waveform1, waveform2, return_individual_scores=True)
    
    print(f"Similarity scores: {result['similarity']}")
    print(f"Individual scores: {list(result['individual_scores'].keys())}")
    
    # Compute similarity matrix
    matrix_result = matcher.batch_similarity_matrix(waveform1[:3])
    print(f"Similarity matrix shape: {matrix_result['similarity_matrix'].shape}")
    
    # Find most similar
    query = waveform1[:1]
    database = waveform1[1:]
    
    similar_result = matcher.find_most_similar(query, database, top_k=2)
    print(f"Top similar indices: {similar_result['top_indices']}")
    print(f"Top similarities: {similar_result['top_similarities']}")
"""Collection of PyTorch modules recreated per guidelines_v3."""

from .adaptive_computation import AdaptiveComputation
from .anti_aliased_conv import AntiAliasedConv
from .attention_decoder import AttentionDecoder
from .autoencoder import AutoEncoder
from .causal_conv1d import CausalConv1d
from .contrastive_learner import ContrastiveLearner
from .conv_encoder import ConvEncoder
from .cross_modal_fusion import CrossModalFusion
from .data_sampler import DataSampler
from .data_validator import DataValidator
from .data_versioner import DataVersioner
from .feature_store import FeatureStore
from .graph_encoder import GraphEncoder
from .memory_bank import MemoryBank
from .multi_scale_stft_loss import MultiScaleSTFTLoss
from .residual_vector_quantizer import ResidualVectorQuantizer
from .sequence_encoder import SequenceEncoder
from .sequence_to_sequence_model import SequenceToSequenceModel
from .set_encoder import SetEncoder
from .snake_activation import SnakeActivation
from .stream_joiner import StreamJoiner
from .stream_processor import StreamProcessor
from .time_series_encoder import TimeSeriesEncoder
from .transformer_block import TransformerBlock
from .vit_patch_encoder import ViTPatchEncoder

__all__ = [
    "AdaptiveComputation",
    "AntiAliasedConv",
    "AttentionDecoder",
    "AutoEncoder",
    "CausalConv1d",
    "ContrastiveLearner",
    "ConvEncoder",
    "CrossModalFusion",
    "DataSampler",
    "DataValidator",
    "DataVersioner",
    "FeatureStore",
    "GraphEncoder",
    "MemoryBank",
    "MultiScaleSTFTLoss",
    "ResidualVectorQuantizer",
    "SequenceEncoder",
    "SequenceToSequenceModel",
    "SetEncoder",
    "SnakeActivation",
    "StreamJoiner",
    "StreamProcessor",
    "TimeSeriesEncoder",
    "TransformerBlock",
    "ViTPatchEncoder",
]

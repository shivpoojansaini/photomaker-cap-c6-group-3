from .model import PhotoMakerIDEncoder
from .model_v2 import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from .model_v2 import MultiIdentityFuseModule, PhotoMakerIDEncoder_MultiIdentity
from .resampler import FacePerceiverResampler
from .pipeline import PhotoMakerStableDiffusionXLPipeline
from .pipeline_controlnet import PhotoMakerStableDiffusionXLControlNetPipeline
from .pipeline_t2i_adapter import PhotoMakerStableDiffusionXLAdapterPipeline
from .pipeline_multi_identity import PhotoMakerMultiIdentityPipeline

# InsightFace Package
from .insightface_package import FaceAnalysis2, analyze_faces

# Multi-identity support
from .prompt_parser import MultiIdentityPromptParser, parse_multi_identity_prompt
from .spatial_masks import SpatialMaskGenerator, RegionalMasks, create_masks_for_generation
from .regional_attention import (
    RegionalAttentionConfig,
    RegionalAttnProcessor2_0,
    set_regional_attention_processors,
    RegionalAttentionContext,
)

__all__ = [
    # Face detection
    "FaceAnalysis2",
    "analyze_faces",
    # Core models
    "FacePerceiverResampler",
    "PhotoMakerIDEncoder",
    "PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken",
    "MultiIdentityFuseModule",
    "PhotoMakerIDEncoder_MultiIdentity",
    # Pipelines
    "PhotoMakerStableDiffusionXLPipeline",
    "PhotoMakerStableDiffusionXLControlNetPipeline",
    "PhotoMakerStableDiffusionXLAdapterPipeline",
    "PhotoMakerMultiIdentityPipeline",
    # Multi-identity support
    "MultiIdentityPromptParser",
    "parse_multi_identity_prompt",
    "SpatialMaskGenerator",
    "RegionalMasks",
    "create_masks_for_generation",
    "RegionalAttentionConfig",
    "RegionalAttnProcessor2_0",
    "set_regional_attention_processors",
    "RegionalAttentionContext",
]
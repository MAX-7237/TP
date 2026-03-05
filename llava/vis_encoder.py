"""
Vision Encoder wrapper module.
Provides build_vision_encoder function that uses the multimodal encoder.
"""
import os
from .model.multimodal_encoder.builder import build_vision_tower


def build_vision_encoder(config, **kwargs):
    """
    Build vision encoder from config.
    
    Args:
        config: Model config with vision tower settings
        **kwargs: Additional arguments
        
    Returns:
        Vision encoder model
    """
    vision_tower_cfg = config
    return build_vision_tower(vision_tower_cfg, **kwargs)

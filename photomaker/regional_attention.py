# Regional attention processor for multi-identity generation
# Custom attention processor that applies different ID embeddings to different spatial regions

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any


@dataclass
class RegionalAttentionConfig:
    """Configuration for regional attention"""
    num_identities: int
    tokens_per_identity: int      # Number of ID tokens per identity (2 for PhotoMaker v2)
    shared_token_count: int       # Usually 77 for CLIP
    enable_regional: bool = True
    shared_attention_weight: float = 0.5  # Weight for shared tokens contribution


class RegionalAttnProcessor2_0:
    """
    Custom attention processor for multi-identity regional generation.

    Replaces the default AttnProcessor2_0 to enable:
    - Standard attention for self-attention layers
    - Regional masked attention for cross-attention layers

    Token layout in encoder_hidden_states:
    [shared_prompt (77)] + [id1_tokens (N)] + [id2_tokens (N)] + ...
    """

    def __init__(
        self,
        regional_config: Optional[RegionalAttentionConfig] = None,
    ):
        self.regional_config = regional_config

    def __call__(
        self,
        attn,  # Attention module from diffusers
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # Custom kwargs for regional attention
        regional_masks: Optional[torch.Tensor] = None,
        identity_token_ranges: Optional[Dict[int, Tuple[int, int]]] = None,
        scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with regional attention support.

        Args:
            attn: Attention module
            hidden_states: Query features from UNet (B, H*W, C)
            encoder_hidden_states: Key/Value features (B, seq_len, C)
            attention_mask: Optional attention mask
            temb: Optional timestep embedding
            regional_masks: Spatial masks (num_identities, H_latent, W_latent)
            identity_token_ranges: {identity_idx: (start_token, end_token)}
            scale: LoRA scale factor
        """
        residual = hidden_states

        # Handle spatial norm if present
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape
            # Infer spatial dimensions
            height = width = int(sequence_length ** 0.5)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

        # Group norm if present
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Determine attention type
        is_cross_attention = encoder_hidden_states is not None

        # Check if we should apply regional attention
        should_apply_regional = (
            is_cross_attention and
            regional_masks is not None and
            self.regional_config is not None and
            self.regional_config.enable_regional and
            identity_token_ranges is not None
        )

        if should_apply_regional:
            hidden_states = self._regional_cross_attention(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                regional_masks=regional_masks,
                identity_token_ranges=identity_token_ranges,
                height=height,
                width=width,
                batch_size=batch_size,
                attention_mask=attention_mask,
            )
        else:
            # Standard attention
            hidden_states = self._standard_attention(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # Residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _regional_cross_attention(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        regional_masks: torch.Tensor,
        identity_token_ranges: Dict[int, Tuple[int, int]],
        height: int,
        width: int,
        batch_size: int,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute cross-attention with regional masking.

        Strategy:
        1. Shared tokens (prompt) contribute to entire image
        2. Identity-specific tokens contribute only to their regions
        3. Blend results using spatial masks
        """
        cfg = self.regional_config
        num_identities = cfg.num_identities
        shared_end = cfg.shared_token_count

        # Project queries
        query = attn.to_q(hidden_states)

        # Resize masks to current resolution
        if regional_masks.shape[-2:] != (height, width):
            regional_masks_resized = F.interpolate(
                regional_masks.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            regional_masks_resized = regional_masks

        # Flatten masks: (num_ids, h, w) -> (num_ids, h*w)
        masks_flat = regional_masks_resized.view(num_identities, -1)

        # --- Compute attention for shared tokens ---
        shared_tokens = encoder_hidden_states[:, :shared_end, :]
        key_shared = attn.to_k(shared_tokens)
        value_shared = attn.to_v(shared_tokens)

        attn_output_shared = self._scaled_dot_product_attention(
            query, key_shared, value_shared, attn, attention_mask
        )

        # --- Compute attention for each identity's tokens ---
        identity_outputs = []
        for identity_idx in range(num_identities):
            if identity_idx in identity_token_ranges:
                start_tok, end_tok = identity_token_ranges[identity_idx]
            else:
                # Default: evenly divided after shared tokens
                tokens_per_id = cfg.tokens_per_identity
                start_tok = shared_end + identity_idx * tokens_per_id
                end_tok = start_tok + tokens_per_id

            # Handle case where token range is out of bounds
            if start_tok >= encoder_hidden_states.shape[1]:
                # Use shared tokens as fallback
                id_tokens = shared_tokens
            else:
                end_tok = min(end_tok, encoder_hidden_states.shape[1])
                id_tokens = encoder_hidden_states[:, start_tok:end_tok, :]

            key_id = attn.to_k(id_tokens)
            value_id = attn.to_v(id_tokens)

            attn_output_id = self._scaled_dot_product_attention(
                query, key_id, value_id, attn, attention_mask
            )
            identity_outputs.append(attn_output_id)

        # --- Blend outputs using regional masks ---
        shared_weight = cfg.shared_attention_weight

        # Start with shared contribution
        blended_output = attn_output_shared * shared_weight

        # Add identity-specific contributions weighted by masks
        for identity_idx, id_output in enumerate(identity_outputs):
            # Get mask for this identity: (h*w,)
            mask = masks_flat[identity_idx]

            # Expand mask for broadcasting: (batch, h*w, 1)
            # mask shape: (h*w,) -> unsqueeze(0): (1, h*w) -> unsqueeze(-1): (1, h*w, 1)
            # -> expand: (batch_size, h*w, 1)
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
            mask_expanded = mask_expanded.to(id_output.dtype)

            # Weight identity contribution by mask
            identity_weight = (1.0 - shared_weight) / num_identities
            blended_output = blended_output + id_output * mask_expanded * identity_weight * 2

        return blended_output

    def _standard_attention(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard attention computation (fallback)"""
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        return self._scaled_dot_product_attention(
            query, key, value, attn, attention_mask
        )

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Efficient attention using PyTorch 2.0 SDPA"""
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape for multi-head attention
        query = query.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(key.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        # Use scaled dot product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(
            hidden_states.shape[0], -1, attn.heads * head_dim
        )

        return hidden_states


def set_regional_attention_processors(
    unet,
    regional_config: RegionalAttentionConfig,
) -> Dict[str, RegionalAttnProcessor2_0]:
    """
    Replace all cross-attention processors in UNet with regional-aware versions.

    Args:
        unet: UNet model from diffusers
        regional_config: Configuration for regional attention

    Returns:
        Dict mapping processor names to processors (for restoration)
    """
    attn_procs = {}

    for name in unet.attn_processors.keys():
        # Apply regional processing only to cross-attention layers (attn2)
        if "attn2" in name:
            attn_procs[name] = RegionalAttnProcessor2_0(regional_config)
        else:
            # Self-attention uses standard processing
            attn_procs[name] = RegionalAttnProcessor2_0(None)

    unet.set_attn_processor(attn_procs)
    return attn_procs


def restore_attention_processors(
    unet,
    original_processors: Dict[str, Any]
):
    """
    Restore original attention processors.

    Args:
        unet: UNet model
        original_processors: Original processors from before regional setup
    """
    unet.set_attn_processor(original_processors)


class RegionalAttentionContext:
    """
    Context manager for regional attention.

    Usage:
        with RegionalAttentionContext(unet, config) as ctx:
            # Generate with regional attention
            pass
        # Original processors restored automatically
    """

    def __init__(
        self,
        unet,
        num_identities: int,
        tokens_per_identity: int = 2,
        shared_token_count: int = 77,
    ):
        self.unet = unet
        self.config = RegionalAttentionConfig(
            num_identities=num_identities,
            tokens_per_identity=tokens_per_identity,
            shared_token_count=shared_token_count,
            enable_regional=True
        )
        self.original_processors = None

    def __enter__(self):
        # Store original processors
        self.original_processors = dict(self.unet.attn_processors)

        # Set regional processors
        set_regional_attention_processors(self.unet, self.config)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original processors
        if self.original_processors is not None:
            self.unet.set_attn_processor(self.original_processors)

        return False  # Don't suppress exceptions

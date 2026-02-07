# Merge image encoder and fuse module to create an ID Encoder
# send multiple ID images, we can directly obtain the updated text encoder containing a stacked ID embedding

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig

from .resampler import FacePerceiverResampler

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class QFormerPerceiver(nn.Module):
    def __init__(self, id_embeddings_dim, cross_attention_dim, num_tokens, embedding_dim=1024, use_residual=True, ratio=4):
        super().__init__()

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.use_residual = use_residual
        print(cross_attention_dim*num_tokens)
        self.token_proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*ratio),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*ratio, cross_attention_dim*num_tokens),
        )
        self.token_norm = nn.LayerNorm(cross_attention_dim)
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=128,
            heads=cross_attention_dim // 128,
            embedding_dim=embedding_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, x, last_hidden_state):
        x = self.token_proj(x)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.token_norm(x) # cls token
        out = self.perceiver_resampler(x, last_hidden_state) # retrieve from patch tokens
        if self.use_residual: # TODO: if use_residual is not true
            out = x + 1.0 * out 
        return out


class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [b, max_num_inputs, 1, 2048]
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        num_inputs = class_tokens_mask.sum().unsqueeze(0) # TODO: check for training case
        batch_size, max_num_inputs = id_embeds.shape[:2]
        # seq_length: 77
        seq_length = prompt_embeds.shape[1]
        # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        flat_id_embeds = id_embeds.view(
            -1, id_embeds.shape[-2], id_embeds.shape[-1]
        )
        # valid_id_mask [b*max_num_inputs]
        valid_id_mask = (
            torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        # slice out the image token embeddings
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds


class PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken(CLIPVisionModelWithProjection):
    def __init__(self, id_embeddings_dim=512):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.fuse_module = FuseModule(2048)
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)

        cross_attention_dim = 2048
        # projection
        self.num_tokens = 2
        self.cross_attention_dim = cross_attention_dim
        self.qformer_perceiver = QFormerPerceiver(
                                    id_embeddings_dim, 
                                    cross_attention_dim, 
                                    self.num_tokens,
                                )

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        last_hidden_state = self.vision_model(id_pixel_values)[0]
        id_embeds = id_embeds.view(b * num_inputs, -1)

        id_embeds = self.qformer_perceiver(id_embeds, last_hidden_state)
        id_embeds = id_embeds.view(b, num_inputs, self.num_tokens, -1)
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)

        return updated_prompt_embeds

class MultiIdentityFuseModule(nn.Module):
    """
    Extended FuseModule that handles multiple identities with separate token ranges.

    Instead of replacing class tokens in the prompt, this module:
    1. Processes each identity's embeddings separately
    2. Concatenates them after the shared prompt tokens
    3. Returns token ranges for regional attention mapping
    """

    def __init__(self, embed_dim, num_tokens_per_identity=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens_per_identity = num_tokens_per_identity

        # Reuse existing fusion logic
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        """Same as original FuseModule.fuse_fn"""
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        base_prompt_embeds: torch.Tensor,
        id_embeds_list: list,
    ):
        """
        Process multiple identities and concatenate to base prompt.

        Args:
            base_prompt_embeds: Base text embeddings (batch, 77, 2048)
            id_embeds_list: List of ID embeddings, each (batch, num_images, num_tokens, 2048)

        Returns:
            combined_embeds: (batch, 77 + num_id_tokens, 2048)
            token_ranges: {identity_idx: (start, end)} mapping
        """
        batch_size = base_prompt_embeds.shape[0]
        seq_len = base_prompt_embeds.shape[1]  # 77

        # Collect ID tokens for each identity
        all_id_tokens = []
        token_ranges = {}
        current_pos = seq_len

        for idx, id_embeds in enumerate(id_embeds_list):
            # id_embeds: (batch, num_images, num_tokens, embed_dim)
            # For multi-identity, take first image's tokens
            if id_embeds.dim() == 4:
                id_tokens = id_embeds[:, 0, :, :]  # (batch, num_tokens, embed_dim)
            else:
                id_tokens = id_embeds

            num_tokens = id_tokens.shape[1]
            all_id_tokens.append(id_tokens)

            token_ranges[idx] = (current_pos, current_pos + num_tokens)
            current_pos += num_tokens

        # Concatenate: base_prompt + all identity tokens
        if all_id_tokens:
            combined_embeds = torch.cat([base_prompt_embeds] + all_id_tokens, dim=1)
        else:
            combined_embeds = base_prompt_embeds

        return combined_embeds, token_ranges


class PhotoMakerIDEncoder_MultiIdentity(PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken):
    """
    Extended encoder supporting multiple identities in single forward pass.

    Processes each identity separately through the base encoder, then combines
    the results using MultiIdentityFuseModule.
    """

    def __init__(self, id_embeddings_dim=512, max_identities=4):
        super().__init__(id_embeddings_dim)
        self.max_identities = max_identities
        self.multi_fuse_module = MultiIdentityFuseModule(2048, num_tokens_per_identity=self.num_tokens)

    def forward_single_identity(
        self,
        id_pixel_values: torch.Tensor,
        id_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process a single identity through vision model and QFormer.

        Args:
            id_pixel_values: (batch, num_images, C, H, W)
            id_embeds: InsightFace embeddings (batch*num_images, 512)

        Returns:
            Processed ID embeddings (batch, num_images, num_tokens, 2048)
        """
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values_flat = id_pixel_values.view(b * num_inputs, c, h, w)

        # CLIP vision features
        last_hidden_state = self.vision_model(id_pixel_values_flat)[0]

        # Reshape ID embeds
        id_embeds_flat = id_embeds.view(b * num_inputs, -1)

        # QFormer perceiver
        id_embeds_processed = self.qformer_perceiver(id_embeds_flat, last_hidden_state)
        id_embeds_processed = id_embeds_processed.view(b, num_inputs, self.num_tokens, -1)

        return id_embeds_processed

    def forward_multi(
        self,
        id_pixel_values_list: list,
        prompt_embeds: torch.Tensor,
        id_embeds_list: list,
    ):
        """
        Process multiple identities and return combined encoder states.

        Args:
            id_pixel_values_list: List of (batch, num_images, C, H, W) per identity
            prompt_embeds: Base text embeddings (batch, 77, 2048)
            id_embeds_list: List of InsightFace embeddings per identity

        Returns:
            combined_prompt_embeds: (batch, 77 + total_id_tokens, 2048)
            token_ranges: {identity_idx: (start, end)} mapping
        """
        processed_id_embeds = []

        for id_pixel_values, id_embeds in zip(id_pixel_values_list, id_embeds_list):
            processed = self.forward_single_identity(id_pixel_values, id_embeds)
            processed_id_embeds.append(processed)

        # Combine using multi-identity fuse module
        combined_embeds, token_ranges = self.multi_fuse_module(
            prompt_embeds,
            processed_id_embeds
        )

        return combined_embeds, token_ranges


if __name__ == "__main__":
    PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()

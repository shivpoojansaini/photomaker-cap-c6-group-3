# Spatial mask generation for multi-identity regional attention
# Converts face bounding boxes to latent-space attention masks

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class RegionalMasks:
    """Collection of masks for multi-identity generation"""
    masks: torch.Tensor              # Shape: (num_identities, H_latent, W_latent)
    blend_weights: torch.Tensor      # Normalized weights for blending
    identity_mapping: Dict[str, int] # region_id -> identity_index


class SpatialMaskGenerator:
    """
    Generate spatial attention masks from face bounding boxes.

    Converts face detection results to soft attention masks in latent space
    that control which identity influences which spatial region during generation.
    """

    def __init__(
        self,
        latent_height: int,
        latent_width: int,
        blur_sigma: float = 8.0,
        expansion_ratio: float = 1.5,
        min_mask_value: float = 0.1,
    ):
        """
        Args:
            latent_height: Height of latent space (image_height / 8 for SDXL)
            latent_width: Width of latent space (image_width / 8 for SDXL)
            blur_sigma: Sigma for Gaussian blur (soft boundaries)
            expansion_ratio: How much to expand face bbox to cover body
            min_mask_value: Minimum attention value even outside region
        """
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.blur_sigma = blur_sigma
        self.expansion_ratio = expansion_ratio
        self.min_mask_value = min_mask_value

    def generate_masks_from_faces(
        self,
        face_bboxes: List[Tuple[float, float, float, float]],
        face_positions: List[str],
        image_width: int,
        image_height: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RegionalMasks:
        """
        Generate attention masks from face bounding boxes.

        Args:
            face_bboxes: List of [x1, y1, x2, y2] in pixel coordinates
            face_positions: Spatial position labels for each face ("left", "right", etc.)
            image_width, image_height: Original image dimensions
            device: Target device
            dtype: Target dtype

        Returns:
            RegionalMasks with soft attention masks for each identity
        """
        num_identities = len(face_bboxes)
        masks = torch.zeros(
            num_identities, self.latent_height, self.latent_width,
            device=device, dtype=dtype
        )

        identity_mapping = {}

        for idx, (bbox, position) in enumerate(zip(face_bboxes, face_positions)):
            # Normalize bbox to [0, 1]
            x1, y1, x2, y2 = bbox
            x1_norm = x1 / image_width
            x2_norm = x2 / image_width
            y1_norm = y1 / image_height
            y2_norm = y2 / image_height

            # Expand bbox to cover body region
            expanded_bbox = self._expand_bbox(x1_norm, y1_norm, x2_norm, y2_norm)

            # Create soft mask
            mask = self._create_soft_mask(expanded_bbox, device, dtype)
            masks[idx] = mask

            identity_mapping[position] = idx

        # Normalize masks for proper blending
        blend_weights = self._compute_blend_weights(masks)

        return RegionalMasks(
            masks=masks,
            blend_weights=blend_weights,
            identity_mapping=identity_mapping
        )

    def generate_position_based_masks(
        self,
        positions: List[str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RegionalMasks:
        """
        Generate masks based on position keywords when bboxes not available.

        Useful for specifying layout without face detection.

        Args:
            positions: List of position keywords ("left", "right", "center")
            device: Target device
            dtype: Target dtype

        Returns:
            RegionalMasks with position-based attention masks
        """
        num_identities = len(positions)
        masks = torch.zeros(
            num_identities, self.latent_height, self.latent_width,
            device=device, dtype=dtype
        )

        identity_mapping = {}

        for idx, position in enumerate(positions):
            bbox = self._position_to_bbox(position, num_identities, idx)
            masks[idx] = self._create_soft_mask(bbox, device, dtype)
            identity_mapping[position] = idx

        blend_weights = self._compute_blend_weights(masks)

        return RegionalMasks(
            masks=masks,
            blend_weights=blend_weights,
            identity_mapping=identity_mapping
        )

    def _position_to_bbox(
        self,
        position: str,
        num_identities: int,
        idx: int
    ) -> Tuple[float, float, float, float]:
        """Convert position keyword to normalized bbox"""
        position = position.lower()

        if position == 'left':
            return (0.0, 0.0, 0.55, 1.0)  # Left half with overlap
        elif position == 'right':
            return (0.45, 0.0, 1.0, 1.0)  # Right half with overlap
        elif position == 'center':
            return (0.25, 0.0, 0.75, 1.0)
        elif position == 'top':
            return (0.0, 0.0, 1.0, 0.55)
        elif position == 'bottom':
            return (0.0, 0.45, 1.0, 1.0)
        else:
            # Default: divide evenly by index
            width = 1.0 / num_identities
            overlap = 0.1
            x1 = max(0.0, idx * width - overlap)
            x2 = min(1.0, (idx + 1) * width + overlap)
            return (x1, 0.0, x2, 1.0)

    def _expand_bbox(
        self,
        x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float, float, float]:
        """Expand face bbox to approximate body region"""
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2

        # Expand horizontally
        new_width = width * self.expansion_ratio

        # Extend more downward (body is below face)
        new_height_above = height * 0.5
        new_height_below = height * self.expansion_ratio * 2

        new_x1 = max(0.0, center_x - new_width / 2)
        new_x2 = min(1.0, center_x + new_width / 2)
        new_y1 = max(0.0, y1 - new_height_above)
        new_y2 = min(1.0, y2 + new_height_below)

        return (new_x1, new_y1, new_x2, new_y2)

    def _create_soft_mask(
        self,
        bbox: Tuple[float, float, float, float],
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create a soft mask with Gaussian-blurred boundaries"""
        x1, y1, x2, y2 = bbox

        # Convert normalized coords to latent space
        lx1 = int(x1 * self.latent_width)
        lx2 = int(x2 * self.latent_width)
        ly1 = int(y1 * self.latent_height)
        ly2 = int(y2 * self.latent_height)

        # Ensure valid ranges
        lx1 = max(0, min(lx1, self.latent_width - 1))
        lx2 = max(lx1 + 1, min(lx2, self.latent_width))
        ly1 = max(0, min(ly1, self.latent_height - 1))
        ly2 = max(ly1 + 1, min(ly2, self.latent_height))

        # Create hard mask
        mask = torch.full(
            (self.latent_height, self.latent_width),
            self.min_mask_value,
            device=device, dtype=dtype
        )
        mask[ly1:ly2, lx1:lx2] = 1.0

        # Apply Gaussian blur for soft boundaries
        mask = self._gaussian_blur_2d(mask, self.blur_sigma)

        # Re-normalize to [min_mask_value, 1.0]
        mask_min = mask.min()
        mask_max = mask.max()
        if mask_max > mask_min:
            mask = (mask - mask_min) / (mask_max - mask_min)
            mask = mask * (1.0 - self.min_mask_value) + self.min_mask_value

        return mask

    def _gaussian_blur_2d(
        self,
        tensor: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """Apply Gaussian blur to a 2D tensor"""
        if sigma <= 0:
            return tensor

        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype)
        x = x - (kernel_size - 1) / 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()

        # Add batch and channel dims for conv2d
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # Horizontal pass
        kernel_h = kernel_1d.view(1, 1, 1, -1)
        pad_h = kernel_size // 2
        tensor = F.pad(tensor, (pad_h, pad_h, 0, 0), mode='replicate')
        tensor = F.conv2d(tensor, kernel_h)

        # Vertical pass
        kernel_v = kernel_1d.view(1, 1, -1, 1)
        pad_v = kernel_size // 2
        tensor = F.pad(tensor, (0, 0, pad_v, pad_v), mode='replicate')
        tensor = F.conv2d(tensor, kernel_v)

        return tensor.squeeze()

    def _compute_blend_weights(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute blending weights for overlapping regions"""
        # Sum of all masks at each position
        mask_sum = masks.sum(dim=0, keepdim=True)
        mask_sum = torch.clamp(mask_sum, min=1e-6)

        # Normalize so weights sum to 1
        blend_weights = masks / mask_sum

        return blend_weights

    def resize_masks(
        self,
        masks: torch.Tensor,
        target_height: int,
        target_width: int
    ) -> torch.Tensor:
        """
        Resize masks to a different resolution.

        Useful when attention layers operate at different resolutions.

        Args:
            masks: Input masks (num_identities, H, W)
            target_height: Target height
            target_width: Target width

        Returns:
            Resized masks (num_identities, target_height, target_width)
        """
        if masks.shape[-2:] == (target_height, target_width):
            return masks

        # Add batch dim for interpolate
        masks_4d = masks.unsqueeze(0)

        resized = F.interpolate(
            masks_4d,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        return resized.squeeze(0)


def extract_face_bboxes_and_positions(
    faces: list,
    sort_by_x: bool = True
) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
    """
    Extract bounding boxes and assign positions from InsightFace detection results.

    Args:
        faces: List of face objects from InsightFace (with 'bbox' attribute)
        sort_by_x: If True, sort faces left-to-right and assign positions

    Returns:
        Tuple of (bboxes, positions)
    """
    if not faces:
        return [], []

    # Extract bboxes
    face_data = [(face['bbox'], face) for face in faces]

    if sort_by_x:
        # Sort by x-coordinate (left to right)
        face_data.sort(key=lambda x: x[0][0])

    bboxes = [fd[0] for fd in face_data]

    # Assign positions based on count and order
    num_faces = len(bboxes)
    if num_faces == 1:
        positions = ['center']
    elif num_faces == 2:
        positions = ['left', 'right']
    elif num_faces == 3:
        positions = ['left', 'center', 'right']
    else:
        positions = [f'region{i}' for i in range(num_faces)]

    return bboxes, positions


def create_masks_for_generation(
    faces: list,
    image_width: int,
    image_height: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    blur_sigma: float = 8.0,
    expansion_ratio: float = 1.5,
) -> RegionalMasks:
    """
    Convenience function to create masks from face detection results.

    Args:
        faces: List of face objects from InsightFace
        image_width, image_height: Original image dimensions
        latent_height, latent_width: Latent space dimensions
        device: Target device
        dtype: Target dtype
        blur_sigma: Gaussian blur sigma
        expansion_ratio: Bbox expansion ratio

    Returns:
        RegionalMasks ready for use in generation
    """
    bboxes, positions = extract_face_bboxes_and_positions(faces)

    if not bboxes:
        # No faces detected, create default left/right split
        generator = SpatialMaskGenerator(
            latent_height, latent_width,
            blur_sigma=blur_sigma,
            expansion_ratio=expansion_ratio
        )
        return generator.generate_position_based_masks(
            ['left', 'right'], device, dtype
        )

    generator = SpatialMaskGenerator(
        latent_height, latent_width,
        blur_sigma=blur_sigma,
        expansion_ratio=expansion_ratio
    )

    return generator.generate_masks_from_faces(
        bboxes, positions,
        image_width, image_height,
        device, dtype
    )

#####
# Multi-Identity PhotoMaker Pipeline
# Single-pass generation with regional attention for multiple identities
# Based on PhotoMaker v2 pipeline
#####

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL
import numpy as np

import torch
from transformers import CLIPImageProcessor

from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import deprecate

from .pipeline import (
    PhotoMakerStableDiffusionXLPipeline,
    retrieve_timesteps,
    rescale_noise_cfg,
    PipelineImageInput,
)
from .model_v2 import PhotoMakerIDEncoder_MultiIdentity
from .prompt_parser import MultiIdentityPromptParser, ParsedMultiPrompt
from .spatial_masks import (
    SpatialMaskGenerator,
    RegionalMasks,
    extract_face_bboxes_and_positions,
    create_masks_for_generation,
)
from .regional_attention import (
    RegionalAttentionConfig,
    RegionalAttnProcessor2_0,
    set_regional_attention_processors,
    RegionalAttentionContext,
)
from .insightface_package import FaceAnalysis2, analyze_faces


class PhotoMakerMultiIdentityPipeline(PhotoMakerStableDiffusionXLPipeline):
    """
    Extended PhotoMaker pipeline supporting multiple identities in a single generation pass.

    Key features:
    - Natural language prompt parsing for multi-identity instructions
    - Regional attention that applies different ID embeddings to different spatial regions
    - Face detection-based spatial mask generation
    - Single diffusion pass for all identities

    Usage:
        pipe = PhotoMakerMultiIdentityPipeline.from_pretrained(...)
        pipe.load_photomaker_adapter(...)
        pipe.setup_face_detector()

        result = pipe(
            prompt="left person as astronaut, right person as doctor",
            input_id_images=[[img1, img2], [img3]],  # Images per identity
            id_embeds_list=[embed1, embed2],          # InsightFace embeddings per identity
        )
    """

    # Don't override __init__ - use lazy initialization for custom attributes
    # This allows from_pretrained() to work correctly

    @property
    def prompt_parser(self):
        """Lazy initialization of prompt parser"""
        if not hasattr(self, '_prompt_parser') or self._prompt_parser is None:
            self._prompt_parser = MultiIdentityPromptParser()
        return self._prompt_parser

    @property
    def face_detector(self):
        """Face detector (must be initialized with setup_face_detector)"""
        return getattr(self, '_face_detector', None)

    @face_detector.setter
    def face_detector(self, value):
        self._face_detector = value

    @property
    def _original_attn_processors(self):
        return getattr(self, '_orig_attn_procs', None)

    @_original_attn_processors.setter
    def _original_attn_processors(self, value):
        self._orig_attn_procs = value

    def setup_face_detector(self, device: str = "cuda"):
        """Initialize face detector for bbox extraction"""
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.face_detector = FaceAnalysis2(
            providers=providers,
            allowed_modules=["detection", "recognition"]
        )
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))

    @torch.no_grad()
    def generate_from_single_image(
        self,
        input_image: PIL.Image.Image,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        start_merge_step: int = 10,
        **kwargs,
    ):
        """
        Generate image from a single input image containing multiple people.

        This is the simplified interface for the common use case:
        - Input: 1 image with 2 people
        - Output: 1 generated image with both people transformed

        Args:
            input_image: PIL Image containing 2 people
            prompt: Natural language prompt like "left person as astronaut, right person as doctor"
            height: Output height (default: 1024)
            width: Output width (default: 1024)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            negative_prompt: Negative prompt
            generator: Random generator for reproducibility
            start_merge_step: Step to start injecting identity (default: 10)

        Returns:
            StableDiffusionXLPipelineOutput with generated image

        Example:
            pipe.setup_face_detector()
            result = pipe.generate_from_single_image(
                input_image=Image.open("couple.jpg"),
                prompt="left person as astronaut, right person as doctor"
            )
            result.images[0].save("output.png")
        """
        if self.face_detector is None:
            raise RuntimeError("Face detector not initialized. Call setup_face_detector() first.")

        # Detect faces in input image
        img_array = np.array(input_image)[:, :, ::-1]  # RGB to BGR for InsightFace
        faces = analyze_faces(self.face_detector, img_array)

        if len(faces) < 2:
            raise ValueError(f"Need at least 2 faces in input image, found {len(faces)}")

        # Sort faces by x-coordinate (left to right)
        faces_sorted = sorted(faces, key=lambda f: f['bbox'][0])

        # Extract embeddings and bboxes
        device = self._execution_device
        dtype = next(self.id_encoder.parameters()).dtype

        embed_left = torch.from_numpy(faces_sorted[0]['embedding']).float().to(device, dtype=dtype)
        embed_right = torch.from_numpy(faces_sorted[1]['embedding']).float().to(device, dtype=dtype)

        bbox_left = tuple(faces_sorted[0]['bbox'])
        bbox_right = tuple(faces_sorted[1]['bbox'])

        # Call main pipeline with extracted data
        return self(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            start_merge_step=start_merge_step,
            # Multi-identity parameters
            multi_identity_mode=True,
            identity_images=[[input_image], [input_image]],  # Same image for both
            identity_embeds=[embed_left.unsqueeze(0), embed_right.unsqueeze(0)],
            regional_positions=["left", "right"],
            face_bboxes=[bbox_left, bbox_right],
            **kwargs,
        )

    def _setup_regional_attention(
        self,
        num_identities: int,
        latent_height: int,
        latent_width: int,
    ) -> Tuple[RegionalAttentionConfig, SpatialMaskGenerator]:
        """Configure UNet for regional attention"""
        config = RegionalAttentionConfig(
            num_identities=num_identities,
            tokens_per_identity=self.num_tokens,
            shared_token_count=77,
            enable_regional=True
        )

        # Store original processors
        self._original_attn_processors = dict(self.unet.attn_processors)

        # Set regional processors
        set_regional_attention_processors(self.unet, config)

        # Create mask generator
        mask_generator = SpatialMaskGenerator(
            latent_height=latent_height,
            latent_width=latent_width
        )

        return config, mask_generator

    def _restore_standard_attention(self):
        """Restore original attention processors"""
        if self._original_attn_processors is not None:
            self.unet.set_attn_processor(self._original_attn_processors)
            self._original_attn_processors = None

    def extract_face_embeddings(
        self,
        images: List[PIL.Image.Image]
    ) -> Tuple[torch.Tensor, List]:
        """
        Extract InsightFace embeddings and bboxes from images.

        Args:
            images: List of PIL images

        Returns:
            embeddings: Tensor of 512D face embeddings
            faces: List of face detection results
        """
        if self.face_detector is None:
            raise RuntimeError("Face detector not initialized. Call setup_face_detector() first.")

        embeddings = []
        all_faces = []

        for img in images:
            img_array = np.array(img)[:, :, ::-1]  # RGB to BGR
            faces = analyze_faces(self.face_detector, img_array)

            if len(faces) > 0:
                embeddings.append(torch.from_numpy(faces[0]['embedding']).float())
                all_faces.append(faces[0])

        if not embeddings:
            raise ValueError("No faces detected in provided images")

        return torch.stack(embeddings), all_faces

    def _prepare_multi_identity_inputs(
        self,
        identity_images: List[List[PIL.Image.Image]],
        identity_embeds: Optional[List[torch.Tensor]] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List]:
        """
        Prepare inputs for multiple identities.

        Args:
            identity_images: List of image lists, one list per identity
            identity_embeds: Optional pre-computed embeddings
            device: Target device
            dtype: Target dtype

        Returns:
            id_pixel_values_list: Processed pixel values per identity
            id_embeds_list: Face embeddings per identity
            all_faces: Face detection results
        """
        id_pixel_values_list = []
        id_embeds_list = []
        all_faces = []

        for idx, images in enumerate(identity_images):
            # Process pixel values
            pixel_values = self.id_image_processor(images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.unsqueeze(0).to(device=device, dtype=dtype)
            id_pixel_values_list.append(pixel_values)

            # Get embeddings
            if identity_embeds is not None and idx < len(identity_embeds):
                embeds = identity_embeds[idx]
                if embeds.dim() == 1:
                    embeds = embeds.unsqueeze(0)
                id_embeds_list.append(embeds.to(device=device, dtype=dtype))
            else:
                # Extract embeddings from images
                embeds, faces = self.extract_face_embeddings(images)
                id_embeds_list.append(embeds.to(device=device, dtype=dtype))
                all_faces.extend(faces)

        return id_pixel_values_list, id_embeds_list, all_faces

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # Standard PhotoMaker parameters
        input_id_images: PipelineImageInput = None,
        start_merge_step: int = 10,
        class_tokens_mask: Optional[torch.LongTensor] = None,
        id_embeds: Optional[torch.FloatTensor] = None,
        # Multi-identity parameters
        multi_identity_mode: bool = False,
        identity_images: Optional[List[List[PIL.Image.Image]]] = None,
        identity_embeds: Optional[List[torch.Tensor]] = None,
        identity_prompts: Optional[List[str]] = None,
        regional_positions: Optional[List[str]] = None,
        face_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
        enable_regional_attention: bool = True,
        **kwargs,
    ):
        """
        Generate images with multiple identities in a single pass.

        Multi-identity mode is activated when:
        - multi_identity_mode=True, or
        - identity_images is provided, or
        - identity_prompts is provided

        Args:
            prompt: Natural language prompt (e.g., "left person as astronaut, right person as doctor")
            identity_images: List of image lists, one per identity
            identity_embeds: Pre-computed InsightFace embeddings per identity
            identity_prompts: Explicit prompts per identity (overrides parsing)
            regional_positions: Position keywords per identity (["left", "right"])
            face_bboxes: Pre-computed face bounding boxes
            enable_regional_attention: Whether to use regional attention

        Returns:
            StableDiffusionXLPipelineOutput with generated images
        """
        # Determine if multi-identity mode
        use_multi_identity = (
            multi_identity_mode or
            identity_images is not None or
            identity_prompts is not None
        )

        if not use_multi_identity:
            # Fall back to standard single-identity generation
            return super().__call__(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                denoising_end=denoising_end,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                original_size=original_size,
                crops_coords_top_left=crops_coords_top_left,
                target_size=target_size,
                input_id_images=input_id_images,
                start_merge_step=start_merge_step,
                class_tokens_mask=class_tokens_mask,
                id_embeds=id_embeds,
                **kwargs
            )

        # === Multi-Identity Generation ===

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Setup dimensions
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # Parse prompt if needed
        if identity_prompts is None:
            parsed = self.prompt_parser.parse(prompt)
            identity_prompts = [r.description for r in parsed.regions]
            if regional_positions is None:
                regional_positions = [r.spatial_hint for r in parsed.regions]

        num_identities = len(identity_prompts)

        # Default positions if not specified
        if regional_positions is None:
            regional_positions = ['left', 'right'][:num_identities]

        # Validate inputs
        if identity_images is None:
            raise ValueError("identity_images must be provided for multi-identity generation")

        if len(identity_images) != num_identities:
            raise ValueError(
                f"Number of identity image groups ({len(identity_images)}) must match "
                f"number of identity prompts ({num_identities})"
            )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        device = self._execution_device
        dtype = next(self.id_encoder.parameters()).dtype

        # Prepare multi-identity inputs
        id_pixel_values_list, id_embeds_list, detected_faces = self._prepare_multi_identity_inputs(
            identity_images=identity_images,
            identity_embeds=identity_embeds,
            device=device,
            dtype=dtype,
        )

        # Setup regional attention
        if enable_regional_attention:
            regional_config, mask_generator = self._setup_regional_attention(
                num_identities=num_identities,
                latent_height=latent_height,
                latent_width=latent_width,
            )

            # Generate spatial masks
            if face_bboxes is not None:
                regional_masks = mask_generator.generate_masks_from_faces(
                    face_bboxes=face_bboxes,
                    face_positions=regional_positions,
                    image_width=width,
                    image_height=height,
                    device=device,
                    dtype=dtype,
                )
            else:
                regional_masks = mask_generator.generate_position_based_masks(
                    positions=regional_positions,
                    device=device,
                    dtype=dtype,
                )
        else:
            regional_masks = None

        # Encode prompts for each identity
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None else None
        )

        # Use first identity's prompt for base encoding
        base_prompt = identity_prompts[0]
        num_id_images = len(identity_images[0])

        (
            prompt_embeds_base,
            _,
            pooled_prompt_embeds,
            _,
            class_tokens_mask_base,
        ) = self.encode_prompt_with_trigger_word(
            prompt=base_prompt,
            prompt_2=prompt_2,
            device=device,
            num_id_images=num_id_images,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,  # Handle CFG separately
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # Process each identity through ID encoder
        processed_id_embeds = []
        for idx, (id_pixel_values, id_embeds_single) in enumerate(zip(id_pixel_values_list, id_embeds_list)):
            b, num_inputs, c, h, w = id_pixel_values.shape
            id_pixel_values_flat = id_pixel_values.view(b * num_inputs, c, h, w)

            # CLIP vision features
            last_hidden_state = self.id_encoder.vision_model(id_pixel_values_flat)[0]

            # Prepare embeddings
            id_embeds_flat = id_embeds_single.view(b * num_inputs, -1)

            # QFormer perceiver
            id_embeds_processed = self.id_encoder.qformer_perceiver(id_embeds_flat, last_hidden_state)
            id_embeds_processed = id_embeds_processed.view(b, num_inputs, self.num_tokens, -1)

            processed_id_embeds.append(id_embeds_processed)

        # Combine identity tokens with base prompt
        # Layout: [base_prompt (77)] + [id1_tokens] + [id2_tokens] + ...
        all_id_tokens = []
        token_ranges = {}
        current_pos = prompt_embeds_base.shape[1]  # 77

        for idx, id_embeds in enumerate(processed_id_embeds):
            # Take first image's tokens
            id_tokens = id_embeds[:, 0, :, :]  # (batch, num_tokens, embed_dim)
            all_id_tokens.append(id_tokens)

            num_tokens = id_tokens.shape[1]
            token_ranges[idx] = (current_pos, current_pos + num_tokens)
            current_pos += num_tokens

        # Concatenate
        combined_prompt_embeds = torch.cat([prompt_embeds_base] + all_id_tokens, dim=1)

        # Prepare text-only embeddings for delayed conditioning
        tokens_text_only = self.tokenizer.encode(base_prompt, add_special_tokens=False)
        trigger_word_token = self.tokenizer.convert_tokens_to_ids(self.trigger_word)
        if trigger_word_token in tokens_text_only:
            tokens_text_only.remove(trigger_word_token)
        prompt_text_only = self.tokenizer.decode(tokens_text_only, add_special_tokens=False)

        (
            prompt_embeds_text_only,
            negative_prompt_embeds,
            _,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt_text_only,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # Duplicate embeddings for num_images_per_prompt
        bs_embed, seq_len, _ = combined_prompt_embeds.shape
        combined_prompt_embeds = combined_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        combined_prompt_embeds = combined_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            1 * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare added time ids
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            # Pad negative embeddings to match combined length
            neg_seq_len = negative_prompt_embeds.shape[1]
            combined_seq_len = combined_prompt_embeds.shape[1]

            if neg_seq_len < combined_seq_len:
                padding = torch.zeros(
                    negative_prompt_embeds.shape[0],
                    combined_seq_len - neg_seq_len,
                    negative_prompt_embeds.shape[2],
                    device=device, dtype=dtype
                )
                negative_prompt_embeds_padded = torch.cat([negative_prompt_embeds, padding], dim=1)
            else:
                negative_prompt_embeds_padded = negative_prompt_embeds[:, :combined_seq_len, :]

            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1 * num_images_per_prompt, 1)

        # Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Choose embeddings based on merge step (delayed ID injection)
                if i <= start_merge_step:
                    # Use text-only embeddings
                    if self.do_classifier_free_guidance:
                        # Pad text-only to match combined length
                        text_only_seq_len = prompt_embeds_text_only.shape[1]
                        if text_only_seq_len < combined_seq_len:
                            padding = torch.zeros(
                                prompt_embeds_text_only.shape[0],
                                combined_seq_len - text_only_seq_len,
                                prompt_embeds_text_only.shape[2],
                                device=device, dtype=dtype
                            )
                            prompt_embeds_text_only_padded = torch.cat([prompt_embeds_text_only, padding], dim=1)
                        else:
                            prompt_embeds_text_only_padded = prompt_embeds_text_only[:, :combined_seq_len, :]

                        current_prompt_embeds = torch.cat(
                            [negative_prompt_embeds_padded, prompt_embeds_text_only_padded], dim=0
                        )
                    else:
                        current_prompt_embeds = prompt_embeds_text_only

                    # No regional attention during warmup
                    current_cross_attn_kwargs = self.cross_attention_kwargs
                else:
                    # Use ID-injected embeddings with regional attention
                    if self.do_classifier_free_guidance:
                        current_prompt_embeds = torch.cat(
                            [negative_prompt_embeds_padded, combined_prompt_embeds], dim=0
                        )
                    else:
                        current_prompt_embeds = combined_prompt_embeds

                    # Add regional attention kwargs
                    if enable_regional_attention and regional_masks is not None:
                        current_cross_attn_kwargs = {
                            **(self.cross_attention_kwargs or {}),
                            "regional_masks": regional_masks.masks,
                            "identity_token_ranges": token_ranges,
                        }
                    else:
                        current_cross_attn_kwargs = self.cross_attention_kwargs

                # Prepare added conditions
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # UNet forward
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=current_cross_attn_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # CFG
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    current_prompt_embeds = callback_outputs.pop("prompt_embeds", current_prompt_embeds)

                # Progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Restore standard attention
        self._restore_standard_attention()

        # Decode latents
        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

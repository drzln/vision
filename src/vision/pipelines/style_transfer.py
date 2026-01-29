"""Style Transfer Pipeline - Find superior visuals and apply their style to seed images."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from vision.pipelines.base import Pipeline, PipelineResult, PipelineStatus, StepResult
from vision.pipelines.registry import register_pipeline

if TYPE_CHECKING:
    from PIL import Image

    from vision.config import VisionConfig
    from vision.pipelines.base import PipelineContext

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models for YAML
# =============================================================================


class SearchVibe(str, Enum):
    """Visual style/vibe for reference search."""

    PROFESSIONAL = "professional"
    CINEMATIC = "cinematic"
    MINIMALIST = "minimalist"
    EDITORIAL = "editorial"
    VIBRANT = "vibrant"
    MOODY = "moody"
    NATURAL = "natural"
    STUDIO = "studio"


class StyleFeature(str, Enum):
    """Features to focus on during style transfer."""

    SMOOTH_SKIN = "smooth_skin"
    CRISP_EDGES = "crisp_edges"
    DRAMATIC_SHADOWS = "dramatic_shadows"
    SOFT_LIGHTING = "soft_lighting"
    HIGH_CONTRAST = "high_contrast"
    COLOR_GRADING = "color_grading"
    TEXTURE_DETAIL = "texture_detail"
    BOKEH = "bokeh"
    SHARPNESS = "sharpness"


class SearchConfig(BaseModel):
    """Configuration for Step 1: The Search."""

    query: str = Field(description="Search query to find reference images")
    num_results: int = Field(default=5, ge=1, le=20, description="Number of reference images to fetch")
    vibe: SearchVibe = Field(default=SearchVibe.PROFESSIONAL, description="Visual style/vibe to search for")
    source: str = Field(default="google", description="Image search source (google, kgiz)")
    filter_min_resolution: int = Field(
        default=1024, ge=256, description="Minimum resolution for reference images"
    )


class RefinementConfig(BaseModel):
    """Configuration for Step 2: The Refinement."""

    strength: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Strength of style transfer (0.0-1.0)"
    )
    features: list[StyleFeature] = Field(
        default_factory=lambda: [StyleFeature.SOFT_LIGHTING, StyleFeature.COLOR_GRADING],
        description="Specific features to focus on during transfer",
    )
    preserve_identity: bool = Field(
        default=True, description="Preserve the core identity/content of seed image"
    )
    blend_mode: str = Field(
        default="adaptive", description="How to blend styles: adaptive, overlay, soft_light"
    )


class PolishConfig(BaseModel):
    """Configuration for Step 3: The Final Polish."""

    enabled: bool = Field(default=True, description="Enable upscaling/resolution enhancement")
    output_width: int = Field(default=2048, ge=256, le=8192, description="Target output width")
    output_height: int | None = Field(
        default=None, description="Target output height (None = maintain aspect ratio)"
    )
    sharpening: float = Field(default=0.3, ge=0.0, le=1.0, description="Post-upscale sharpening")
    denoise: bool = Field(default=True, description="Apply denoising after upscale")


class StyleTransferPipelineConfig(BaseModel):
    """Full configuration for the Style Transfer Pipeline."""

    seed_image: Path = Field(description="Path to the seed/input image")
    output_path: Path = Field(description="Path for the final output image")
    search: SearchConfig = Field(default_factory=SearchConfig, description="Search configuration")
    refinement: RefinementConfig = Field(
        default_factory=RefinementConfig, description="Style refinement configuration"
    )
    polish: PolishConfig = Field(default_factory=PolishConfig, description="Final polish configuration")
    save_intermediates: bool = Field(
        default=False, description="Save intermediate results for debugging"
    )
    intermediate_dir: Path | None = Field(
        default=None, description="Directory for intermediate files"
    )


# =============================================================================
# Pipeline Implementation
# =============================================================================


@dataclass
class SearchResult:
    """Result from the image search step."""

    url: str
    score: float
    resolution: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleAnalysis:
    """Analysis of a reference image's style."""

    lighting: dict[str, float] = field(default_factory=dict)
    colors: dict[str, Any] = field(default_factory=dict)
    texture: dict[str, float] = field(default_factory=dict)
    composition: dict[str, Any] = field(default_factory=dict)


@register_pipeline
class StyleTransferPipeline(Pipeline):
    """Pipeline for finding superior visuals and applying their style to seed images.

    This pipeline implements a three-step workflow:
    1. Search: Find visually superior reference images
    2. Refinement: Apply the visual style to the seed image
    3. Polish: Upscale and enhance the final result

    Example YAML configuration:
    ```yaml
    pipelines:
      - name: hero-image-enhancement
        type: style_transfer
        config:
          seed_image: ./input/product-photo.jpg
          output_path: ./output/enhanced-product.png
          search:
            query: "professional product photography white background"
            num_results: 5
            vibe: professional
          refinement:
            strength: 0.6
            features:
              - soft_lighting
              - crisp_edges
              - color_grading
            preserve_identity: true
          polish:
            enabled: true
            output_width: 2048
            sharpening: 0.4
    ```
    """

    name = "style-transfer"
    description = "Find superior visuals and apply their style to seed images"
    version = "1.0.0"

    def __init__(self, config: VisionConfig | None = None) -> None:
        super().__init__(config)
        self._pipeline_config: StyleTransferPipelineConfig | None = None

    def configure(self, config: StyleTransferPipelineConfig) -> None:
        """Set the pipeline-specific configuration."""
        self._pipeline_config = config

    async def execute(self, ctx: PipelineContext) -> PipelineResult:
        """Execute the style transfer pipeline."""
        result = PipelineResult(
            pipeline_name=self.name,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Extract pipeline config from context
            pipeline_config = self._extract_config(ctx)
            if pipeline_config is None:
                msg = "Style transfer pipeline configuration not found"
                raise ValueError(msg)

            self._pipeline_config = pipeline_config

            # Validate seed image exists
            if not pipeline_config.seed_image.exists():
                msg = f"Seed image not found: {pipeline_config.seed_image}"
                raise FileNotFoundError(msg)

            self._logger.info("Starting style transfer pipeline")
            self._logger.info("Seed image: %s", pipeline_config.seed_image)
            self._logger.info("Output: %s", pipeline_config.output_path)

            # Step 1: The Search
            search_result = await self._execute_search(ctx, pipeline_config, result)
            if search_result.status == PipelineStatus.FAILED:
                raise RuntimeError(search_result.error or "Search step failed")
            result.step_results.append(search_result)

            # Step 2: The Refinement
            refinement_result = await self._execute_refinement(ctx, pipeline_config, result)
            if refinement_result.status == PipelineStatus.FAILED:
                raise RuntimeError(refinement_result.error or "Refinement step failed")
            result.step_results.append(refinement_result)

            # Step 3: The Final Polish
            if pipeline_config.polish.enabled:
                polish_result = await self._execute_polish(ctx, pipeline_config, result)
                if polish_result.status == PipelineStatus.FAILED:
                    raise RuntimeError(polish_result.error or "Polish step failed")
                result.step_results.append(polish_result)

            # Mark output file
            result.output_files.append(pipeline_config.output_path)
            result.status = PipelineStatus.SUCCESS

        except Exception as e:
            self._logger.exception("Style transfer pipeline failed")
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        finally:
            result.completed_at = datetime.now()

        return result

    def _extract_config(self, ctx: PipelineContext) -> StyleTransferPipelineConfig | None:
        """Extract StyleTransferPipelineConfig from pipeline context."""
        # Check if there's a 'config' in the pipeline params
        for step in ctx.pipeline_config.steps:
            if step.type.value == "style_transfer" and step.params:
                return StyleTransferPipelineConfig.model_validate(step.params)

        # Check for top-level config in pipeline params (the whole pipeline is style transfer)
        if hasattr(ctx.pipeline_config, "config") and ctx.pipeline_config.config:
            return StyleTransferPipelineConfig.model_validate(ctx.pipeline_config.config)

        # Try to build from pipeline step params
        config_data = {}
        for step in ctx.pipeline_config.steps:
            if step.name == "config" or step.type.value == "custom":
                config_data.update(step.params)

        if config_data:
            return StyleTransferPipelineConfig.model_validate(config_data)

        return self._pipeline_config

    async def _execute_search(
        self,
        ctx: PipelineContext,
        config: StyleTransferPipelineConfig,
        result: PipelineResult,
    ) -> StepResult:
        """Step 1: Search for visually superior reference images."""
        step_start = datetime.now()
        step_result = StepResult(
            step_name="search",
            status=PipelineStatus.RUNNING,
            duration_ms=0,
        )

        try:
            self._logger.info("Step 1: The Search")
            self._logger.info("  Query: %s", config.search.query)
            self._logger.info("  Vibe: %s", config.search.vibe.value)
            self._logger.info("  Results requested: %d", config.search.num_results)

            if ctx.dry_run:
                self._logger.info("[DRY RUN] Would search for reference images")
                step_result.status = PipelineStatus.SKIPPED
                step_result.output = {
                    "query": config.search.query,
                    "vibe": config.search.vibe.value,
                    "num_results": config.search.num_results,
                    "dry_run": True,
                }
                return step_result

            # Perform the search
            references = await self._search_references(config.search)
            self._logger.info("  Found %d reference images", len(references))

            # Analyze styles from references
            style_analysis = await self._analyze_styles(references)

            # Save intermediate if requested
            if config.save_intermediates and config.intermediate_dir:
                await self._save_search_results(config.intermediate_dir, references)

            step_result.status = PipelineStatus.SUCCESS
            step_result.output = {
                "references_found": len(references),
                "style_analysis": style_analysis,
                "top_reference": references[0].url if references else None,
            }
            step_result.metadata["references"] = [r.url for r in references[:3]]

        except Exception as e:
            self._logger.error("Search step failed: %s", e)
            step_result.status = PipelineStatus.FAILED
            step_result.error = str(e)

        finally:
            step_result.duration_ms = (datetime.now() - step_start).total_seconds() * 1000

        return step_result

    async def _execute_refinement(
        self,
        ctx: PipelineContext,
        config: StyleTransferPipelineConfig,
        result: PipelineResult,
    ) -> StepResult:
        """Step 2: Apply visual style from references to seed image."""
        step_start = datetime.now()
        step_result = StepResult(
            step_name="refinement",
            status=PipelineStatus.RUNNING,
            duration_ms=0,
        )

        try:
            self._logger.info("Step 2: The Refinement")
            self._logger.info("  Strength: %.1f%%", config.refinement.strength * 100)
            self._logger.info("  Features: %s", [f.value for f in config.refinement.features])
            self._logger.info("  Preserve identity: %s", config.refinement.preserve_identity)

            if ctx.dry_run:
                self._logger.info("[DRY RUN] Would apply style transfer")
                step_result.status = PipelineStatus.SKIPPED
                step_result.output = {
                    "strength": config.refinement.strength,
                    "features": [f.value for f in config.refinement.features],
                    "dry_run": True,
                }
                return step_result

            # Load seed image
            seed_image = await self._load_image(config.seed_image)

            # Get style analysis from previous step
            search_output = result.step_results[-1].output if result.step_results else {}
            style_analysis = search_output.get("style_analysis", {})

            # Apply style transfer
            refined_image = await self._apply_style_transfer(
                seed_image,
                style_analysis,
                config.refinement,
            )

            # Save intermediate if requested
            if config.save_intermediates and config.intermediate_dir:
                intermediate_path = config.intermediate_dir / "refined.png"
                await self._save_image(refined_image, intermediate_path)
                self._logger.info("  Saved intermediate: %s", intermediate_path)

            # Store for next step
            step_result.metadata["refined_image"] = refined_image

            step_result.status = PipelineStatus.SUCCESS
            step_result.output = {
                "features_applied": [f.value for f in config.refinement.features],
                "strength": config.refinement.strength,
            }

        except Exception as e:
            self._logger.error("Refinement step failed: %s", e)
            step_result.status = PipelineStatus.FAILED
            step_result.error = str(e)

        finally:
            step_result.duration_ms = (datetime.now() - step_start).total_seconds() * 1000

        return step_result

    async def _execute_polish(
        self,
        ctx: PipelineContext,
        config: StyleTransferPipelineConfig,
        result: PipelineResult,
    ) -> StepResult:
        """Step 3: Final polish - upscale and enhance."""
        step_start = datetime.now()
        step_result = StepResult(
            step_name="polish",
            status=PipelineStatus.RUNNING,
            duration_ms=0,
        )

        try:
            self._logger.info("Step 3: The Final Polish")
            self._logger.info("  Output width: %d", config.polish.output_width)
            self._logger.info("  Sharpening: %.1f%%", config.polish.sharpening * 100)
            self._logger.info("  Denoise: %s", config.polish.denoise)

            if ctx.dry_run:
                self._logger.info("[DRY RUN] Would upscale and polish")
                step_result.status = PipelineStatus.SKIPPED
                step_result.output = {
                    "output_width": config.polish.output_width,
                    "output_path": str(config.output_path),
                    "dry_run": True,
                }
                return step_result

            # Get refined image from previous step
            refinement_result = next(
                (r for r in result.step_results if r.step_name == "refinement"), None
            )
            refined_image = (
                refinement_result.metadata.get("refined_image")
                if refinement_result
                else None
            )

            if refined_image is None:
                # Fallback to loading seed image if refinement was skipped
                refined_image = await self._load_image(config.seed_image)

            # Upscale
            polished_image = await self._upscale_image(
                refined_image,
                config.polish,
            )

            # Save final output
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            await self._save_image(polished_image, config.output_path)
            self._logger.info("  Saved final output: %s", config.output_path)

            step_result.status = PipelineStatus.SUCCESS
            step_result.output = {
                "output_path": str(config.output_path),
                "output_size": polished_image.size if hasattr(polished_image, "size") else None,
            }

        except Exception as e:
            self._logger.error("Polish step failed: %s", e)
            step_result.status = PipelineStatus.FAILED
            step_result.error = str(e)

        finally:
            step_result.duration_ms = (datetime.now() - step_start).total_seconds() * 1000

        return step_result

    # =========================================================================
    # Helper Methods (to be implemented with actual image processing)
    # =========================================================================

    async def _search_references(self, config: SearchConfig) -> list[SearchResult]:
        """Search for reference images based on configuration.

        This is a placeholder - implement with actual Google Images / KGIZ API.
        """
        self._logger.debug("Searching with source: %s", config.source)

        # Build enhanced query with vibe
        vibe_modifiers = {
            SearchVibe.PROFESSIONAL: "professional high-quality",
            SearchVibe.CINEMATIC: "cinematic film look",
            SearchVibe.MINIMALIST: "minimalist clean aesthetic",
            SearchVibe.EDITORIAL: "editorial magazine style",
            SearchVibe.VIBRANT: "vibrant colorful",
            SearchVibe.MOODY: "moody atmospheric",
            SearchVibe.NATURAL: "natural organic look",
            SearchVibe.STUDIO: "studio lighting professional",
        }

        enhanced_query = f"{config.query} {vibe_modifiers.get(config.vibe, '')}"
        self._logger.debug("Enhanced query: %s", enhanced_query)

        # Placeholder results - replace with actual API calls
        # TODO: Integrate with Google Custom Search API or KGIZ
        await asyncio.sleep(0.1)  # Simulate API call

        return [
            SearchResult(
                url=f"https://example.com/reference_{i}.jpg",
                score=0.9 - (i * 0.1),
                resolution=(2048, 1536),
                metadata={"vibe": config.vibe.value},
            )
            for i in range(min(config.num_results, 5))
        ]

    async def _analyze_styles(self, references: list[SearchResult]) -> dict[str, Any]:
        """Analyze the visual styles from reference images.

        Returns aggregated style information from all references.
        """
        # Placeholder - implement with actual image analysis
        # TODO: Use image processing to extract color palettes, lighting, etc.
        await asyncio.sleep(0.05)

        return {
            "dominant_colors": ["#f5f5f5", "#2c3e50", "#e74c3c"],
            "lighting": {
                "type": "soft_diffused",
                "direction": "front_top",
                "intensity": 0.8,
            },
            "texture": {
                "smoothness": 0.7,
                "detail_level": 0.6,
            },
            "contrast": 0.65,
            "saturation": 0.55,
        }

    async def _load_image(self, path: Path) -> Image.Image:
        """Load an image from disk."""
        from PIL import Image as PILImage

        return PILImage.open(path)

    async def _save_image(self, image: Image.Image, path: Path) -> None:
        """Save an image to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    async def _apply_style_transfer(
        self,
        seed_image: Image.Image,
        style_analysis: dict[str, Any],
        config: RefinementConfig,
    ) -> Image.Image:
        """Apply style transfer to the seed image.

        This is a placeholder - implement with actual style transfer algorithms.
        """
        from PIL import ImageEnhance, ImageFilter

        # Placeholder implementation using basic PIL operations
        # TODO: Integrate with neural style transfer or KGIZ API

        result = seed_image.copy()

        # Apply basic enhancements based on features
        for feature in config.features:
            if feature == StyleFeature.SMOOTH_SKIN:
                result = result.filter(ImageFilter.SMOOTH_MORE)
            elif feature == StyleFeature.SHARPNESS:
                enhancer = ImageEnhance.Sharpness(result)
                result = enhancer.enhance(1.0 + config.strength * 0.5)
            elif feature == StyleFeature.HIGH_CONTRAST:
                enhancer = ImageEnhance.Contrast(result)
                result = enhancer.enhance(1.0 + config.strength * 0.3)
            elif feature == StyleFeature.COLOR_GRADING:
                enhancer = ImageEnhance.Color(result)
                result = enhancer.enhance(1.0 + config.strength * 0.2)
            elif feature == StyleFeature.SOFT_LIGHTING:
                enhancer = ImageEnhance.Brightness(result)
                result = enhancer.enhance(1.0 + config.strength * 0.1)

        return result

    async def _upscale_image(
        self,
        image: Image.Image,
        config: PolishConfig,
    ) -> Image.Image:
        """Upscale and polish the image.

        This is a placeholder - implement with actual upscaling algorithms.
        """
        from PIL import Image as PILImage, ImageEnhance, ImageFilter

        # Calculate target size
        original_width, original_height = image.size
        aspect_ratio = original_height / original_width

        target_width = config.output_width
        target_height = config.output_height or int(target_width * aspect_ratio)

        # Upscale using high-quality resampling
        # TODO: Integrate with ESRGAN or similar for better upscaling
        result = image.resize((target_width, target_height), PILImage.Resampling.LANCZOS)

        # Apply sharpening
        if config.sharpening > 0:
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(1.0 + config.sharpening)

        # Apply denoising
        if config.denoise:
            # Simple denoise via slight blur + sharpen
            result = result.filter(ImageFilter.SMOOTH)
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(1.2)

        return result

    async def _save_search_results(
        self,
        output_dir: Path,
        references: list[SearchResult],
    ) -> None:
        """Save search results for debugging."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / "search_results.json"
        with results_file.open("w") as f:
            json.dump(
                [
                    {
                        "url": r.url,
                        "score": r.score,
                        "resolution": r.resolution,
                        "metadata": r.metadata,
                    }
                    for r in references
                ],
                f,
                indent=2,
            )

    def validate_config(self) -> list[str]:
        """Validate pipeline configuration."""
        errors = []

        if self._pipeline_config is None:
            errors.append("Pipeline configuration not set")
            return errors

        config = self._pipeline_config

        if not config.seed_image.exists():
            errors.append(f"Seed image not found: {config.seed_image}")

        if config.save_intermediates and not config.intermediate_dir:
            errors.append("intermediate_dir required when save_intermediates is True")

        return errors

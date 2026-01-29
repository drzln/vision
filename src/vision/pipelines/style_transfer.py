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


class ImageSource(str, Enum):
    """Available image search sources."""

    # Google Custom Search API - text-based image search
    GOOGLE_IMAGES = "google_images"

    # Google Cloud Vision API - reverse image search / find similar
    GOOGLE_ENGINE = "google_engine"

    # Unsplash API - high-quality curated photos (free tier: 50 req/hr)
    UNSPLASH = "unsplash"

    # Pexels API - free stock photos (200 req/hr)
    PEXELS = "pexels"

    # KGIZ service - custom image processing service
    KGIZ = "kgiz"

    # Local directory - use images from a local folder as references
    LOCAL = "local"

    # URL list - fetch from a predefined list of URLs
    URL_LIST = "url_list"


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


class GoogleImagesSourceConfig(BaseModel):
    """Configuration for Google Custom Search API source."""

    # Uses app.google_images credentials from VisionConfig
    safe_search: bool = Field(default=True, description="Enable safe search filtering")
    image_type: str | None = Field(
        default=None, description="Filter by image type: photo, clipart, lineart, animated"
    )
    image_size: str | None = Field(
        default=None, description="Filter by size: large, medium, icon, xlarge, xxlarge"
    )
    dominant_color: str | None = Field(
        default=None, description="Filter by dominant color: black, blue, brown, gray, green, etc."
    )


class GoogleEngineSourceConfig(BaseModel):
    """Configuration for Google Lens / reverse image search source.

    Uses the seed image to find visually similar images, then filters
    by the query and vibe to select the best references.
    """

    use_seed_image: bool = Field(
        default=True, description="Use seed image for reverse search (recommended)"
    )
    reference_image: Path | None = Field(
        default=None, description="Alternative reference image for reverse search"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum visual similarity score"
    )
    include_similar_products: bool = Field(
        default=False, description="Include visually similar products in results"
    )
    include_pages_with_image: bool = Field(
        default=True, description="Include pages containing the image"
    )


class KgizSourceConfig(BaseModel):
    """Configuration for KGIZ image service source."""

    # Uses app.kgiz credentials from VisionConfig
    model: str = Field(default="default", description="KGIZ model to use for search")
    style_match: bool = Field(
        default=True, description="Use style matching algorithm"
    )
    category: str | None = Field(
        default=None, description="Category filter (portrait, landscape, product, etc.)"
    )


class UnsplashSourceConfig(BaseModel):
    """Configuration for Unsplash API source.

    Requires UNSPLASH_ACCESS_KEY environment variable or app config.
    Free tier: 50 requests/hour (demo), 5000 requests/hour (production).
    """

    access_key: str | None = Field(
        default=None, description="Unsplash API access key (or use env UNSPLASH_ACCESS_KEY)"
    )
    orientation: str | None = Field(
        default=None, description="Filter: landscape, portrait, squarish"
    )
    color: str | None = Field(
        default=None,
        description="Filter by color: black_and_white, black, white, yellow, "
        "orange, red, purple, magenta, green, teal, blue",
    )
    content_filter: str = Field(
        default="high", description="Content filter: low (all), high (safe)"
    )


class PexelsSourceConfig(BaseModel):
    """Configuration for Pexels API source.

    Requires PEXELS_API_KEY environment variable or app config.
    Free tier: 200 requests/hour, 20,000 requests/month.
    """

    api_key: str | None = Field(
        default=None, description="Pexels API key (or use env PEXELS_API_KEY)"
    )
    orientation: str | None = Field(
        default=None, description="Filter: landscape, portrait, square"
    )
    size: str | None = Field(
        default=None, description="Filter: large, medium, small"
    )
    color: str | None = Field(
        default=None,
        description="Filter by color: red, orange, yellow, green, turquoise, "
        "blue, violet, pink, brown, black, gray, white",
    )


class LocalSourceConfig(BaseModel):
    """Configuration for local directory source."""

    directory: Path = Field(description="Path to directory containing reference images")
    patterns: list[str] = Field(
        default_factory=lambda: ["*.jpg", "*.jpeg", "*.png", "*.webp"],
        description="File patterns to match",
    )
    recursive: bool = Field(default=False, description="Search subdirectories")
    sort_by: str = Field(
        default="modified", description="Sort order: modified, name, random"
    )


class UrlListSourceConfig(BaseModel):
    """Configuration for URL list source."""

    urls: list[str] = Field(description="List of image URLs to use as references")
    shuffle: bool = Field(default=False, description="Randomize URL order")


class SearchConfig(BaseModel):
    """Configuration for Step 1: The Search.

    Supports multiple image sources:
    - google_images: Text-based search via Google Custom Search API
    - google_engine: Reverse image search via Google Cloud Vision API
    - unsplash: High-quality curated photos (free, 50 req/hr demo)
    - pexels: Free stock photos (200 req/hr)
    - kgiz: Custom KGIZ image service
    - local: Images from a local directory
    - url_list: Predefined list of image URLs
    """

    query: str = Field(
        default="",
        description="Search query (required for most sources)",
    )
    num_results: int = Field(
        default=5, ge=1, le=20, description="Number of reference images to fetch"
    )
    vibe: SearchVibe = Field(
        default=SearchVibe.PROFESSIONAL, description="Visual style/vibe to search for"
    )
    source: ImageSource = Field(
        default=ImageSource.UNSPLASH, description="Image search source"
    )
    filter_min_resolution: int = Field(
        default=1024, ge=256, description="Minimum resolution for reference images"
    )

    # Source-specific configurations (only one should be set based on source)
    google_images: GoogleImagesSourceConfig | None = Field(
        default=None, description="Google Images source configuration"
    )
    google_engine: GoogleEngineSourceConfig | None = Field(
        default=None, description="Google Engine/Lens source configuration"
    )
    unsplash: UnsplashSourceConfig | None = Field(
        default=None, description="Unsplash API source configuration"
    )
    pexels: PexelsSourceConfig | None = Field(
        default=None, description="Pexels API source configuration"
    )
    kgiz: KgizSourceConfig | None = Field(
        default=None, description="KGIZ source configuration"
    )
    local: LocalSourceConfig | None = Field(
        default=None, description="Local directory source configuration"
    )
    url_list: UrlListSourceConfig | None = Field(
        default=None, description="URL list source configuration"
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
    polish: PolishConfig = Field(
        default_factory=PolishConfig, description="Final polish configuration"
    )
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

        Dispatches to the appropriate source handler based on config.source.
        """
        self._logger.info("  Source: %s", config.source.value)

        # Dispatch to source-specific handler
        match config.source:
            case ImageSource.GOOGLE_IMAGES:
                return await self._search_google_images(config)
            case ImageSource.GOOGLE_ENGINE:
                return await self._search_google_engine(config)
            case ImageSource.UNSPLASH:
                return await self._search_unsplash(config)
            case ImageSource.PEXELS:
                return await self._search_pexels(config)
            case ImageSource.KGIZ:
                return await self._search_kgiz(config)
            case ImageSource.LOCAL:
                return await self._search_local(config)
            case ImageSource.URL_LIST:
                return await self._search_url_list(config)
            case _:
                msg = f"Unknown image source: {config.source}"
                raise ValueError(msg)

    async def _search_google_images(self, config: SearchConfig) -> list[SearchResult]:
        """Search using Google Custom Search API (text-based image search).

        Requires app.google_images configuration with API key and search engine ID.
        API docs: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
        """
        import os

        import httpx

        self._logger.debug("Searching Google Images with query: %s", config.query)

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

        enhanced_query = f"{config.query} {vibe_modifiers.get(config.vibe, '')}".strip()
        self._logger.debug("  Enhanced query: %s", enhanced_query)

        # Get source-specific config
        source_config = config.google_images or GoogleImagesSourceConfig()

        # Get API credentials from app config or environment
        api_key = None
        search_engine_id = None

        if self._config and self._config.app.google_images:
            api_key = self._config.app.google_images.api_key
            search_engine_id = self._config.app.google_images.search_engine_id

        # Fall back to environment variables (with ${VAR} expansion support)
        if not api_key or api_key.startswith("${"):
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not search_engine_id or search_engine_id.startswith("${"):
            search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")

        if not api_key:
            msg = "Google API key required. Set GOOGLE_API_KEY env var or app.google_images.api_key"
            raise ValueError(msg)
        if not search_engine_id:
            msg = (
                "Google Search Engine ID required. Set GOOGLE_SEARCH_ENGINE_ID "
                "env var or app.google_images.search_engine_id"
            )
            raise ValueError(msg)

        # Build request parameters
        params: dict[str, str | int] = {
            "key": api_key,
            "cx": search_engine_id,
            "q": enhanced_query,
            "searchType": "image",
            "num": min(config.num_results, 10),  # API max is 10
            "safe": "active" if source_config.safe_search else "off",
        }

        if source_config.image_type:
            params["imgType"] = source_config.image_type
        if source_config.image_size:
            params["imgSize"] = source_config.image_size
        if source_config.dominant_color:
            params["imgDominantColor"] = source_config.dominant_color

        # Make API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        # Parse results
        results: list[SearchResult] = []
        items = data.get("items", [])

        for i, item in enumerate(items):
            image_info = item.get("image", {})
            width = image_info.get("width", 0)
            height = image_info.get("height", 0)

            # Filter by minimum resolution
            if width < config.filter_min_resolution and height < config.filter_min_resolution:
                self._logger.debug("  Skipping low-res image: %dx%d", width, height)
                continue

            results.append(
                SearchResult(
                    url=item.get("link", ""),
                    score=1.0 - (i * 0.05),  # Decay score by position
                    resolution=(width, height),
                    metadata={
                        "source": "google_images",
                        "vibe": config.vibe.value,
                        "query": enhanced_query,
                        "title": item.get("title", ""),
                        "context_link": item.get("image", {}).get("contextLink", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                    },
                )
            )

        self._logger.debug("  Retrieved %d images from Google Custom Search", len(results))
        return results

    async def _search_google_engine(self, config: SearchConfig) -> list[SearchResult]:
        """Search using Google Cloud Vision API WEB_DETECTION (reverse image search).

        Finds visually similar images based on the seed image.
        API docs: https://cloud.google.com/vision/docs/detecting-web
        """
        import base64
        import os

        import httpx

        source_config = config.google_engine or GoogleEngineSourceConfig()

        # Determine which image to use for reverse search
        if source_config.use_seed_image:
            self._logger.debug("  Using seed image for reverse search")
            reference_path = self._pipeline_config.seed_image if self._pipeline_config else None
        else:
            reference_path = source_config.reference_image
            self._logger.debug("  Using reference image: %s", reference_path)

        if not reference_path or not reference_path.exists():
            msg = f"Reference image not found: {reference_path}"
            raise FileNotFoundError(msg)

        self._logger.debug("  Similarity threshold: %.2f", source_config.similarity_threshold)
        self._logger.debug("  Include similar products: %s", source_config.include_similar_products)

        # Get API key from app config or environment
        api_key = None
        if self._config and self._config.app.google_images:
            api_key = self._config.app.google_images.api_key

        if not api_key or api_key.startswith("${"):
            api_key = os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            msg = "Google API key required. Set GOOGLE_API_KEY env var or app.google_images.api_key"
            raise ValueError(msg)

        # Read and encode the reference image as base64
        with open(reference_path, "rb") as f:
            image_content = base64.b64encode(f.read()).decode("utf-8")

        # Build the Vision API request
        request_body = {
            "requests": [
                {
                    "image": {"content": image_content},
                    "features": [
                        {
                            "type": "WEB_DETECTION",
                            "maxResults": config.num_results * 2,  # Request more to filter
                        }
                    ],
                }
            ]
        }

        # Make API request
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

        # Parse results from web detection
        results: list[SearchResult] = []
        responses = data.get("responses", [])

        if not responses:
            self._logger.warning("  No responses from Vision API")
            return results

        web_detection = responses[0].get("webDetection", {})

        # Collect visually similar images
        similar_images = web_detection.get("visuallySimilarImages", [])
        for i, img in enumerate(similar_images):
            url = img.get("url", "")
            score = img.get("score", 0.8 - (i * 0.05))

            # Filter by similarity threshold
            if score < source_config.similarity_threshold:
                continue

            results.append(
                SearchResult(
                    url=url,
                    score=score,
                    resolution=(0, 0),  # Not provided by API
                    metadata={
                        "source": "google_engine",
                        "type": "visually_similar",
                        "vibe": config.vibe.value,
                    },
                )
            )

        # Optionally include pages with matching images
        if source_config.include_pages_with_image:
            pages = web_detection.get("pagesWithMatchingImages", [])
            for page in pages[:config.num_results]:
                # Get full matching images from pages
                full_images = page.get("fullMatchingImages", [])
                partial_images = page.get("partialMatchingImages", [])

                for img in full_images + partial_images:
                    url = img.get("url", "")
                    if url and not any(r.url == url for r in results):
                        results.append(
                            SearchResult(
                                url=url,
                                score=0.85,  # High score for page matches
                                resolution=(0, 0),
                                metadata={
                                    "source": "google_engine",
                                    "type": "page_match",
                                    "page_url": page.get("url", ""),
                                    "page_title": page.get("pageTitle", ""),
                                    "vibe": config.vibe.value,
                                },
                            )
                        )

        # Optionally include similar products
        if source_config.include_similar_products:
            products = web_detection.get("visuallySimilarProducts", []) or web_detection.get(
                "productSearchResults", []
            )
            for product in products[:config.num_results]:
                url = product.get("image", {}).get("url", "") or product.get("url", "")
                if url and not any(r.url == url for r in results):
                    results.append(
                        SearchResult(
                            url=url,
                            score=0.75,
                            resolution=(0, 0),
                            metadata={
                                "source": "google_engine",
                                "type": "similar_product",
                                "vibe": config.vibe.value,
                            },
                        )
                    )

        # Limit to requested number
        results = results[: config.num_results]
        self._logger.debug("  Retrieved %d similar images from Vision API", len(results))
        return results

    async def _search_kgiz(self, config: SearchConfig) -> list[SearchResult]:
        """Search using KGIZ image service.

        Requires app.kgiz configuration with endpoint and API key.
        """
        import os

        import httpx

        source_config = config.kgiz or KgizSourceConfig()

        self._logger.debug("  KGIZ model: %s", source_config.model)
        self._logger.debug("  Style match: %s", source_config.style_match)
        if source_config.category:
            self._logger.debug("  Category: %s", source_config.category)

        # Get KGIZ credentials from app config or environment
        endpoint = None
        api_key = None

        if self._config and self._config.app.kgiz:
            endpoint = self._config.app.kgiz.endpoint
            api_key = self._config.app.kgiz.api_key

        if not endpoint or endpoint.startswith("${"):
            endpoint = os.environ.get("KGIZ_ENDPOINT")
        if not api_key or api_key.startswith("${"):
            api_key = os.environ.get("KGIZ_API_KEY")

        if not endpoint:
            msg = "KGIZ endpoint required. Set KGIZ_ENDPOINT env var or app.kgiz.endpoint"
            raise ValueError(msg)
        if not api_key:
            msg = "KGIZ API key required. Set KGIZ_API_KEY env var or app.kgiz.api_key"
            raise ValueError(msg)

        # Build request body
        request_body = {
            "query": config.query,
            "vibe": config.vibe.value,
            "model": source_config.model,
            "style_match": source_config.style_match,
            "limit": config.num_results,
            "min_resolution": config.filter_min_resolution,
        }

        if source_config.category:
            request_body["category"] = source_config.category

        # Make API request
        timeout = self._config.app.kgiz.timeout if self._config and self._config.app.kgiz else 30
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/v1/search",
                json=request_body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        # Parse results
        results: list[SearchResult] = []
        items = data.get("results", data.get("images", []))

        for i, item in enumerate(items):
            url = item.get("url", item.get("image_url", ""))
            width = item.get("width", 0)
            height = item.get("height", 0)
            score = item.get("score", item.get("relevance", 0.9 - (i * 0.05)))

            if not url:
                continue

            results.append(
                SearchResult(
                    url=url,
                    score=score,
                    resolution=(width, height),
                    metadata={
                        "source": "kgiz",
                        "model": source_config.model,
                        "vibe": config.vibe.value,
                        "id": item.get("id", ""),
                        "category": item.get("category", source_config.category),
                    },
                )
            )

        self._logger.debug("  Retrieved %d images from KGIZ", len(results))
        return results

    async def _search_unsplash(self, config: SearchConfig) -> list[SearchResult]:
        """Search using Unsplash API.

        Requires UNSPLASH_ACCESS_KEY environment variable or config.
        API docs: https://unsplash.com/documentation#search-photos
        Free tier: 50 requests/hour (demo), 5000 requests/hour (production).
        """
        import os

        import httpx

        source_config = config.unsplash or UnsplashSourceConfig()

        # Get access key from config or environment
        access_key = source_config.access_key
        if not access_key or access_key.startswith("${"):
            access_key = os.environ.get("UNSPLASH_ACCESS_KEY")

        if not access_key:
            msg = (
                "Unsplash access key required. Set UNSPLASH_ACCESS_KEY "
                "env var or search.unsplash.access_key"
            )
            raise ValueError(msg)

        # Build enhanced query with vibe
        vibe_modifiers = {
            SearchVibe.PROFESSIONAL: "professional",
            SearchVibe.CINEMATIC: "cinematic film",
            SearchVibe.MINIMALIST: "minimalist clean",
            SearchVibe.EDITORIAL: "editorial magazine",
            SearchVibe.VIBRANT: "vibrant colorful",
            SearchVibe.MOODY: "moody dark",
            SearchVibe.NATURAL: "natural organic",
            SearchVibe.STUDIO: "studio lighting",
        }

        enhanced_query = f"{config.query} {vibe_modifiers.get(config.vibe, '')}".strip()
        self._logger.debug("  Unsplash query: %s", enhanced_query)

        # Build request parameters
        params: dict[str, str | int] = {
            "query": enhanced_query,
            "per_page": min(config.num_results, 30),  # API max is 30
            "content_filter": source_config.content_filter,
        }

        if source_config.orientation:
            params["orientation"] = source_config.orientation
        if source_config.color:
            params["color"] = source_config.color

        # Make API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.unsplash.com/search/photos",
                params=params,
                headers={
                    "Authorization": f"Client-ID {access_key}",
                    "Accept-Version": "v1",
                },
            )
            response.raise_for_status()
            data = response.json()

        # Parse results
        results: list[SearchResult] = []
        photos = data.get("results", [])

        for i, photo in enumerate(photos):
            urls = photo.get("urls", {})
            # Prefer 'regular' size (1080px width) or 'full' for high quality
            url = urls.get("regular", urls.get("full", urls.get("raw", "")))

            width = photo.get("width", 0)
            height = photo.get("height", 0)

            # Filter by minimum resolution
            if width < config.filter_min_resolution and height < config.filter_min_resolution:
                self._logger.debug("  Skipping low-res image: %dx%d", width, height)
                continue

            results.append(
                SearchResult(
                    url=url,
                    score=1.0 - (i * 0.03),  # Decay by position
                    resolution=(width, height),
                    metadata={
                        "source": "unsplash",
                        "vibe": config.vibe.value,
                        "id": photo.get("id", ""),
                        "description": photo.get("description", ""),
                        "alt_description": photo.get("alt_description", ""),
                        "user": photo.get("user", {}).get("name", ""),
                        "download_link": photo.get("links", {}).get("download", ""),
                        "color": photo.get("color", ""),
                    },
                )
            )

        self._logger.debug("  Retrieved %d images from Unsplash", len(results))
        return results

    async def _search_pexels(self, config: SearchConfig) -> list[SearchResult]:
        """Search using Pexels API.

        Requires PEXELS_API_KEY environment variable or config.
        API docs: https://www.pexels.com/api/documentation/#photos-search
        Free tier: 200 requests/hour, 20,000 requests/month.
        """
        import os

        import httpx

        source_config = config.pexels or PexelsSourceConfig()

        # Get API key from config or environment
        api_key = source_config.api_key
        if not api_key or api_key.startswith("${"):
            api_key = os.environ.get("PEXELS_API_KEY")

        if not api_key:
            msg = "Pexels API key required. Set PEXELS_API_KEY env var or search.pexels.api_key"
            raise ValueError(msg)

        # Build enhanced query with vibe
        vibe_modifiers = {
            SearchVibe.PROFESSIONAL: "professional",
            SearchVibe.CINEMATIC: "cinematic",
            SearchVibe.MINIMALIST: "minimalist",
            SearchVibe.EDITORIAL: "editorial",
            SearchVibe.VIBRANT: "vibrant colorful",
            SearchVibe.MOODY: "moody atmospheric",
            SearchVibe.NATURAL: "natural",
            SearchVibe.STUDIO: "studio",
        }

        enhanced_query = f"{config.query} {vibe_modifiers.get(config.vibe, '')}".strip()
        self._logger.debug("  Pexels query: %s", enhanced_query)

        # Build request parameters
        params: dict[str, str | int] = {
            "query": enhanced_query,
            "per_page": min(config.num_results, 80),  # API max is 80
        }

        if source_config.orientation:
            params["orientation"] = source_config.orientation
        if source_config.size:
            params["size"] = source_config.size
        if source_config.color:
            params["color"] = source_config.color

        # Make API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.pexels.com/v1/search",
                params=params,
                headers={
                    "Authorization": api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Parse results
        results: list[SearchResult] = []
        photos = data.get("photos", [])

        for i, photo in enumerate(photos):
            src = photo.get("src", {})
            # Prefer 'large2x' (940x650 or larger) or 'original' for high quality
            url = src.get("large2x", src.get("original", src.get("large", "")))

            width = photo.get("width", 0)
            height = photo.get("height", 0)

            # Filter by minimum resolution
            if width < config.filter_min_resolution and height < config.filter_min_resolution:
                self._logger.debug("  Skipping low-res image: %dx%d", width, height)
                continue

            results.append(
                SearchResult(
                    url=url,
                    score=1.0 - (i * 0.03),  # Decay by position
                    resolution=(width, height),
                    metadata={
                        "source": "pexels",
                        "vibe": config.vibe.value,
                        "id": photo.get("id", ""),
                        "photographer": photo.get("photographer", ""),
                        "photographer_url": photo.get("photographer_url", ""),
                        "avg_color": photo.get("avg_color", ""),
                        "alt": photo.get("alt", ""),
                    },
                )
            )

        self._logger.debug("  Retrieved %d images from Pexels", len(results))
        return results

    async def _search_local(self, config: SearchConfig) -> list[SearchResult]:
        """Search for reference images in a local directory."""
        source_config = config.local
        if source_config is None:
            msg = "Local source requires 'local' configuration"
            raise ValueError(msg)

        self._logger.debug("  Directory: %s", source_config.directory)
        self._logger.debug("  Patterns: %s", source_config.patterns)
        self._logger.debug("  Recursive: %s", source_config.recursive)

        if not source_config.directory.exists():
            msg = f"Local directory not found: {source_config.directory}"
            raise FileNotFoundError(msg)

        # Find matching files
        import random
        from glob import glob

        all_files: list[Path] = []
        for pattern in source_config.patterns:
            if source_config.recursive:
                matches = glob(str(source_config.directory / "**" / pattern), recursive=True)
            else:
                matches = glob(str(source_config.directory / pattern))
            all_files.extend(Path(m) for m in matches)

        # Sort files
        match source_config.sort_by:
            case "modified":
                all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            case "name":
                all_files.sort(key=lambda p: p.name)
            case "random":
                random.shuffle(all_files)

        # Limit to requested number
        selected_files = all_files[: config.num_results]
        self._logger.info("  Found %d local reference images", len(selected_files))

        # Build results
        results = []
        for i, file_path in enumerate(selected_files):
            # Get image resolution
            try:
                from PIL import Image as PILImage

                with PILImage.open(file_path) as img:
                    resolution = img.size
            except Exception:
                resolution = (0, 0)

            results.append(
                SearchResult(
                    url=f"file://{file_path.absolute()}",
                    score=1.0 - (i * 0.02),  # Slight decay by order
                    resolution=resolution,
                    metadata={
                        "source": "local",
                        "path": str(file_path),
                        "vibe": config.vibe.value,
                    },
                )
            )

        return results

    async def _search_url_list(self, config: SearchConfig) -> list[SearchResult]:
        """Use a predefined list of URLs as reference images."""
        source_config = config.url_list
        if source_config is None:
            msg = "URL list source requires 'url_list' configuration"
            raise ValueError(msg)

        self._logger.debug("  URLs provided: %d", len(source_config.urls))

        urls = list(source_config.urls)
        if source_config.shuffle:
            import random

            random.shuffle(urls)

        # Limit to requested number
        selected_urls = urls[: config.num_results]

        return [
            SearchResult(
                url=url,
                score=1.0 - (i * 0.01),  # Slight decay by order
                resolution=(0, 0),  # Unknown until fetched
                metadata={
                    "source": "url_list",
                    "vibe": config.vibe.value,
                },
            )
            for i, url in enumerate(selected_urls)
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
        from PIL import Image as PILImage
        from PIL import ImageEnhance, ImageFilter

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

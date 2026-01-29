"""Pydantic models for YAML configuration validation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class GoogleImagesConfig(BaseModel):
    """Google Images API configuration."""

    api_key: SecretStr = Field(description="Google Custom Search API key")
    search_engine_id: str = Field(description="Google Custom Search Engine ID")
    safe_search: Literal["off", "medium", "high"] = Field(
        default="medium", description="Safe search level"
    )
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results per query")


class KgizConfig(BaseModel):
    """KGIZ service configuration."""

    endpoint: str = Field(description="KGIZ API endpoint URL")
    api_key: SecretStr | None = Field(default=None, description="KGIZ API key (if required)")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")


class AppConfig(BaseModel):
    """Application-level configuration including credentials."""

    google_images: GoogleImagesConfig | None = Field(
        default=None, description="Google Images API configuration"
    )
    kgiz: KgizConfig | None = Field(default=None, description="KGIZ service configuration")
    output_dir: Path = Field(default=Path("./output"), description="Default output directory")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    max_concurrent: int = Field(
        default=4, ge=1, le=32, description="Maximum concurrent operations"
    )


class StepType(str, Enum):
    """Available pipeline step types."""

    FETCH_GOOGLE = "fetch_google"
    FETCH_KGIZ = "fetch_kgiz"
    RESIZE = "resize"
    CROP = "crop"
    FILTER = "filter"
    TRANSFORM = "transform"
    SAVE = "save"
    CUSTOM = "custom"
    STYLE_TRANSFER = "style_transfer"


class PipelineType(str, Enum):
    """Available pipeline types."""

    YAML_DRIVEN = "yaml_driven"
    STYLE_TRANSFER = "style_transfer"
    GOOGLE_IMAGES = "google_images"
    KGIZ = "kgiz"


class PipelineStep(BaseModel):
    """A single step in a pipeline."""

    name: str = Field(description="Step name for identification")
    type: StepType = Field(description="Type of operation to perform")
    params: dict[str, Any] = Field(default_factory=dict, description="Step-specific parameters")
    condition: str | None = Field(default=None, description="Optional condition expression")
    on_error: Literal["fail", "skip", "retry"] = Field(
        default="fail", description="Error handling strategy"
    )
    retry_count: int = Field(default=3, ge=1, le=10, description="Number of retries if on_error=retry")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure step name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            msg = "Step name must be alphanumeric with underscores or hyphens"
            raise ValueError(msg)
        return v


class PipelineConfig(BaseModel):
    """Configuration for a single pipeline.

    Supports two modes:
    1. Step-based: Define individual steps for yaml-driven execution
    2. Type-based: Use a specialized pipeline type with its own config

    Example (step-based):
    ```yaml
    pipelines:
      - name: resize-images
        steps:
          - name: resize
            type: resize
            params:
              width: 800
    ```

    Example (type-based):
    ```yaml
    pipelines:
      - name: enhance-photos
        type: style_transfer
        config:
          seed_image: ./input/photo.jpg
          output_path: ./output/enhanced.png
          search:
            query: "professional photography"
            vibe: cinematic
    ```
    """

    name: str = Field(description="Pipeline name")
    description: str = Field(default="", description="Pipeline description")
    enabled: bool = Field(default=True, description="Whether pipeline is enabled")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")

    # Pipeline type (optional - defaults to yaml_driven if steps are provided)
    type: PipelineType | None = Field(
        default=None,
        description="Pipeline type for specialized pipelines (style_transfer, etc.)",
    )

    # For step-based pipelines
    steps: list[PipelineStep] = Field(
        default_factory=list,
        description="Ordered list of pipeline steps (for yaml_driven type)",
    )

    # For type-based pipelines
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline-specific configuration (for typed pipelines)",
    )

    input_patterns: list[str] = Field(
        default_factory=lambda: ["*.jpg", "*.png"],
        description="Input file patterns to match",
    )
    output_format: str = Field(default="png", description="Default output format")

    @field_validator("name")
    @classmethod
    def validate_pipeline_name(cls, v: str) -> str:
        """Ensure pipeline name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            msg = "Pipeline name must be alphanumeric with underscores or hyphens"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_pipeline_config(self) -> PipelineConfig:
        """Validate that pipeline has either steps or type+config."""
        has_steps = bool(self.steps)
        has_type = self.type is not None
        has_config = bool(self.config)

        # If type is specified, it's a typed pipeline
        if has_type:
            # style_transfer requires config
            if self.type == PipelineType.STYLE_TRANSFER and not has_config:
                msg = "style_transfer pipeline requires 'config' section"
                raise ValueError(msg)
            return self

        # If no type, must have steps (yaml_driven)
        if not has_steps:
            msg = "Pipeline must have either 'type' or 'steps' defined"
            raise ValueError(msg)

        return self

    @property
    def effective_type(self) -> PipelineType:
        """Get the effective pipeline type."""
        if self.type is not None:
            return self.type
        return PipelineType.YAML_DRIVEN


class VisionConfig(BaseModel):
    """Root configuration model for Vision."""

    version: Annotated[str, Field(pattern=r"^\d+\.\d+$")] = Field(
        default="1.0", description="Configuration schema version"
    )
    app: AppConfig = Field(default_factory=AppConfig, description="Application configuration")
    pipelines: list[PipelineConfig] = Field(
        default_factory=list, description="Declared pipelines"
    )

    @classmethod
    def from_yaml(cls, path: Path) -> VisionConfig:
        """Load configuration from a YAML file."""
        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            data = yaml.safe_load(f)

        if data is None:
            msg = f"Configuration file is empty: {path}"
            raise ValueError(msg)

        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        with path.open("w") as f:
            yaml.safe_dump(
                self.model_dump(mode="json", exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def get_pipeline(self, name: str) -> PipelineConfig | None:
        """Get a pipeline by name."""
        for pipeline in self.pipelines:
            if pipeline.name == name:
                return pipeline
        return None

    def get_enabled_pipelines(self) -> list[PipelineConfig]:
        """Get all enabled pipelines."""
        return [p for p in self.pipelines if p.enabled]

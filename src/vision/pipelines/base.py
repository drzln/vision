"""Base pipeline class for image processing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vision.config import PipelineConfig, VisionConfig

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineContext:
    """Context passed through pipeline execution."""

    config: VisionConfig
    pipeline_config: PipelineConfig
    input_files: list[Path] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    metadata: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Ensure output directory exists."""
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class StepResult:
    """Result of a single pipeline step."""

    step_name: str
    status: PipelineStatus
    duration_ms: float
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution."""

    pipeline_name: str
    status: PipelineStatus
    started_at: datetime
    completed_at: datetime | None = None
    step_results: list[StepResult] = field(default_factory=list)
    output_files: list[Path] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        """Calculate total pipeline duration in milliseconds."""
        if self.completed_at is None:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000

    @property
    def successful_steps(self) -> int:
        """Count of successful steps."""
        return sum(1 for r in self.step_results if r.status == PipelineStatus.SUCCESS)

    @property
    def failed_steps(self) -> int:
        """Count of failed steps."""
        return sum(1 for r in self.step_results if r.status == PipelineStatus.FAILED)


class Pipeline(ABC):
    """Abstract base class for image processing pipelines.

    Inherit from this class to create custom pipelines with specialized
    processing logic. Implement the `execute` method to define the
    pipeline's behavior.

    Example:
        ```python
        from vision.pipelines import Pipeline, PipelineContext, PipelineResult

        class MyCustomPipeline(Pipeline):
            name = "my-custom-pipeline"
            description = "Processes images with custom logic"

            async def execute(self, ctx: PipelineContext) -> PipelineResult:
                # Your custom processing logic here
                return PipelineResult(
                    pipeline_name=self.name,
                    status=PipelineStatus.SUCCESS,
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )
        ```
    """

    name: str = "base-pipeline"
    description: str = "Base pipeline class"
    version: str = "1.0.0"

    def __init__(self, config: VisionConfig | None = None) -> None:
        """Initialize pipeline with optional configuration."""
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.name}")

    @property
    def config(self) -> VisionConfig | None:
        """Get the pipeline configuration."""
        return self._config

    @config.setter
    def config(self, value: VisionConfig) -> None:
        """Set the pipeline configuration."""
        self._config = value

    def configure(self, pipeline_config: Any) -> None:  # noqa: B027
        """Configure the pipeline with pipeline-specific settings.

        Override this method in subclasses to accept typed configuration.
        The default implementation does nothing.

        Args:
            pipeline_config: Pipeline-specific configuration object.
        """

    @abstractmethod
    async def execute(self, ctx: PipelineContext) -> PipelineResult:
        """Execute the pipeline.

        Args:
            ctx: Pipeline execution context containing configuration,
                 input files, and metadata.

        Returns:
            PipelineResult containing execution status, timing, and outputs.

        Raises:
            PipelineError: If pipeline execution fails unrecoverably.
        """
        ...

    async def setup(self, ctx: PipelineContext) -> None:
        """Optional setup hook called before execution.

        Override this method to perform initialization tasks like
        validating credentials or preparing resources.
        """
        self._logger.debug("Pipeline setup: %s", self.name)

    async def teardown(self, ctx: PipelineContext, result: PipelineResult) -> None:
        """Optional teardown hook called after execution.

        Override this method to perform cleanup tasks like
        closing connections or releasing resources.
        """
        self._logger.debug("Pipeline teardown: %s", self.name)

    async def run(self, ctx: PipelineContext) -> PipelineResult:
        """Run the complete pipeline lifecycle.

        Calls setup -> execute -> teardown in sequence.
        """
        result = PipelineResult(
            pipeline_name=self.name,
            status=PipelineStatus.PENDING,
            started_at=datetime.now(),
        )

        try:
            self._logger.info("Starting pipeline: %s", self.name)
            await self.setup(ctx)

            result.status = PipelineStatus.RUNNING
            result = await self.execute(ctx)

        except Exception as e:
            self._logger.exception("Pipeline failed: %s", self.name)
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        finally:
            result.completed_at = datetime.now()
            await self.teardown(ctx, result)
            self._logger.info(
                "Pipeline completed: %s (status=%s, duration=%.2fms)",
                self.name,
                result.status.value,
                result.duration_ms,
            )

        return result

    def validate_config(self) -> list[str]:
        """Validate pipeline configuration and return any errors.

        Override this method to add custom validation logic.

        Returns:
            List of validation error messages (empty if valid).
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"

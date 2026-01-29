"""Pipeline registry for discovering and managing pipelines."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vision.pipelines.base import Pipeline

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Registry for pipeline classes.

    Use this to register custom pipeline implementations and retrieve
    them by name for execution.

    Example:
        ```python
        from vision.pipelines import Pipeline, PipelineRegistry

        class MyPipeline(Pipeline):
            name = "my-pipeline"
            async def execute(self, ctx):
                ...

        # Register the pipeline
        registry = PipelineRegistry()
        registry.register(MyPipeline)

        # Later, retrieve and instantiate
        pipeline_cls = registry.get("my-pipeline")
        pipeline = pipeline_cls(config)
        ```
    """

    _instance: PipelineRegistry | None = None

    def __new__(cls) -> PipelineRegistry:
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipelines = {}
        return cls._instance

    def __init__(self) -> None:
        """Initialize registry (only runs once due to singleton)."""
        if not hasattr(self, "_pipelines"):
            self._pipelines: dict[str, type[Pipeline]] = {}

    def register(self, pipeline_cls: type[Pipeline]) -> type[Pipeline]:
        """Register a pipeline class.

        Can be used as a decorator:
            ```python
            @registry.register
            class MyPipeline(Pipeline):
                ...
            ```

        Or called directly:
            ```python
            registry.register(MyPipeline)
            ```
        """
        name = pipeline_cls.name
        if name in self._pipelines:
            logger.warning("Overwriting existing pipeline registration: %s", name)
        self._pipelines[name] = pipeline_cls
        logger.debug("Registered pipeline: %s", name)
        return pipeline_cls

    def unregister(self, name: str) -> bool:
        """Unregister a pipeline by name.

        Returns True if pipeline was found and removed, False otherwise.
        """
        if name in self._pipelines:
            del self._pipelines[name]
            logger.debug("Unregistered pipeline: %s", name)
            return True
        return False

    def get(self, name: str) -> type[Pipeline] | None:
        """Get a pipeline class by name.

        Returns None if not found.
        """
        return self._pipelines.get(name)

    def list_all(self) -> list[str]:
        """List all registered pipeline names."""
        return list(self._pipelines.keys())

    def get_all(self) -> dict[str, type[Pipeline]]:
        """Get all registered pipeline classes."""
        return dict(self._pipelines)

    def clear(self) -> None:
        """Clear all registered pipelines."""
        self._pipelines.clear()
        logger.debug("Cleared all pipeline registrations")

    def __contains__(self, name: str) -> bool:
        """Check if a pipeline is registered."""
        return name in self._pipelines

    def __len__(self) -> int:
        """Return number of registered pipelines."""
        return len(self._pipelines)


# Global registry instance
registry = PipelineRegistry()


def register_pipeline(cls: type[Pipeline]) -> type[Pipeline]:
    """Decorator to register a pipeline with the global registry.

    Example:
        ```python
        @register_pipeline
        class MyPipeline(Pipeline):
            name = "my-pipeline"
            ...
        ```
    """
    return registry.register(cls)

"""Built-in pipeline implementations."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from vision.pipelines.base import Pipeline, PipelineResult, PipelineStatus, StepResult
from vision.pipelines.registry import register_pipeline

if TYPE_CHECKING:
    from vision.pipelines.base import PipelineContext


@register_pipeline
class GoogleImagesPipeline(Pipeline):
    """Pipeline for fetching and processing images from Google Images.

    This pipeline uses the Google Custom Search API to find images
    based on search queries, then processes them according to
    configured steps.
    """

    name = "google-images"
    description = "Fetch and process images from Google Images API"
    version = "1.0.0"

    async def execute(self, ctx: PipelineContext) -> PipelineResult:
        """Execute the Google Images pipeline."""
        result = PipelineResult(
            pipeline_name=self.name,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Validate Google Images config exists
            if ctx.config.app.google_images is None:
                msg = "Google Images configuration required but not provided"
                raise ValueError(msg)

            config = ctx.config.app.google_images

            # Process each step in the pipeline
            for step in ctx.pipeline_config.steps:
                step_start = datetime.now()
                step_result = StepResult(
                    step_name=step.name,
                    status=PipelineStatus.RUNNING,
                    duration_ms=0,
                )

                try:
                    self._logger.info("Executing step: %s", step.name)

                    if ctx.dry_run:
                        self._logger.info("[DRY RUN] Would execute step: %s", step.name)
                        step_result.status = PipelineStatus.SKIPPED
                    else:
                        # Placeholder for actual step execution
                        await asyncio.sleep(0.01)  # Simulate work
                        step_result.status = PipelineStatus.SUCCESS
                        step_result.output = {"message": f"Completed {step.name}"}

                except Exception as e:
                    self._logger.error("Step failed: %s - %s", step.name, e)
                    step_result.status = PipelineStatus.FAILED
                    step_result.error = str(e)

                    if step.on_error == "fail":
                        raise

                finally:
                    step_end = datetime.now()
                    step_result.duration_ms = (step_end - step_start).total_seconds() * 1000
                    result.step_results.append(step_result)

            result.status = PipelineStatus.SUCCESS

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        finally:
            result.completed_at = datetime.now()

        return result


@register_pipeline
class KgizPipeline(Pipeline):
    """Pipeline for processing images with KGIZ service.

    KGIZ provides advanced image analysis and transformation
    capabilities through its API.
    """

    name = "kgiz"
    description = "Process images with KGIZ service"
    version = "1.0.0"

    async def execute(self, ctx: PipelineContext) -> PipelineResult:
        """Execute the KGIZ pipeline."""
        result = PipelineResult(
            pipeline_name=self.name,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Validate KGIZ config exists
            if ctx.config.app.kgiz is None:
                msg = "KGIZ configuration required but not provided"
                raise ValueError(msg)

            # Process each step in the pipeline
            for step in ctx.pipeline_config.steps:
                step_start = datetime.now()
                step_result = StepResult(
                    step_name=step.name,
                    status=PipelineStatus.RUNNING,
                    duration_ms=0,
                )

                try:
                    self._logger.info("Executing step: %s", step.name)

                    if ctx.dry_run:
                        self._logger.info("[DRY RUN] Would execute step: %s", step.name)
                        step_result.status = PipelineStatus.SKIPPED
                    else:
                        await asyncio.sleep(0.01)  # Simulate work
                        step_result.status = PipelineStatus.SUCCESS

                except Exception as e:
                    step_result.status = PipelineStatus.FAILED
                    step_result.error = str(e)
                    if step.on_error == "fail":
                        raise

                finally:
                    step_end = datetime.now()
                    step_result.duration_ms = (step_end - step_start).total_seconds() * 1000
                    result.step_results.append(step_result)

            result.status = PipelineStatus.SUCCESS

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        finally:
            result.completed_at = datetime.now()

        return result


@register_pipeline
class YamlDrivenPipeline(Pipeline):
    """Generic pipeline that executes steps from YAML configuration.

    This pipeline interprets step configurations from YAML and
    dispatches to appropriate handlers based on step type.
    """

    name = "yaml-driven"
    description = "Execute pipeline steps defined in YAML configuration"
    version = "1.0.0"

    async def execute(self, ctx: PipelineContext) -> PipelineResult:
        """Execute the YAML-driven pipeline."""
        result = PipelineResult(
            pipeline_name=ctx.pipeline_config.name,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            for step in ctx.pipeline_config.steps:
                step_start = datetime.now()
                step_result = await self._execute_step(ctx, step)
                step_result.duration_ms = (datetime.now() - step_start).total_seconds() * 1000
                result.step_results.append(step_result)

                if step_result.status == PipelineStatus.FAILED and step.on_error == "fail":
                    result.status = PipelineStatus.FAILED
                    result.error = step_result.error
                    break
            else:
                result.status = PipelineStatus.SUCCESS

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        finally:
            result.completed_at = datetime.now()

        return result

    async def _execute_step(
        self, ctx: PipelineContext, step: "PipelineStep"
    ) -> StepResult:
        """Execute a single pipeline step."""
        from vision.config import PipelineStep, StepType

        result = StepResult(
            step_name=step.name,
            status=PipelineStatus.RUNNING,
            duration_ms=0,
        )

        if ctx.dry_run:
            self._logger.info("[DRY RUN] Would execute: %s (%s)", step.name, step.type)
            result.status = PipelineStatus.SKIPPED
            return result

        try:
            match step.type:
                case StepType.FETCH_GOOGLE:
                    result.output = await self._step_fetch_google(ctx, step)
                case StepType.FETCH_KGIZ:
                    result.output = await self._step_fetch_kgiz(ctx, step)
                case StepType.RESIZE:
                    result.output = await self._step_resize(ctx, step)
                case StepType.CROP:
                    result.output = await self._step_crop(ctx, step)
                case StepType.FILTER:
                    result.output = await self._step_filter(ctx, step)
                case StepType.TRANSFORM:
                    result.output = await self._step_transform(ctx, step)
                case StepType.SAVE:
                    result.output = await self._step_save(ctx, step)
                case StepType.CUSTOM:
                    result.output = await self._step_custom(ctx, step)
                case _:
                    msg = f"Unknown step type: {step.type}"
                    raise ValueError(msg)

            result.status = PipelineStatus.SUCCESS

        except Exception as e:
            self._logger.error("Step %s failed: %s", step.name, e)
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        return result

    async def _step_fetch_google(self, ctx: PipelineContext, step: "PipelineStep") -> dict:
        """Fetch images from Google Images API."""
        self._logger.info("Fetching from Google Images: %s", step.params)
        # Implementation placeholder
        return {"fetched": 0, "source": "google"}

    async def _step_fetch_kgiz(self, ctx: PipelineContext, step: "PipelineStep") -> dict:
        """Fetch/process images with KGIZ service."""
        self._logger.info("Processing with KGIZ: %s", step.params)
        # Implementation placeholder
        return {"processed": 0, "source": "kgiz"}

    async def _step_resize(self, ctx: PipelineContext, step: "PipelineStep") -> dict:
        """Resize images."""
        width = step.params.get("width", 800)
        height = step.params.get("height", 600)
        self._logger.info("Resizing to %dx%d", width, height)
        return {"width": width, "height": height}

    async def _step_crop(self, ctx: PipelineContext, step: "PipelineStep") -> dict:
        """Crop images."""
        self._logger.info("Cropping: %s", step.params)
        return {"cropped": True}

    async def _step_filter(self, ctx: PipelineContext, step: "PipelineStep") -> dict:
        """Apply filters to images."""
        filter_name = step.params.get("filter", "none")
        self._logger.info("Applying filter: %s", filter_name)
        return {"filter": filter_name}

    async def _step_transform(self, ctx: PipelineContext, step: "PipelineStep") -> dict:
        """Apply transformations to images."""
        self._logger.info("Transforming: %s", step.params)
        return {"transformed": True}

    async def _step_save(self, ctx: PipelineContext, step: "PipelineStep") -> dict:
        """Save processed images."""
        output_dir = Path(step.params.get("output_dir", ctx.output_dir))
        self._logger.info("Saving to: %s", output_dir)
        return {"output_dir": str(output_dir)}

    async def _step_custom(self, ctx: PipelineContext, step: "PipelineStep") -> dict:
        """Execute custom step logic."""
        handler = step.params.get("handler")
        if not handler:
            msg = "Custom step requires 'handler' parameter"
            raise ValueError(msg)
        self._logger.info("Executing custom handler: %s", handler)
        return {"handler": handler, "executed": True}


# Type hint import for step execution
from vision.config import PipelineStep  # noqa: E402

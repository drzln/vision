"""Tests for configuration models."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from vision.config import PipelineConfig, PipelineStep, StepType, VisionConfig


class TestPipelineStep:
    """Tests for PipelineStep model."""

    def test_valid_step(self):
        """Test creating a valid pipeline step."""
        step = PipelineStep(
            name="test-step",
            type=StepType.RESIZE,
            params={"width": 800, "height": 600},
        )
        assert step.name == "test-step"
        assert step.type == StepType.RESIZE
        assert step.params["width"] == 800

    def test_invalid_step_name(self):
        """Test that invalid step names are rejected."""
        with pytest.raises(ValueError, match="alphanumeric"):
            PipelineStep(name="invalid name!", type=StepType.RESIZE)

    def test_default_on_error(self):
        """Test default error handling is 'fail'."""
        step = PipelineStep(name="test", type=StepType.SAVE)
        assert step.on_error == "fail"


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_valid_pipeline(self):
        """Test creating a valid pipeline config."""
        pipeline = PipelineConfig(
            name="test-pipeline",
            description="A test pipeline",
            steps=[
                PipelineStep(name="step1", type=StepType.RESIZE),
            ],
        )
        assert pipeline.name == "test-pipeline"
        assert len(pipeline.steps) == 1

    def test_empty_steps_rejected(self):
        """Test that pipelines with no steps are rejected."""
        with pytest.raises(ValueError, match="at least one step"):
            PipelineConfig(name="empty", steps=[])

    def test_invalid_pipeline_name(self):
        """Test that invalid pipeline names are rejected."""
        with pytest.raises(ValueError, match="alphanumeric"):
            PipelineConfig(
                name="invalid name!",
                steps=[PipelineStep(name="step", type=StepType.SAVE)],
            )


class TestVisionConfig:
    """Tests for VisionConfig model."""

    def test_default_config(self):
        """Test creating a default configuration."""
        config = VisionConfig()
        assert config.version == "1.0"
        assert config.app.log_level == "INFO"
        assert config.pipelines == []

    def test_from_yaml(self):
        """Test loading configuration from YAML."""
        yaml_content = """
version: "1.0"
app:
  output_dir: ./test-output
  log_level: DEBUG
pipelines:
  - name: test-pipeline
    steps:
      - name: test-step
        type: resize
        params:
          width: 100
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = VisionConfig.from_yaml(Path(f.name))
            assert config.app.log_level == "DEBUG"
            assert len(config.pipelines) == 1
            assert config.pipelines[0].name == "test-pipeline"

    def test_get_pipeline(self):
        """Test getting a pipeline by name."""
        config = VisionConfig(
            pipelines=[
                PipelineConfig(
                    name="pipeline-a",
                    steps=[PipelineStep(name="step", type=StepType.SAVE)],
                ),
                PipelineConfig(
                    name="pipeline-b",
                    steps=[PipelineStep(name="step", type=StepType.SAVE)],
                ),
            ]
        )

        assert config.get_pipeline("pipeline-a") is not None
        assert config.get_pipeline("pipeline-a").name == "pipeline-a"
        assert config.get_pipeline("nonexistent") is None

    def test_get_enabled_pipelines(self):
        """Test filtering enabled pipelines."""
        config = VisionConfig(
            pipelines=[
                PipelineConfig(
                    name="enabled",
                    enabled=True,
                    steps=[PipelineStep(name="step", type=StepType.SAVE)],
                ),
                PipelineConfig(
                    name="disabled",
                    enabled=False,
                    steps=[PipelineStep(name="step", type=StepType.SAVE)],
                ),
            ]
        )

        enabled = config.get_enabled_pipelines()
        assert len(enabled) == 1
        assert enabled[0].name == "enabled"

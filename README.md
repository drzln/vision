# Vision

Image processing pipelines with Google Images and KGIZ.

## Features

- **YAML Configuration**: Define pipelines declaratively in YAML
- **Pydantic Validation**: Type-safe configuration with runtime validation
- **Custom Pipelines**: Inherit from `Pipeline` base class for custom logic
- **Rich CLI**: Beautiful terminal output with the Rich library
- **Google Images Integration**: Fetch images via Google Custom Search API
- **KGIZ Integration**: Process images with KGIZ service

## Installation

### With Nix (Recommended)

```bash
# Run directly
nix run .#vision -- --help

# Enter development shell
nix develop
```

### With uv

```bash
uv sync
uv run vision --help
```

## Usage

### Initialize Configuration

```bash
vision config init
```

This creates a `vision.yaml` file with example pipelines.

### Validate Configuration

```bash
vision config validate vision.yaml
```

### List Pipelines

```bash
vision pipeline list --config vision.yaml
```

### Run Pipelines

```bash
# Run all enabled pipelines
vision run --config vision.yaml

# Run a specific pipeline
vision run --config vision.yaml --pipeline google-product-images

# Dry run (show what would happen)
vision run --config vision.yaml --dry-run
```

## Custom Pipelines

Create custom pipelines by inheriting from the `Pipeline` base class:

```python
from vision.pipelines import Pipeline, PipelineContext, PipelineResult
from vision.pipelines.base import PipelineStatus
from vision.pipelines.registry import register_pipeline
from datetime import datetime

@register_pipeline
class MyCustomPipeline(Pipeline):
    name = "my-custom"
    description = "My custom image processing pipeline"
    version = "1.0.0"

    async def execute(self, ctx: PipelineContext) -> PipelineResult:
        # Your custom processing logic
        return PipelineResult(
            pipeline_name=self.name,
            status=PipelineStatus.SUCCESS,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
```

## Configuration Schema

```yaml
version: "1.0"

app:
  google_images:
    api_key: "${GOOGLE_API_KEY}"
    search_engine_id: "your-id"
    safe_search: medium
    max_results: 10

  kgiz:
    endpoint: "https://api.kgiz.example.com"
    api_key: "${KGIZ_API_KEY}"
    timeout: 30

  output_dir: ./output
  log_level: INFO
  max_concurrent: 4

pipelines:
  - name: my-pipeline
    description: Pipeline description
    enabled: true
    tags: [tag1, tag2]
    steps:
      - name: step-name
        type: resize  # fetch_google, fetch_kgiz, resize, crop, filter, transform, save, custom
        params:
          width: 800
          height: 600
        on_error: fail  # fail, skip, retry
```

## Development

```bash
# Enter dev shell
nix develop

# Run tests
pytest

# Lint
ruff check .

# Type check
pyright
```

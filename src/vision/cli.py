"""Vision CLI - Image processing pipelines with Google Images and KGIZ."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from vision import __version__
from vision.config import VisionConfig
from vision.pipelines import PipelineContext, PipelineRegistry
from vision.pipelines.base import PipelineStatus

# Rich console for beautiful output
console = Console()
err_console = Console(stderr=True)

# Main Typer app
app = typer.Typer(
    name="vision",
    help="Image processing pipelines with Google Images and KGIZ",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Sub-commands
pipeline_app = typer.Typer(help="Pipeline management commands")
config_app = typer.Typer(help="Configuration management commands")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(config_app, name="config")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True, show_path=False)],
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(
            Panel(
                f"[bold cyan]Vision[/] v{__version__}\n"
                "[dim]Image processing pipelines with Google Images and KGIZ[/]",
                box=box.ROUNDED,
                border_style="cyan",
            )
        )
        raise typer.Exit()


def load_config(config_path: Path | None) -> VisionConfig:
    """Load configuration from file or return defaults."""
    if config_path is None:
        # Look for default config locations
        default_locations = [
            Path("vision.yaml"),
            Path("vision.yml"),
            Path("config/vision.yaml"),
            Path(".vision.yaml"),
        ]
        for loc in default_locations:
            if loc.exists():
                config_path = loc
                break

    if config_path is not None and config_path.exists():
        console.print(f"[dim]Loading config from:[/] {config_path}")
        return VisionConfig.from_yaml(config_path)

    return VisionConfig()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Vision - Image processing pipelines with Google Images and KGIZ.

    Use [bold cyan]vision --help[/] to see available commands.
    """
    pass


@app.command()
def run(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    pipeline_name: Annotated[
        str | None,
        typer.Option("--pipeline", "-p", help="Run specific pipeline by name"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for processed images"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show what would be done without executing"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Run image processing pipelines.

    Execute one or more pipelines defined in the configuration file.
    """
    setup_logging(verbose)

    try:
        config = load_config(config_path)
    except Exception as e:
        err_console.print(f"[red bold]Error loading config:[/] {e}")
        raise typer.Exit(1) from None

    # Get pipelines to run
    if pipeline_name:
        pipeline_config = config.get_pipeline(pipeline_name)
        if pipeline_config is None:
            err_console.print(f"[red bold]Pipeline not found:[/] {pipeline_name}")
            available = [p.name for p in config.pipelines]
            if available:
                err_console.print(f"[dim]Available pipelines:[/] {', '.join(available)}")
            raise typer.Exit(1)
        pipelines = [pipeline_config]
    else:
        pipelines = config.get_enabled_pipelines()

    if not pipelines:
        console.print("[yellow]No pipelines to run.[/]")
        console.print("[dim]Define pipelines in your configuration file.[/]")
        raise typer.Exit(0)

    # Display run info
    console.print()
    console.print(
        Panel(
            f"[bold]Running {len(pipelines)} pipeline(s)[/]\n"
            f"[dim]Output:[/] {output_dir or config.app.output_dir}\n"
            f"[dim]Dry run:[/] {'Yes' if dry_run else 'No'}",
            title="[cyan]Vision[/]",
            box=box.ROUNDED,
        )
    )
    console.print()

    # Run pipelines
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[cyan]Processing pipelines...", total=len(pipelines))

        for pipeline_cfg in pipelines:
            # Get the appropriate pipeline executor based on type
            registry = PipelineRegistry()
            pipeline_type = pipeline_cfg.effective_type

            # Map pipeline types to registered pipeline names
            from vision.config import PipelineType

            type_to_name = {
                PipelineType.YAML_DRIVEN: "yaml-driven",
                PipelineType.STYLE_TRANSFER: "style-transfer",
                PipelineType.GOOGLE_IMAGES: "google-images",
                PipelineType.KGIZ: "kgiz",
            }

            pipeline_name = type_to_name.get(pipeline_type, "yaml-driven")
            pipeline_cls = registry.get(pipeline_name)

            if pipeline_cls is None:
                err_console.print(f"[red]Pipeline type not found: {pipeline_name}[/]")
                err_console.print(f"[dim]Available: {', '.join(registry.list_all())}[/]")
                raise typer.Exit(1)

            pipeline = pipeline_cls(config)

            # For typed pipelines, configure with the config section
            if pipeline_cfg.config and hasattr(pipeline, "configure"):
                from vision.pipelines.style_transfer import StyleTransferPipelineConfig

                if pipeline_type == PipelineType.STYLE_TRANSFER:
                    typed_config = StyleTransferPipelineConfig.model_validate(pipeline_cfg.config)
                    pipeline.configure(typed_config)

            ctx = PipelineContext(
                config=config,
                pipeline_config=pipeline_cfg,
                output_dir=output_dir or config.app.output_dir,
                dry_run=dry_run,
            )

            progress.update(
                overall_task, description=f"[cyan]Running: {pipeline_cfg.name}..."
            )

            result = asyncio.run(pipeline.run(ctx))
            results.append(result)

            progress.advance(overall_task)

    # Display results
    console.print()
    _display_results(results)


def _display_results(results: list) -> None:
    """Display pipeline execution results."""
    table = Table(
        title="Pipeline Results",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )

    table.add_column("Pipeline", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Steps", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Error", style="red dim")

    for result in results:
        status_color = {
            PipelineStatus.SUCCESS: "green",
            PipelineStatus.FAILED: "red",
            PipelineStatus.SKIPPED: "yellow",
        }.get(result.status, "white")

        error_text = result.error or ""
        if result.error and len(result.error) > 50:
            error_text = result.error[:50] + "..."

        table.add_row(
            result.pipeline_name,
            f"[{status_color}]{result.status.value.upper()}[/]",
            f"{result.successful_steps}/{len(result.step_results)}",
            f"{result.duration_ms:.1f}ms",
            error_text,
        )

    console.print(table)

    # Summary
    success_count = sum(1 for r in results if r.status == PipelineStatus.SUCCESS)
    fail_count = sum(1 for r in results if r.status == PipelineStatus.FAILED)

    if fail_count == 0:
        console.print(f"\n[green bold]All {success_count} pipeline(s) completed successfully.[/]")
    else:
        console.print(f"\n[red bold]{fail_count} pipeline(s) failed.[/]")


# ============================================================================
# Pipeline commands
# ============================================================================


@pipeline_app.command("list")
def pipeline_list(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to YAML configuration file"),
    ] = None,
) -> None:
    """List all configured pipelines."""
    config = load_config(config_path)

    if not config.pipelines:
        console.print("[yellow]No pipelines configured.[/]")
        return

    table = Table(
        title="Configured Pipelines",
        box=box.ROUNDED,
        header_style="bold cyan",
    )

    table.add_column("Name", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Description")
    table.add_column("Steps", justify="center")
    table.add_column("Enabled", justify="center")
    table.add_column("Tags", style="dim")

    for pipeline in config.pipelines:
        enabled = "[green]Yes[/]" if pipeline.enabled else "[red]No[/]"
        tags = ", ".join(pipeline.tags) if pipeline.tags else "-"
        pipeline_type = pipeline.effective_type.value
        steps_count = str(len(pipeline.steps)) if pipeline.steps else "[dim]n/a[/]"

        table.add_row(
            pipeline.name,
            pipeline_type,
            pipeline.description or "[dim]No description[/]",
            steps_count,
            enabled,
            tags,
        )

    console.print(table)


@pipeline_app.command("show")
def pipeline_show(
    name: Annotated[str, typer.Argument(help="Pipeline name to show")],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to YAML configuration file"),
    ] = None,
) -> None:
    """Show detailed information about a pipeline."""
    config = load_config(config_path)
    pipeline = config.get_pipeline(name)

    if pipeline is None:
        err_console.print(f"[red bold]Pipeline not found:[/] {name}")
        raise typer.Exit(1)

    # Pipeline info panel
    console.print(
        Panel(
            f"[bold]{pipeline.name}[/]\n"
            f"[dim]{pipeline.description or 'No description'}[/]\n\n"
            f"[cyan]Enabled:[/] {'Yes' if pipeline.enabled else 'No'}\n"
            f"[cyan]Output format:[/] {pipeline.output_format}\n"
            f"[cyan]Input patterns:[/] {', '.join(pipeline.input_patterns)}\n"
            f"[cyan]Tags:[/] {', '.join(pipeline.tags) if pipeline.tags else 'None'}",
            title="[cyan]Pipeline Details[/]",
            box=box.ROUNDED,
        )
    )

    # Steps tree
    tree = Tree(f"[bold cyan]Steps ({len(pipeline.steps)})[/]")
    for i, step in enumerate(pipeline.steps, 1):
        step_node = tree.add(f"[bold]{i}. {step.name}[/] [dim]({step.type.value})[/]")
        step_node.add(f"[dim]On error:[/] {step.on_error}")
        if step.params:
            params_node = step_node.add("[dim]Parameters:[/]")
            for key, value in step.params.items():
                params_node.add(f"{key}: {value}")

    console.print(tree)


@pipeline_app.command("registry")
def pipeline_registry() -> None:
    """List all registered pipeline classes."""
    # Import builtins to ensure they're registered
    from vision.pipelines import builtin  # noqa: F401

    registry = PipelineRegistry()
    pipelines = registry.get_all()

    if not pipelines:
        console.print("[yellow]No pipeline classes registered.[/]")
        return

    table = Table(
        title="Registered Pipeline Classes",
        box=box.ROUNDED,
        header_style="bold cyan",
    )

    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Version", justify="center")

    for name, cls in pipelines.items():
        table.add_row(name, cls.description, cls.version)

    console.print(table)


# ============================================================================
# Config commands
# ============================================================================


@config_app.command("validate")
def config_validate(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to YAML configuration file to validate"),
    ],
) -> None:
    """Validate a YAML configuration file."""
    if not config_path.exists():
        err_console.print(f"[red bold]File not found:[/] {config_path}")
        raise typer.Exit(1)

    try:
        config = VisionConfig.from_yaml(config_path)
        console.print("[green bold]Configuration is valid.[/]")
        console.print(f"[dim]Loaded {len(config.pipelines)} pipeline(s)[/]")

    except Exception as e:
        err_console.print("[red bold]Validation failed:[/]")
        err_console.print(f"  {e}")
        raise typer.Exit(1) from None


@config_app.command("init")
def config_init(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("vision.yaml"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing file"),
    ] = False,
) -> None:
    """Generate a sample configuration file."""
    if output.exists() and not force:
        err_console.print(f"[red bold]File already exists:[/] {output}")
        err_console.print("[dim]Use --force to overwrite[/]")
        raise typer.Exit(1)

    sample_config = """\
# Vision Configuration
# Image processing pipelines with Google Images and KGIZ
version: "1.0"

app:
  # Google Images API configuration
  google_images:
    api_key: "${GOOGLE_API_KEY}"  # Use environment variable
    search_engine_id: "your-search-engine-id"
    safe_search: medium
    max_results: 10

  # KGIZ service configuration
  kgiz:
    endpoint: "https://api.kgiz.example.com"
    api_key: "${KGIZ_API_KEY}"
    timeout: 30

  # Output settings
  output_dir: ./output
  log_level: INFO
  max_concurrent: 4

pipelines:
  # Example: Fetch and process Google Images
  - name: google-product-images
    description: Fetch product images from Google and optimize them
    enabled: true
    tags: [google, products]
    input_patterns:
      - "*.jpg"
      - "*.png"
    output_format: webp
    steps:
      - name: fetch-images
        type: fetch_google
        params:
          query: "product photography"
          num_images: 5
        on_error: fail

      - name: resize
        type: resize
        params:
          width: 1200
          height: 800
          maintain_aspect: true
        on_error: skip

      - name: optimize
        type: filter
        params:
          filter: sharpen
          intensity: 0.5
        on_error: skip

      - name: save-output
        type: save
        params:
          output_dir: ./output/products
          format: webp
          quality: 85

  # Example: KGIZ processing pipeline
  - name: kgiz-enhancement
    description: Enhance images using KGIZ AI service
    enabled: true
    tags: [kgiz, ai, enhancement]
    steps:
      - name: process-with-kgiz
        type: fetch_kgiz
        params:
          operation: enhance
          model: default
        on_error: retry
        retry_count: 3

      - name: save-enhanced
        type: save
        params:
          output_dir: ./output/enhanced
"""

    output.write_text(sample_config)
    console.print(f"[green bold]Configuration created:[/] {output}")
    console.print("[dim]Edit the file to add your API credentials and customize pipelines.[/]")


@config_app.command("show")
def config_show(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to YAML configuration file"),
    ],
) -> None:
    """Display configuration file with syntax highlighting."""
    if not config_path.exists():
        err_console.print(f"[red bold]File not found:[/] {config_path}")
        raise typer.Exit(1)

    content = config_path.read_text()
    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)

    console.print(
        Panel(
            syntax,
            title=f"[cyan]{config_path}[/]",
            box=box.ROUNDED,
        )
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    app()

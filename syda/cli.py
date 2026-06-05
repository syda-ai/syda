"""
Command-line interface for syda synthetic data generation.

Usage:
    syda generate --schema patients.yaml --rows 100
    syda generate --schema schemas/ --rows 50 --output-dir ./output
    syda validate --schema patients.yaml
    syda db infer --db-url sqlite:///mydb.db --output-dir schemas/
    syda db generate --db-url sqlite:///mydb.db --rows 50 --write-back
    syda version
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .schemas import ModelConfig


SCHEMA_EXTENSIONS = {".yaml", ".yml", ".json"}

PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "grok": "GROK_API_KEY",
    "azureopenai": "AZURE_OPENAI_API_KEY",
    # openai_compatible intentionally excluded — it requires --base-url so
    # auto-detection is meaningless; always pass --provider explicitly.
}

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "grok": "grok-2-latest",
    "openai_compatible": "llama3",
    "azureopenai": "gpt-4o-mini",
}


def _detect_provider() -> Optional[str]:
    """Return the first provider whose env var is set."""
    for provider, env_var in PROVIDER_ENV_VARS.items():
        if os.environ.get(env_var):
            return provider
    return None


def _load_schema_files(schema_path: str) -> dict:
    """Load schema(s) from a file or directory. Returns {name: path} mapping."""
    p = Path(schema_path)
    if not p.exists():
        raise click.BadParameter(f"Path does not exist: {schema_path}")

    if p.is_file():
        if p.suffix not in SCHEMA_EXTENSIONS:
            raise click.BadParameter(
                f"Schema file must be YAML or JSON, got: {p.suffix}"
            )
        return {p.stem: str(p)}

    if p.is_dir():
        schemas = {}
        for f in sorted(p.iterdir()):
            if f.suffix in SCHEMA_EXTENSIONS:
                schemas[f.stem] = str(f)
        if not schemas:
            raise click.BadParameter(
                f"No YAML/JSON schema files found in directory: {schema_path}"
            )
        return schemas

    raise click.BadParameter(f"Not a file or directory: {schema_path}")


@click.group()
def main():
    """syda — AI-powered synthetic data generation."""


@main.command("version")
def cmd_version():
    """Print syda version and exit."""
    click.echo(f"syda {__version__}")


@main.command("validate")
@click.option(
    "--schema", "-s",
    required=True,
    metavar="PATH",
    help="Path to a schema file (YAML/JSON) or directory of schema files.",
)
def cmd_validate(schema: str):
    """Validate schema file(s) without generating data."""
    from .schema_loader import SchemaLoader

    try:
        schema_files = _load_schema_files(schema)
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc

    loader = SchemaLoader()
    errors = []
    for name, path in schema_files.items():
        try:
            loader.load_schema(path, schema_name=name)
            click.echo(click.style(f"  [OK] {name} ({path})", fg="green"))
        except Exception as exc:
            errors.append((name, str(exc)))
            click.echo(click.style(f"  [FAIL] {name}: {exc}", fg="red"))

    if errors:
        raise click.ClickException(
            f"{len(errors)} schema(s) failed validation."
        )
    click.echo(click.style(f"\nAll {len(schema_files)} schema(s) valid.", fg="green"))


@main.command("generate")
@click.option(
    "--schema", "-s",
    required=True,
    metavar="PATH",
    help="Path to a schema file (YAML/JSON) or directory of schema files.",
)
@click.option(
    "--rows", "-n",
    default=10,
    show_default=True,
    metavar="N",
    help="Number of rows to generate per table.",
)
@click.option(
    "--output", "-o",
    default=None,
    metavar="FILE",
    help="Output file path (CSV or JSON). Only valid for a single schema.",
)
@click.option(
    "--output-dir",
    default=None,
    metavar="DIR",
    help="Directory to write one output file per schema.",
)
@click.option(
    "--format", "-f",
    "fmt",
    default=None,
    type=click.Choice(["csv", "json"], case_sensitive=False),
    help="Output format (csv or json). Inferred from --output extension when omitted.",
)
@click.option(
    "--provider", "-p",
    default=None,
    type=click.Choice(
        ["anthropic", "openai", "gemini", "grok", "openai_compatible", "azureopenai"],
        case_sensitive=False,
    ),
    help="LLM provider. Auto-detected from env vars when omitted.",
)
@click.option(
    "--model", "-m",
    default=None,
    metavar="NAME",
    help="Model name. Defaults to a sensible model for the chosen provider.",
)
@click.option(
    "--api-key",
    default=None,
    metavar="KEY",
    envvar="SYDA_API_KEY",
    help="API key for the provider. Falls back to the provider's standard env var.",
)
@click.option(
    "--base-url",
    default=None,
    metavar="URL",
    help="Base URL for openai_compatible providers (e.g. http://localhost:11434/v1).",
)
@click.option(
    "--prompt",
    default=None,
    metavar="TEXT",
    help="Optional generation context prompt applied to all schemas.",
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help="Sampling temperature (0.0–1.0).",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    metavar="N",
    help="Max rows per LLM call in direct mode (default: auto-selected).",
)
@click.option(
    "--large-dataset",
    is_flag=True,
    default=False,
    help=(
        "Force code-gen mode: LLM writes Python functions, local execution generates data. "
        "Auto-enabled when --rows > 500."
    ),
)
def cmd_generate(
    schema: str,
    rows: int,
    output: Optional[str],
    output_dir: Optional[str],
    fmt: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    prompt: Optional[str],
    temperature: Optional[float],
    batch_size: Optional[int],
    large_dataset: bool,
):
    """Generate synthetic data from schema(s)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # --- resolve schemas ---
    try:
        schema_files = _load_schema_files(schema)
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc

    multi = len(schema_files) > 1

    # --- validate output options ---
    if output is not None and not output.strip():
        raise click.UsageError("--output cannot be an empty string.")
    if output and multi:
        raise click.UsageError(
            "--output can only be used with a single schema. "
            "Use --output-dir for multiple schemas."
        )
    if output and output_dir:
        raise click.UsageError("Specify either --output or --output-dir, not both.")

    # --- resolve format ---
    resolved_fmt = fmt
    if not resolved_fmt and output:
        ext = Path(output).suffix.lower()
        if ext == ".json":
            resolved_fmt = "json"
        else:
            resolved_fmt = "csv"
    if not resolved_fmt:
        resolved_fmt = "csv"

    # --- resolve provider ---
    if not provider:
        provider = _detect_provider()
    if not provider:
        raise click.UsageError(
            "No LLM provider detected. Set an API key env var "
            "(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROK_API_KEY) "
            "or pass --provider explicitly."
        )

    resolved_model = model or DEFAULT_MODELS.get(provider, "")

    # --- build ModelConfig kwargs ---
    mc_kwargs = dict(provider=provider, model_name=resolved_model)
    if temperature is not None:
        mc_kwargs["temperature"] = temperature
    if batch_size is not None:
        mc_kwargs["batch_size"] = batch_size
    if large_dataset:
        mc_kwargs["generation_mode"] = "codegen"

    extra: dict = {}
    if api_key:
        key_map = {
            "anthropic": "api_key",
            "openai": "api_key",
            "gemini": "api_key",
            "grok": "api_key",
            "openai_compatible": "api_key",
            "azureopenai": "api_key",
        }
        extra[key_map.get(provider, "api_key")] = api_key
    if base_url:
        extra["base_url"] = base_url
    if extra:
        mc_kwargs["extra_kwargs"] = extra

    model_config = ModelConfig(**mc_kwargs)

    # --- build generator ---
    from .generate import SyntheticDataGenerator

    generator_kwargs: dict = dict(model_config=model_config)
    if api_key:
        env_key_kwarg = {
            "anthropic": "anthropic_api_key",
            "openai": "openai_api_key",
            "gemini": "gemini_api_key",
            "grok": "grok_api_key",
        }
        kwarg = env_key_kwarg.get(provider)
        if kwarg:
            generator_kwargs[kwarg] = api_key

    generator = SyntheticDataGenerator(**generator_kwargs)

    # --- run generation ---
    click.echo(
        f"Generating {rows} row(s) for {len(schema_files)} schema(s) "
        f"using {provider}/{resolved_model} ..."
    )

    prompts = {name: prompt for name in schema_files} if prompt else None

    resolved_output_dir = output_dir
    if not resolved_output_dir and not output:
        resolved_output_dir = "."
        click.echo(
            click.style(
                "Warning: no --output-dir specified, writing to current directory.",
                fg="yellow",
            )
        )

    try:
        results = generator.generate_for_schemas(
            schemas=schema_files,
            prompts=prompts,
            default_sample_size=rows,
            output_dir=resolved_output_dir if not output else None,
            output_format=resolved_fmt,
        )
    except Exception as exc:
        raise click.ClickException(f"Generation failed: {exc}") from exc

    # --- handle single-file output ---
    if output and not multi:
        table_name = next(iter(schema_files))
        df = results.get(table_name)
        if df is None:
            df = next(iter(results.values()))
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if resolved_fmt == "json":
            df.to_json(out_path, orient="records", indent=2)
        else:
            df.to_csv(out_path, index=False)
        click.echo(
            click.style(f"  Wrote {len(df)} rows → {out_path}", fg="green")
        )
    else:
        # output_dir was already passed to generate_for_schemas; just report
        base = Path(resolved_output_dir or ".")
        for name, df in results.items():
            ext = "json" if resolved_fmt == "json" else "csv"
            click.echo(
                click.style(
                    f"  Wrote {len(df)} rows → {base / f'{name}.{ext}'}",
                    fg="green",
                )
            )

    click.echo(click.style("\nDone.", fg="green", bold=True))


# ---------------------------------------------------------------------------
# syda db  — database integration commands
# ---------------------------------------------------------------------------

@main.group("db")
def cmd_db():
    """Database integration: infer schemas and generate data from a live DB."""


def _build_generator(
    provider, model, api_key, base_url, temperature,
    batch_size: Optional[int] = None, large_dataset: bool = False,
):
    """Shared helper: build a SyntheticDataGenerator from CLI options."""
    if not provider:
        provider = _detect_provider()
    if not provider:
        raise click.UsageError(
            "No LLM provider detected. Set an API key env var "
            "(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROK_API_KEY) "
            "or pass --provider explicitly."
        )

    resolved_model = model or DEFAULT_MODELS.get(provider, "")
    mc_kwargs: dict = dict(provider=provider, model_name=resolved_model)
    if temperature is not None:
        mc_kwargs["temperature"] = temperature
    if batch_size is not None:
        mc_kwargs["batch_size"] = batch_size
    if large_dataset:
        mc_kwargs["generation_mode"] = "codegen"

    extra: dict = {}
    if api_key:
        extra["api_key"] = api_key
    if base_url:
        extra["base_url"] = base_url
    if extra:
        mc_kwargs["extra_kwargs"] = extra

    model_config = ModelConfig(**mc_kwargs)

    from .generate import SyntheticDataGenerator

    gen_kwargs: dict = dict(model_config=model_config)
    if api_key:
        kwarg = {
            "anthropic": "anthropic_api_key",
            "openai": "openai_api_key",
            "gemini": "gemini_api_key",
            "grok": "grok_api_key",
        }.get(provider)
        if kwarg:
            gen_kwargs[kwarg] = api_key

    return SyntheticDataGenerator(**gen_kwargs), provider, resolved_model


@cmd_db.command("infer")
@click.option(
    "--db-url", "-d",
    required=True,
    metavar="URL",
    help=(
        "SQLAlchemy database URL. Examples:\n"
        "  sqlite:///mydb.db\n"
        "  postgresql://user:pass@host/dbname\n"
        "  mysql+pymysql://user:pass@host/dbname"
    ),
)
@click.option(
    "--output-dir", "-o",
    required=True,
    metavar="DIR",
    help="Directory to write inferred schema files.",
)
@click.option(
    "--tables", "-t",
    default=None,
    metavar="TABLE,...",
    help="Comma-separated list of tables to infer. Defaults to all tables.",
)
@click.option(
    "--format", "-f",
    "fmt",
    default="yaml",
    show_default=True,
    type=click.Choice(["yaml", "json"], case_sensitive=False),
    help="Schema file format.",
)
def cmd_db_infer(db_url: str, output_dir: str, tables: Optional[str], fmt: str):
    """Infer schemas from a database and save them as YAML/JSON files."""
    from .db_schema_loader import DatabaseSchemaLoader

    table_list = [t.strip() for t in tables.split(",")] if tables else None

    try:
        loader = DatabaseSchemaLoader(db_url)
    except Exception as exc:
        raise click.ClickException(f"Could not connect to database: {exc}") from exc

    click.echo(f"Connected to: {db_url}")
    click.echo(f"Inferring schemas → {output_dir} ...")

    try:
        schema_files = loader.save_schemas(
            output_dir=output_dir,
            table_names=table_list,
            format=fmt,
        )
    except Exception as exc:
        raise click.ClickException(f"Schema inference failed: {exc}") from exc

    for name, path in schema_files.items():
        click.echo(click.style(f"  [OK] {name} → {path}", fg="green"))

    click.echo(
        click.style(f"\nInferred {len(schema_files)} schema(s).", fg="green", bold=True)
    )


@cmd_db.command("generate")
@click.option(
    "--db-url", "-d",
    required=True,
    metavar="URL",
    help=(
        "SQLAlchemy database URL. Examples:\n"
        "  sqlite:///mydb.db\n"
        "  postgresql://user:pass@host/dbname\n"
        "  mysql+pymysql://user:pass@host/dbname"
    ),
)
@click.option(
    "--rows", "-n",
    default=10,
    show_default=True,
    metavar="N",
    help="Number of rows to generate per table.",
)
@click.option(
    "--tables", "-t",
    default=None,
    metavar="TABLE,...",
    help="Comma-separated list of tables to generate. Defaults to all tables.",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    metavar="DIR",
    help="Directory to write generated CSV/JSON files. Skipped if not provided.",
)
@click.option(
    "--format", "-f",
    "fmt",
    default="csv",
    show_default=True,
    type=click.Choice(["csv", "json"], case_sensitive=False),
    help="Output file format when --output-dir is set.",
)
@click.option(
    "--write-back", "-w",
    is_flag=True,
    default=False,
    help="Write generated rows back into the database.",
)
@click.option(
    "--if-exists",
    default="append",
    show_default=True,
    type=click.Choice(["append", "replace", "fail"], case_sensitive=False),
    help=(
        "Behaviour when target table already has rows (only used with --write-back):\n"
        "  append  — add rows (default)\n"
        "  replace — truncate then insert\n"
        "  fail    — raise an error if table is non-empty"
    ),
)
@click.option(
    "--provider", "-p",
    default=None,
    type=click.Choice(
        ["anthropic", "openai", "gemini", "grok", "openai_compatible", "azureopenai"],
        case_sensitive=False,
    ),
    help="LLM provider. Auto-detected from env vars when omitted.",
)
@click.option(
    "--model", "-m",
    default=None,
    metavar="NAME",
    help="Model name. Defaults to a sensible model for the chosen provider.",
)
@click.option(
    "--api-key",
    default=None,
    metavar="KEY",
    envvar="SYDA_API_KEY",
    help="API key for the provider. Falls back to the provider's standard env var.",
)
@click.option(
    "--base-url",
    default=None,
    metavar="URL",
    help="Base URL for openai_compatible providers.",
)
@click.option(
    "--prompt",
    default=None,
    metavar="TEXT",
    help="Optional generation context prompt applied to all tables.",
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help="Sampling temperature (0.0–1.0).",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    metavar="N",
    help="Max rows per LLM call in direct mode (default: auto-selected).",
)
@click.option(
    "--large-dataset",
    is_flag=True,
    default=False,
    help=(
        "Force code-gen mode: LLM writes Python functions, local execution generates data. "
        "Auto-enabled when --rows > 500."
    ),
)
def cmd_db_generate(
    db_url: str,
    rows: int,
    tables: Optional[str],
    output_dir: Optional[str],
    fmt: str,
    write_back: bool,
    if_exists: str,
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    prompt: Optional[str],
    temperature: Optional[float],
    batch_size: Optional[int],
    large_dataset: bool,
):
    """Infer schemas from a database, generate synthetic data, and optionally write it back."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from .db_schema_loader import DatabaseSchemaLoader

    table_list = [t.strip() for t in tables.split(",")] if tables else None

    # --- connect and infer schemas ---
    try:
        loader = DatabaseSchemaLoader(db_url)
    except Exception as exc:
        raise click.ClickException(f"Could not connect to database: {exc}") from exc

    click.echo(f"Connected to: {db_url}")

    try:
        schemas = loader.load_schemas(table_names=table_list)
    except Exception as exc:
        raise click.ClickException(f"Schema inference failed: {exc}") from exc

    click.echo(
        f"Inferred {len(schemas)} table(s): {', '.join(schemas.keys())}"
    )

    # --- build generator ---
    try:
        generator, resolved_provider, resolved_model = _build_generator(
            provider, model, api_key, base_url, temperature,
            batch_size=batch_size, large_dataset=large_dataset,
        )
    except click.UsageError:
        raise

    click.echo(
        f"Generating {rows} row(s) per table using {resolved_provider}/{resolved_model} ..."
    )

    prompts = {name: prompt for name in schemas} if prompt else None

    try:
        results = generator.generate_for_schemas(
            schemas=schemas,
            prompts=prompts,
            default_sample_size=rows,
            output_dir=output_dir,
            output_format=fmt,
        )
    except Exception as exc:
        raise click.ClickException(f"Generation failed: {exc}") from exc

    # --- report output files ---
    if output_dir:
        base = Path(output_dir)
        for name, df in results.items():
            click.echo(
                click.style(
                    f"  Wrote {len(df)} rows → {base / f'{name}.{fmt}'}",
                    fg="green",
                )
            )
    else:
        for name, df in results.items():
            click.echo(click.style(f"  Generated {len(df)} rows for '{name}'", fg="cyan"))

    # --- write back ---
    if write_back:
        click.echo(f"\nWriting back to database (if_exists={if_exists}) ...")
        try:
            loader.write_to_database(results, if_exists=if_exists)
        except Exception as exc:
            raise click.ClickException(
                f"Write-back failed (some tables may have been partially written): {exc}"
            ) from exc

    click.echo(click.style("\nDone.", fg="green", bold=True))

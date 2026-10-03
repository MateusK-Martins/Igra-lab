import tomllib
from pathlib import Path

from src.config.errors import ConfigError


def load_toml(path: Path) -> dict[str, object]:
    """Load one TOML configuration file into a dictionary."""

    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")

    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")

    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            f"Invalid TOML in configuration file {path}: {error}"
        ) from error
    except OSError as error:
        raise ConfigError(
            f"Could not read configuration file {path}: {error}"
        ) from error

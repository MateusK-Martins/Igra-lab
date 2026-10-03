"""Resolve environment-variable placeholders in parsed configuration data."""

import os
import re
from collections.abc import Mapping

from src.config.errors import ConfigError

ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_environment_values(value: object) -> object:
    """Recursively replace `${VARIABLE}` placeholders with environment values."""
    if isinstance(value, str):
        return ENVIRONMENT_VARIABLE_PATTERN.sub(_environment_value, value)

    if isinstance(value, list):
        return [resolve_environment_values(item) for item in value]

    if isinstance(value, Mapping):
        return {key: resolve_environment_values(item) for key, item in value.items()}

    return value


def _environment_value(match: re.Match[str]) -> str:
    """Return one required environment variable referenced in a config value."""
    variable_name = match.group(1)
    variable_value = os.environ.get(variable_name)

    if variable_value is None:
        raise ConfigError(
            f"Configuration references missing environment variable {variable_name!r}"
        )

    return variable_value

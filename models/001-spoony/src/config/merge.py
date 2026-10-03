from copy import deepcopy

from src.config.errors import ConfigError


def merge_configs(*configs: dict[str, object]) -> dict[str, object]:
    if len(configs) == 1:
        return deepcopy(configs[0])
    if len(configs) == 0:
        raise ConfigError(
            "Not enough arguments: merge_configs needs at least one config."
        )

    result: dict[str, object] = deepcopy(configs[0])

    for config in configs[1:]:
        for key, value in config.items():
            if key not in result:
                result[key] = deepcopy(value)
                continue

            if isinstance(value, dict) != isinstance(result[key], dict):
                raise ConfigError(
                    f"Type mismatch during config merging. {key} : {type(value)} != {type(result[key])}"
                )

            if isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)

    return result

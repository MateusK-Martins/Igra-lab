from pathlib import Path

from src.config.errors import ConfigError


def find_project_root(start: Path) -> Path:
    """Find a model directory by walking upward to pyproject.toml."""

    current = start.expanduser().resolve()

    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise ConfigError(
        f"Could not find a model project root while searching from {start}"
    )


def project_root() -> Path:
    """Return the model root regardless of the terminal's current directory."""
    return find_project_root(Path(__file__))


def _path_inside_project(directory: str, relative_path: str) -> Path:
    """Resolve a path inside a named model project directory."""
    root = project_root()
    path = (root / directory / relative_path).resolve()

    if root not in path.parents and path != root:
        raise ConfigError(f"Path {relative_path!r} escapes the model project directory")

    return path


def data_path(relative_path: str) -> Path:
    """Resolve a path inside data/."""
    return _path_inside_project("data", relative_path)


def results_path(relative_path: str) -> Path:
    """Resolve a path inside results/."""
    return _path_inside_project("results", relative_path)

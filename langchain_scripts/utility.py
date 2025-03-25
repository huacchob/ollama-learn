from pathlib import Path


def find_root_directory(file: str) -> Path:
    """Find the root directory of the project.

    Args:
        file (str): The full path to the file.

    Returns:
        Path: The root directory of the project.
    """
    current_dir: Path = Path(file).parent
    while "poetry.lock" not in [i.name for i in current_dir.iterdir()]:
        current_dir = current_dir.parent
    return current_dir

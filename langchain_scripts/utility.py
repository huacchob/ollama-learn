from pathlib import Path


def find_root_directory(file: str) -> Path:
    return Path(file).parent.parent

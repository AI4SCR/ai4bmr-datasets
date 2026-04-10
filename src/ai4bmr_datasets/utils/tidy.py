import re
from pathlib import Path

def filter_paths(
    dir_path: Path,
    include_files: bool = True,
    include_dirs: bool = True,
    exclude_hidden: bool = True,
    as_posix: bool = True
) -> list[Path]:
    """
    List files and/or directories from dir_path, excluding items that match
    any of the given predicates.

    Args:
        dir_path (Path): Directory to list.
        include_files (bool): Include files if True.
        include_dirs (bool): Include directories if True.
        exclude_hidden (bool): If True, exclude hidden files and directories.

    Returns:
        list[Path]: Filtered list of paths.
    """
    paths = []
    for f in dir_path.iterdir():
        if exclude_hidden and f.name.startswith("."):
            continue
        if f.is_file() and include_files:
            paths.append(f)
        elif f.is_dir() and include_dirs:
            paths.append(f)

    return [i.as_posix() for i in paths] if as_posix else paths



def tidy_name(name: str, to_lower: bool = True, split_camel_case: bool = True) -> str:
    """
    Tidy a string by removing whitespace, special characters, and converting to lowercase.

    Args:
        name: string to tidy

    Returns:
        str
    """
    # 1. Remove leading and trailing whitespace
    name = name.strip()

    # 2. Replace white spaces with underscores
    name = re.sub(r"\s+", "_", name)

    # 3. Split CamelCase into separate words with underscores
    if split_camel_case:
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)

    # 4. Replace special symbols with underscores
    name = re.sub(r"[^\w\s]", "_", name)

    # 5. Convert to lowercase
    if to_lower:
        name = name.lower()

    # 6. Remove multiple underscores in a row (if any), and leading/trailing underscores
    name = re.sub(r"_+", "_", name).strip("_")

    return name

# core/utils/path.py

"""
Path utilities for the project.
"""

from pathlib import Path
from functools import lru_cache

@lru_cache()
def get_project_root() -> Path:
    """Get the absolute path to the project root directory.
    
    This function traverses up from the current file until it finds the 'core' directory,
    then returns its parent which is the project root.
    
    The result is cached so the computation only happens once.
    
    Returns:
        Path: Absolute path to the project root directory
    
    Example:
        >>> from core.utils.path import get_project_root
        >>> project_root = get_project_root()
        >>> log_dir = project_root / 'logs'
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == 'core':
            return parent.parent
    raise RuntimeError("Could not find project root (looking for parent of 'core' directory)") 

if __name__ == "__main__":
    # Test get_project_root function
    try:
        project_root = get_project_root()
        print(f"Project root directory: {project_root}")
    except RuntimeError as e:
        print(f"Error: {e}")

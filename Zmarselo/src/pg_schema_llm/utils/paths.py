import os


def get_dataset_name(path):
    """
    Extract dataset name from a folder path.

    Logic moved 1:1 from repeated usage:
    os.path.basename(os.path.normpath(path))
    """
    return os.path.basename(os.path.normpath(path))


def ensure_dir(path):
    """
    Ensure directory exists.

    Logic moved 1:1 from repeated:
    os.makedirs(path, exist_ok=True)
    """
    os.makedirs(path, exist_ok=True)

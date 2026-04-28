import os


def get_instance_prefix(dataset_name: str, split: str = "test", level: str = "function") -> str:
    prefix = dataset_name if split == "test" else f"{dataset_name}-{split}"
    if level == "file":
        return f"{prefix}_"
    return f"{prefix}-{level}_"


def list_local_instance_dirs(
    dataset_dir: str,
    dataset_name: str,
    split: str = "test",
    level: str = "function",
) -> list[str]:
    prefix = get_instance_prefix(dataset_name, split, level)
    return sorted(
        item
        for item in os.listdir(dataset_dir)
        if item.startswith(prefix) and os.path.isdir(os.path.join(dataset_dir, item))
    )


def instance_id_from_dir(
    instance_dir: str,
    dataset_name: str,
    split: str = "test",
    level: str = "function",
) -> str:
    prefix = get_instance_prefix(dataset_name, split, level)
    return instance_dir.removeprefix(prefix)

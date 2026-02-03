from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)


def get_volume_path(catalog: str, schema: str, volume: str) -> str:
    return f"/Volumes/{catalog}/{schema}/{volume}"


def is_local_env() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" not in os.environ


def resolve_input_root(
    catalog: str,
    schema: str,
    volume: str,
    local_input_path: Optional[str] = None,
) -> Path:
    if is_local_env() and local_input_path:
        return Path(local_input_path).expanduser().resolve()
    return Path(get_volume_path(catalog, schema, volume))


def resolve_output_root(
    catalog: str,
    schema: str,
    volume: str,
    local_output_path: Optional[str] = None,
) -> Path:
    if is_local_env() and local_output_path:
        output_root = Path(local_output_path).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root
    return Path(get_volume_path(catalog, schema, volume))


def resolve_sample_inputs(
    sample_inputs: Iterable[str],
    local_input_path: Optional[str] = None,
) -> List[str]:
    if not is_local_env() or not local_input_path:
        return list(sample_inputs)

    local_root = Path(local_input_path).expanduser().resolve()
    resolved = []
    for path in sample_inputs:
        if path.startswith("/Volumes/"):
            resolved.append(str(local_root / Path(path).name))
        else:
            resolved.append(str(Path(path).expanduser().resolve()))
    return resolved


def copy_local_inputs(source_dir: Union[str, Path], dest_dir: Union[str, Path]) -> int:
    source_path = Path(source_dir).expanduser().resolve()
    dest_path = Path(dest_dir).expanduser().resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        logger.warning("Source directory does not exist: %s", source_path)
        return 0

    copied = 0
    for item in source_path.iterdir():
        if not item.is_file():
            continue
        target = dest_path / item.name
        shutil.copy2(item, target)
        copied += 1

    logger.info("Copied %s files from %s to %s", copied, source_path, dest_path)
    return copied


def get_file_list(directory: Union[str, Path], pattern: str = "*.pdf") -> List[Path]:
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning("Directory does not exist: %s", directory)
        return []

    files = list(dir_path.rglob(pattern))
    logger.info("Found %s files matching %s in %s", len(files), pattern, directory)
    return files


def sanitize_filename(filename: str) -> str:
    path = Path(filename)
    stem = path.stem.replace("%20", "_")
    suffix = path.suffix

    sanitized = re.sub(r"[^\w\s-]", "", stem)
    sanitized = sanitized.replace(" ", "_")
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_").lower()
    return f"{sanitized}{suffix}"


def print_processing_summary(results: Iterable[Dict], method_name: str) -> None:
    results = list(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful

    print("\n" + "=" * 60)
    print(f"{method_name.upper()} PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if results:
        success_rate = (successful / len(results)) * 100
        print(f"Success rate: {success_rate:.1f}%")

    failed_results = [r for r in results if r.get("status") == "error"]
    if failed_results:
        print("\nFailed files:")
        for result in failed_results[:5]:
            file_name = Path(
                result.get("file")
                or result.get("input_path")
                or result.get("input_file")
                or "unknown"
            ).name
            error = result.get("error", "Unknown error")
            print(f"  {file_name}: {error}")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more")

"""
Helper utilities for the project.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Union


def _natural_sort_key(path: Path) -> List[Union[int, str]]:
    """
    Sort key that treats digit runs as numbers (e.g., img_2.png < img_10.png).
    """
    parts = re.split(r"(\d+)", path.name)
    key: List[Union[int, str]] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def create_png_animation(
    directory_path: Union[str, Path],
    *,
    output_filename: str = "animation.gif",
    fps: float = 8.0,
    loop: int = 0,
) -> Path:
    """
    Create an animated GIF from all `.png` files in `directory_path`.

    The output file is written into the same directory.

    Args:
        directory_path: Folder containing `.png` frames.
        output_filename: Name of the GIF to write (e.g. "epoch.gif").
        fps: Frames per second (converted to per-frame duration in ms).
        loop: GIF loop count. 0 means loop forever.

    Returns:
        Path to the created GIF.
    """
    directory = Path(directory_path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pngs: Sequence[Path] = sorted(directory.glob("*.png"), key=_natural_sort_key)
    if not pngs:
        raise ValueError(f"No .png files found in {directory}")

    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Pillow is required to create GIF animations. Install with `pip install pillow`."
        ) from e

    images: List["Image.Image"] = []
    for p in pngs:
        img = Image.open(p)
        # Convert to a GIF-friendly mode; RGBA is fine, but palette conversion happens on save.
        images.append(img.convert("RGBA"))

    if fps <= 0:
        raise ValueError("fps must be > 0")
    duration_ms = int(round(1000.0 / fps))

    output_path = (directory / output_filename).resolve()
    if output_path.suffix.lower() != ".gif":
        output_path = output_path.with_suffix(".gif")

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=2,
    )

    # Close file handles (important on some platforms).
    for im in images:
        try:
            im.close()
        except Exception:
            pass

    return output_path



if __name__ == "__main__":
    create_png_animation(directory_path="BindsNet/results/spike_distribution", output_filename="animation1.gif", fps=5)
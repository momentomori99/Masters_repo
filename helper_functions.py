"""
Helper utilities for the project.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Union
import numpy as np


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

def add_accuracy_to_results(results_final_array):
    # Append a new row to the numpy array results_final_array and resave the file
    new_row = np.array([0.7203, 0.0, 0.017858303375542162, 0.0, 18.8614, 0.0, 7.0, 0.0, 2.0, 0.0, 77.2])
    results_final_array = np.vstack([results_final_array, new_row])
    np.save("results/results_final.npy", results_final_array)


def show_results(results_final_array):
    import matplotlib.pyplot as plt
    import numpy as np

    param_names = ["CV", "CV_std", "rho_mean", "rho_mean_std", "rate", "rate_std", "g", "g_std", "eta", "eta_std", "acc"]
    for result in results_final_array:
        print(f"CV: {result[0]} +/- {result[1]}")
        print(f"rho_mean: {result[2]} +/- {result[3]}")
        print(f"rate: {result[4]} +/- {result[5]}")
        print(f"g: {result[6]} +/- {result[7]}")
        print(f"eta: {result[8]} +/- {result[9]}")
        print(f"acc: {result[10]}")
        print("--------------------------------")

    print("Results sorted by highest accuracy:")

    # Extract the relevant columns
    CVs = results_final_array[:, 0]
    rho_means = results_final_array[:, 2]
    rates = results_final_array[:, 4]
    gs = results_final_array[:, 6]
    etas = results_final_array[:, 8]
    accuracies = results_final_array[:, 10]

    # Create the figure and scatter plot for the accuracy heatmap (CV vs rho_mean)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(CVs, rho_means, c=accuracies, cmap='viridis', s=120, edgecolor='k')
    plt.xlabel("CV")
    plt.ylabel("rho_mean")
    plt.title("Accuracy Heatmap (color) vs CV (x) and rho_mean (y)")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Accuracy (%)')
    plt.tight_layout()
    plt.show()
    print("--------------------------------")

    # Create correlation matrix
    # Use only the mean of each metric and the accuracy (not std)
    # Columns: CV (0), rho_mean (2), rate (4), g (6), eta (8), acc (10)
    cols = [0, 2, 4, 6, 8, 10]
    param_corr_names = ["CV", "rho_mean", "rate", "g", "eta", "acc"]
    corr_data = results_final_array[:, cols]  # shape: (num_results, 6)
    corr_matrix = np.corrcoef(corr_data, rowvar=False)
    print("Correlation matrix (CV, rho_mean, rate, g, eta, acc):")
    for i, row in enumerate(corr_matrix):
        print(f"{param_corr_names[i]:>9}: {['{:.3f}'.format(v) for v in row]}")

    # Also show correlation (Pearson) of each parameter with accuracy
    print("\nPearson correlation of each parameter with accuracy:")
    for i, pname in enumerate(param_corr_names[:-1]):
        corr = corr_matrix[i, -1]
        print(f"{pname:>9} & acc: {corr:.3f}")

    # Optionally, show correlation matrix as a heatmap
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(6), param_corr_names, rotation=45, ha='right')
    plt.yticks(range(6), param_corr_names)
    plt.colorbar(im, label="Correlation Coefficient")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # Print results sorted by highest accuracy
    sorted_indices = results_final_array[:, 10].argsort()[::-1]  # Descending order
    for rank, idx in enumerate(sorted_indices):
        result = results_final_array[idx]
        print(f"Rank {rank+1}:")
        print(f"CV: {result[0]} +/- {result[1]}")
        print(f"rho_mean: {result[2]} +/- {result[3]}")
        print(f"rate: {result[4]} +/- {result[5]}")
        print(f"g: {result[6]} +/- {result[7]}")
        print(f"eta: {result[8]} +/- {result[9]}")
        print(f"acc: {result[10]}")
        print("--------------------------------")



if __name__ == "__main__":
    create_png_animation(directory_path="BindsNet/results/testing", output_filename="animation1.gif", fps=3)
    #results_final_array = np.load("results/results_final.npy")
    # Remove any row in results_final_array where accuracy (last column) is exactly 0.772
    #results_final_array = results_final_array[results_final_array[:, -1] != 0.772]
    #show_results(results_final_array)
    #add_accuracy_to_results(results_final_array)
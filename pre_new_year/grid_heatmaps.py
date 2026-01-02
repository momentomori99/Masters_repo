import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import brian2 as b2

from LIF_brunels_network import simulate_brunels_network

try:
    from tqdm import tqdm  # optional, for nice progress bars
    HAS_TQDM = True
except Exception:
    tqdm = None
    HAS_TQDM = False


def safe_pca_components(X: np.ndarray, max_components: int = 50) -> int:
    n_samples, n_features = X.shape
    return max(1, min(max_components, min(n_samples, n_features) - 1))


def compute_cv_accuracy(estimator, X: np.ndarray, y: np.ndarray) -> float:
    n_components = safe_pca_components(X, max_components=50)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components),
            estimator
        )
        cvres = cross_validate(pipeline, X, y, cv=5, scoring='accuracy', return_train_score=False)
    return float(cvres['test_score'].mean())


def run_grid_once(X_rates: np.ndarray, y: np.ndarray, g_value: float, eta_value: float) -> np.ndarray:
    X_output = []

    if HAS_TQDM:
        iterator = enumerate(tqdm(X_rates, desc=f"obs g={g_value:.2f}, η={eta_value:.2f}", leave=False))
    else:
        iterator = enumerate(X_rates)

    total_obs = len(X_rates)
    for idx, obs in iterator:
        obs_rates = obs * b2.Hz
        vE, _, spE, _, _, _, _, _, _ = simulate_brunels_network(
            input_data=obs_rates,
            g_strength=g_value,
            eta=eta_value
        )
        n_recorded = vE.v.shape[0]
        spike_counts = np.bincount(spE.i, minlength=n_recorded)
        X_output.append(spike_counts)
        if not HAS_TQDM:
            # Print lightweight progress every 10 obs or at the end
            if (idx + 1) % 10 == 0 or (idx + 1) == total_obs:
                print(f"    obs {idx + 1}/{total_obs} done")

    X_output = np.asarray(X_output)
    return X_output


def main() -> None:
    # Load dataset (rates as floats; multiply by Hz before simulation)
    X_rates = np.load('data/iris_X_rates.npy', allow_pickle=True)
    y = np.load('data/iris_y.npy', allow_pickle=True)

    # Grid (adjust as needed; large grids will be slow)
    g_values = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]) #np.linspace(4, 6.0, 1)
    eta_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1, 1.2]) #np.linspace(0.8, 0.9, 1)

    # Classifiers to evaluate
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RidgeClassifier': RidgeClassifier(),
        'KNN-5': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    }

    # Storage for results
    acc_mats: dict[str, np.ndarray] = {name: np.zeros((len(eta_values), len(g_values)), dtype=float) for name in models}

    # Sweep
    total_grid = len(eta_values) * len(g_values)
    grid_idx = 0
    grid_pbar = tqdm(total=total_grid, desc="Grid", position=0) if HAS_TQDM else None
    for ei, eta in enumerate(eta_values):
        for gi, g in enumerate(g_values):
            if not HAS_TQDM:
                print("================================================")
                print(f"[{grid_idx + 1}/{total_grid}] g={g:.3f}, η={eta:.3f}")

            X_out = run_grid_once(X_rates, y, g_value=g, eta_value=eta)

            # Evaluate models
            for name, estimator in models.items():
                acc = compute_cv_accuracy(estimator, X_out, y)
                acc_mats[name][ei, gi] = acc
                print(f"  {name:16s} | cv acc: {acc:.4f}")
            grid_idx += 1
            if HAS_TQDM and grid_pbar is not None:
                grid_pbar.update(1)

    # Plot heatmaps (2x2 grid)
    sns.set(style="white", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    titles = list(models.keys())
    for ax, title in zip(axes, titles):
        acc_matrix = acc_mats[title]
        hm = sns.heatmap(
            acc_matrix,
            xticklabels=[f"{g:.2f}" for g in g_values],
            yticklabels=[f"{e:.2f}" for e in eta_values],
            cmap="viridis",
            annot=True,
            fmt=".2f",
            cbar=True,
            ax=ax
        )
        ax.set_xlabel('g')
        ax.set_ylabel('η')
        ax.set_title(title)

    # Improve figure aesthetics
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('5-fold CV Accuracy across (g, η)', y=1.02)
    plt.savefig('heatmaps_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    if HAS_TQDM and grid_pbar is not None:
        grid_pbar.close()


if __name__ == '__main__':
    main()



import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pairs_to_xy(pairs):
    if len(pairs) == 0:
        raise ValueError("`pairs` must contain at least one (features, label) sample.")
    X = torch.stack([p[0] for p in pairs])
    y = torch.tensor([int(p[1]) for p in pairs], dtype=torch.long)
    return X, y


def plot_tsne(
    pairs,
    perplexity=30,
    max_iter=1000,
    random_state=42,
    figsize=(10, 8),
    title="t-SNE of reservoir features",
):
    """
    Plot a 2D t-SNE embedding directly from (features, label) pairs.
    """
    X, y = _pairs_to_xy(pairs)
    X_np = _to_numpy(X)
    y_np = _to_numpy(y).astype(int)

    if X_np.ndim != 2:
        raise ValueError(f"`features` must be 2D, got shape {X_np.shape}.")
    if y_np.ndim != 1:
        y_np = y_np.reshape(-1)
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("`features` and `labels` must have the same number of samples.")
    if X_np.shape[0] < 2:
        raise ValueError("Need at least 2 samples for t-SNE.")

    effective_perplexity = min(perplexity, max(1, X_np.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )
    X_emb = tsne.fit_transform(X_np)

    classes = np.unique(y_np)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    plt.figure(figsize=figsize)
    for idx, cls in enumerate(classes):
        mask = y_np == cls
        plt.scatter(
            X_emb[mask, 0],
            X_emb[mask, 1],
            s=25,
            alpha=0.8,
            color=colors[idx],
            label=f"{cls}",
        )

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="Class", loc="best")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    return X_emb


def plot_confusion_heatmap(
    pairs,
    normalize=True,
    class_names=None,
    cmap="Blues",
    figsize=(8, 6),
    title=None,
):
    """
    Plot confusion heatmap directly from pairs.

    Predicted labels are built with a nearest-centroid classifier on the same pairs.
    """
    X, y_true = _pairs_to_xy(pairs)
    X_np = _to_numpy(X)
    y_true_np = _to_numpy(y_true).astype(int)

    classes = np.unique(y_true_np)
    centroids = []
    for cls in classes:
        centroids.append(X_np[y_true_np == cls].mean(axis=0))
    centroids = np.stack(centroids, axis=0)  # (C, D)

    # Predict class by nearest centroid (euclidean distance).
    dists = ((X_np[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    y_pred = classes[np.argmin(dists, axis=1)]

    cm = confusion_matrix(y_true_np, y_pred, labels=classes)
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sums, 1e-12))

    if class_names is None:
        tick_labels = [str(c) for c in classes]
    else:
        if len(class_names) != len(classes):
            raise ValueError("`class_names` length must match the number of classes.")
        tick_labels = class_names

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if title is None:
        title = "Confusion Matrix (nearest-centroid, normalized)" if normalize else "Confusion Matrix (nearest-centroid)"
    plt.title(title)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    return cm

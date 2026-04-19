from pathlib import Path
import gc

import numpy as np
import matplotlib.pyplot as plt

from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
from tools.metrics import calculate_fisher_ratio


# ── Sweep values ──────────────────────────────────────────────────────────────
G_VALUES   = [3, 4, 5, 6, 7]
ETA_VALUES = [0.6, 0.9, 1.2, 1.5]

# ── Fixed parameters ──────────────────────────────────────────────────────────
N_NEURONS       = 1000
N_EPOCHS        = 50
EXAMPLES_TRAIN  = 500
EXAMPLES_TEST   = 100

TIME      = 100
DT        = 1.0
INTENSITY = 600
SEED      = 42

MNIST_INPUT   = True
HETEROGENEITY = False
SELF_TUNING   = False
SPATIAL       = False
CONVOLUTION   = False
LOG_NORMAL    = False
STDP          = False

SIGMA_INPUT   = 1
SIGMA_NETWORK = 1
EPSILON       = 0.3


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(g, eta, train_dataset, test_dataset):
    framework = Framework(
        n_neurons=N_NEURONS,
        time=TIME,
        dt=DT,
        seed=SEED,
        log_normal=LOG_NORMAL,
        heterogeneity=HETEROGENEITY,
        mnist_input=MNIST_INPUT,
        self_tuning=SELF_TUNING,
        spatial=SPATIAL,
        convolution=CONVOLUTION,
        g=g,
        eta=eta,
        sigma_input=SIGMA_INPUT,
        sigma_network=SIGMA_NETWORK,
        epsilon=EPSILON,
        intensity=INTENSITY,
        stdp=STDP,
    )
    framework.build_network()

    pairs_train, *_ = framework.run_stimulation(train_dataset, EXAMPLES_TRAIN)
    pairs_test,  *_ = framework.run_stimulation(test_dataset,  EXAMPLES_TEST)

    feature_dim = pairs_train[0][0].numel()
    readout = Readout(input_size=feature_dim, num_classes=10, seed=SEED)
    readout.train_readout(pairs_train, n_epochs=N_EPOCHS)
    accuracy = readout.test_readout(pairs_test)
    fisher_J = calculate_fisher_ratio(pairs_test)

    del readout, pairs_train, pairs_test, framework
    gc.collect()

    return accuracy, fisher_J


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_heatmaps(acc_grid, fisher_grid, plot_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, cmap in [
        (ax1, acc_grid,    "Accuracy (%)",    "viridis"),
        (ax2, fisher_grid, "Fisher Ratio (J)", "plasma"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("eta")
        ax.set_ylabel("g")
        ax.set_xticks(range(len(ETA_VALUES)))
        ax.set_xticklabels(ETA_VALUES)
        ax.set_yticks(range(len(G_VALUES)))
        ax.set_yticklabels(G_VALUES)

        for i in range(len(G_VALUES)):
            for j in range(len(ETA_VALUES)):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                        color="white", fontsize=7)

    plt.suptitle("g × eta sweep", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to: {plot_path}")


def write_results(results_path, acc_grid, fisher_grid):
    with results_path.open("w", encoding="utf-8") as f:
        f.write("g,eta,accuracy_percent,fisher_ratio\n")
        for i, g in enumerate(G_VALUES):
            for j, eta in enumerate(ETA_VALUES):
                f.write(f"{g},{eta},{acc_grid[i, j]:.2f},{fisher_grid[i, j]:.4f}\n")
    print(f"Saved results to: {results_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    repo_root   = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_txt_path  = results_dir / "g_eta_sweep.csv"
    results_plot_path = results_dir / "g_eta_sweep.png"

    data_cnn = Data_CNN(
        dt=DT,
        intensity=INTENSITY,
        kernel_size=9,
        thetas_deg=(0, 45, 90, 135),
        convolution=CONVOLUTION,
    )
    train_dataset, test_dataset = data_cnn.load_MNIST()

    n_g, n_eta = len(G_VALUES), len(ETA_VALUES)
    acc_grid    = np.full((n_g, n_eta), np.nan)
    fisher_grid = np.full((n_g, n_eta), np.nan)
    total = n_g * n_eta

    for i, g in enumerate(G_VALUES):
        for j, eta in enumerate(ETA_VALUES):
            step = i * n_eta + j + 1
            print(f"[{step}/{total}] g={g}, eta={eta}")
            acc, fisher_J = evaluate(g, eta, train_dataset, test_dataset)
            acc_grid[i, j]    = acc
            fisher_grid[i, j] = fisher_J
            print(f"         accuracy={acc:.2f}%  fisher={fisher_J:.4f}")

            # Save incrementally so partial results are never lost
            #write_results(results_txt_path, acc_grid, fisher_grid)
            plot_heatmaps(acc_grid, fisher_grid, results_plot_path)


if __name__ == "__main__":
    main()

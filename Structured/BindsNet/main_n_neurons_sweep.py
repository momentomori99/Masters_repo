from pathlib import Path
import gc

import matplotlib.pyplot as plt

from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
from tools.metrics import calculate_fisher_ratio


# Sweep setup
NEURON_COUNTS = list(range(50, 3001, 100))

# Experiment parameters
N_EPOCHS = 100
EXAMPLES_TRAIN = 500
EXAMPLES_TEST = 100

TIME = 100
DT = 1.0

INTENSITY = 600
SEED = 42

MNIST_INPUT = True
HETEROGENEITY = False
SELF_TUNING = False
SPATIAL = False
CONVOLUTION = False
LOG_NORMAL = False

G = 5
ETA = 0.6
SIGMA_INPUT = 1
SIGMA_NETWORK = 1
EPSILON = 0.3


def evaluate_with_neuron_count(n_neurons, train_dataset, test_dataset):
    framework = Framework(
        n_neurons=n_neurons,
        time=TIME,
        dt=DT,
        seed=SEED,
        log_normal=LOG_NORMAL,
        heterogeneity=HETEROGENEITY,
        mnist_input=MNIST_INPUT,
        self_tuning=SELF_TUNING,
        spatial=SPATIAL,
        convolution=CONVOLUTION,
        g=G,
        eta=ETA,
        sigma_input=SIGMA_INPUT,
        sigma_network=SIGMA_NETWORK,
        epsilon=EPSILON,
        intensity=INTENSITY,
    )
    framework.build_network()

    pairs_train, *_ = framework.run_stimulation(train_dataset, EXAMPLES_TRAIN)
    pairs_test, *_ = framework.run_stimulation(test_dataset, EXAMPLES_TEST)

    feature_dim = pairs_train[0][0].numel()
    readout = Readout(input_size=feature_dim, num_classes=10, seed=SEED)
    readout.train_readout(pairs_train, n_epochs=N_EPOCHS)
    accuracy = readout.test_readout(pairs_test)
    fisher_J = calculate_fisher_ratio(pairs_test)

    # Keep memory usage lower across repeated large simulations.
    del readout, pairs_train, pairs_test, framework
    gc.collect()

    return accuracy, fisher_J


def write_results(results_path, neuron_counts, accuracies, fisher_ratios):
    with results_path.open("w", encoding="utf-8") as f:
        f.write("n_neurons,accuracy_percent,fisher_ratio\n")
        for n_neurons, acc, J in zip(neuron_counts, accuracies, fisher_ratios):
            f.write(f"{n_neurons},{acc:.2f},{J:.4f}\n")


def plot_results(plot_path, neuron_counts, accuracies, fisher_ratios):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(neuron_counts, accuracies, marker="o")
    ax1.set_title("Accuracy vs. Number of Neurons")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True)

    ax2.plot(neuron_counts, fisher_ratios, marker="o", color="orange")
    ax2.set_title("Fisher Ratio vs. Number of Neurons")
    ax2.set_xlabel("Number of Neurons")
    ax2.set_ylabel("Fisher Ratio (J)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_txt_path = results_dir / "n_neurons_accuracies_run2.txt"
    results_plot_path = results_dir / "N_neurons_accuracy_run2.png"

    data_cnn = Data_CNN(
        dt=DT,
        intensity=INTENSITY,
        kernel_size=9,
        thetas_deg=(0, 45, 90, 135),
        convolution=CONVOLUTION,
    )
    train_dataset, test_dataset = data_cnn.load_MNIST()

    accuracies = []
    fisher_ratios = []
    total = len(NEURON_COUNTS)

    for idx, n_neurons in enumerate(NEURON_COUNTS, start=1):
        print(f"[{idx}/{total}] Running simulation with n_neurons={n_neurons}")
        acc, fisher_J = evaluate_with_neuron_count(n_neurons, train_dataset, test_dataset)
        accuracies.append(acc)
        fisher_ratios.append(fisher_J)
        print(f"[{idx}/{total}] Accuracy: {acc:.2f}% | Fisher ratio: {fisher_J:.4f}")

        completed_counts = NEURON_COUNTS[:idx]
        write_results(results_txt_path, completed_counts, accuracies, fisher_ratios)
        plot_results(results_plot_path, completed_counts, accuracies, fisher_ratios)

    print(f"Saved accuracy table to: {results_txt_path}")
    print(f"Saved plot to: {results_plot_path}")


if __name__ == "__main__":
    main()

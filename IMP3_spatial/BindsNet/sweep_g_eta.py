from datetime import datetime
from itertools import product
from pathlib import Path
import argparse

import torch

from brunel import Brunel
from input_data_CNN import Data as Data_CNN
from readout import Readout


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep combinations of g and eta, then log test accuracies."
    )
    parser.add_argument(
        "--g-values",
        nargs="+",
        type=float,
        default=[4.0, 5.0, 6.0, 7.0],
        help="List of g values to evaluate.",
    )
    parser.add_argument(
        "--eta-values",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 0.9, 1.1, 1.5],
        help="List of eta values to evaluate.",
    )
    parser.add_argument("--n-neurons", type=int, default=2500)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--examples-train", type=int, default=100)
    parser.add_argument("--examples-test", type=int, default=100)
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--intensity", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-file",
        type=str,
        default="g_eta_accuracy_results.txt",
        help="Output .txt filename (saved in this script directory by default).",
    )
    return parser.parse_args()


def run_one_experiment(g_value, eta_value, train_dataset, test_dataset, args):
    brunel = Brunel(
        n_neurons=args.n_neurons,
        time=args.time,
        dt=args.dt,
        heterogeneity=True,
        mnist_input=True,
        self_tuning=False,
        eta=eta_value,
        g=g_value,
        intensity=args.intensity,
    )
    brunel.build_brunel()

    training_pairs, *_ = brunel.stimulate_brunel(
        train_dataset, examples=args.examples_train, shuffle=True
    )
    test_pairs, *_ = brunel.stimulate_brunel(
        test_dataset, examples=args.examples_test, shuffle=False
    )

    feature_dim = training_pairs[0][0].numel()
    readout = Readout(input_size=feature_dim, num_classes=10, seed=args.seed)
    readout.train_readout(training_pairs, n_epochs=args.n_epochs)
    accuracy = readout.test_readout(test_pairs)
    return accuracy


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    data_cnn = Data_CNN(
        dt=args.dt, intensity=args.intensity, kernel_size=9, thetas_deg=(0, 45, 90, 135)
    )
    train_dataset, test_dataset = data_cnn.load_MNIST()

    with output_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 72 + "\n")
        f.write(f"Run started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"g values: {args.g_values}\n")
        f.write(f"eta values: {args.eta_values}\n")
        f.write("Format: g=<value>, eta=<value>, accuracy=<value>%\n")

    total_runs = len(args.g_values) * len(args.eta_values)
    run_idx = 0
    for g_value, eta_value in product(args.g_values, args.eta_values):
        run_idx += 1
        print(f"[{run_idx}/{total_runs}] Running g={g_value}, eta={eta_value} ...")
        try:
            accuracy = run_one_experiment(
                g_value=g_value,
                eta_value=eta_value,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                args=args,
            )
            line = f"g={g_value:.6g}, eta={eta_value:.6g}, accuracy={accuracy:.2f}%"
            print(line)
        except Exception as exc:
            line = f"g={g_value:.6g}, eta={eta_value:.6g}, ERROR={exc}"
            print(line)

        with output_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        # Free references before next run.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished. Results saved to: {output_path}")


if __name__ == "__main__":
    main()

from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
import torch
from torch.utils.data import Subset


# parameters
n_neurons = 2500
n_epochs = 100
examples_train = 500
examples_test = 100
n_splits = 10

time = 100
dt = 1.0

intensity = 600
seed = 42

mnist_input = True
heterogeneity = False
self_tuning = False
spatial = True
convolution = False
log_normal = False

g = 5
eta = 0.6
sigma_input = 1
sigma_network = 1
epsilon = 0.3


def build_distinct_splits(dataset_size, samples_per_split, n_splits, seed):
    required = samples_per_split * n_splits
    if required > dataset_size:
        raise ValueError(
            f"Need {required} samples but dataset has only {dataset_size}. "
            "Reduce n_splits or samples_per_split."
        )
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=generator)
    return [
        perm[i * samples_per_split : (i + 1) * samples_per_split].tolist()
        for i in range(n_splits)
    ]


data_CNN = Data_CNN(
    dt=dt,
    intensity=intensity,
    kernel_size=9,
    thetas_deg=(0, 45, 90, 135),
    convolution=convolution
)
train_dataset, test_dataset = data_CNN.load_MNIST()

framework = Framework(
    n_neurons=n_neurons,
    time=time,
    dt=dt,
    seed=seed,
    log_normal=log_normal,
    heterogeneity=heterogeneity,
    mnist_input=mnist_input,
    self_tuning=self_tuning,
    spatial=spatial,
    convolution=convolution,
    g=g,
    eta=eta,
    sigma_input=sigma_input,
    sigma_network=sigma_network,
    epsilon=epsilon,
    intensity=intensity,
)
framework.build_network()

train_split_indices = build_distinct_splits(
    dataset_size=len(train_dataset),
    samples_per_split=examples_train,
    n_splits=n_splits,
    seed=seed,
)
test_split_indices = build_distinct_splits(
    dataset_size=len(test_dataset),
    samples_per_split=examples_test,
    n_splits=n_splits,
    seed=seed + 1,
)

split_accuracies = []

for split_id in range(n_splits):
    train_subset = Subset(train_dataset, train_split_indices[split_id])
    test_subset = Subset(test_dataset, test_split_indices[split_id])

    pairs_train = framework.run_stimulation(
        train_subset, examples_train, shuffle=False
    )
    pairs_test = framework.run_stimulation(
        test_subset, examples_test, shuffle=False
    )

    feature_dim = pairs_train[0][0].numel()
    readout = Readout(input_size=feature_dim, num_classes=10, seed=seed + split_id)
    readout.train_readout(pairs_train, n_epochs=n_epochs)
    acc = readout.test_readout(pairs_test)
    split_accuracies.append(acc)
    print(f"Split {split_id + 1}/{n_splits} accuracy: {acc:.2f}%")

mean_acc = sum(split_accuracies) / len(split_accuracies)
std_acc = torch.tensor(split_accuracies).std(unbiased=False).item()
print(f"Average accuracy over {n_splits} splits: {mean_acc:.2f}% +/- {std_acc:.2f}%")

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST


# Parameters (match your reservoir experiment style for fair comparison)
n_epochs = 100
examples_train = 500
examples_test = 100
n_splits = 100
batch_size = 16
learning_rate = 1e-3
intensity = 600
seed = 42

data_root = "../../data"
device = "cpu"


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def build_distinct_splits(dataset_size, samples_per_split, n_splits, split_seed):
    required = samples_per_split * n_splits
    if required > dataset_size:
        raise ValueError(
            f"Need {required} samples but dataset has only {dataset_size}. "
            "Reduce n_splits or samples_per_split."
        )

    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(dataset_size, generator=generator)
    return [
        perm[i * samples_per_split : (i + 1) * samples_per_split].tolist()
        for i in range(n_splits)
    ]


class MNISTLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_split(model, train_loader, n_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(n_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def evaluate_one_split(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total if total else 0.0


set_seed(seed)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * intensity),
    ]
)

train_dataset = MNIST(
    root=os.path.join(data_root, "MNIST"),
    train=True,
    download=True,
    transform=transform,
)
test_dataset = MNIST(
    root=os.path.join(data_root, "MNIST"),
    train=False,
    download=True,
    transform=transform,
)

train_split_indices = build_distinct_splits(
    dataset_size=len(train_dataset),
    samples_per_split=examples_train,
    n_splits=n_splits,
    split_seed=seed,
)
test_split_indices = build_distinct_splits(
    dataset_size=len(test_dataset),
    samples_per_split=examples_test,
    n_splits=n_splits,
    split_seed=seed + 1,
)

split_accuracies = []

for split_id in range(n_splits):
    train_subset = Subset(train_dataset, train_split_indices[split_id])
    test_subset = Subset(test_dataset, test_split_indices[split_id])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    model = MNISTLinear().to(device)
    train_one_split(model, train_loader, n_epochs=n_epochs, lr=learning_rate)
    acc = evaluate_one_split(model, test_loader)
    split_accuracies.append(acc)
    print(f"Baseline split {split_id + 1}/{n_splits} accuracy: {acc:.2f}%")

mean_acc = sum(split_accuracies) / len(split_accuracies)
std_acc = torch.tensor(split_accuracies).std(unbiased=False).item()
print(f"Baseline average over {n_splits} splits: {mean_acc:.2f}% +/- {std_acc:.2f}%")

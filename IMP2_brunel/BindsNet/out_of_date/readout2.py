import torch
import torch.nn as nn
from tqdm import tqdm


class NN(nn.Module):
    """
    Original-style readout (from your first script):
    - Flattens x
    - Applies a Linear layer
    - Applies sigmoid
    - Returns a 1D vector of length num_classes
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # Matches your original: sigmoid(linear(flatten(x)))
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        return out  # shape: (num_classes,)


class Readout2:
    """
    Wraps the original training/testing logic into a class, but keeps it "as-is":
    - MSELoss(reduction="sum")
    - SGD(lr=1e-4, momentum=0.9)
    - One-hot target vector [1,1,C]
    - torch.max over outputs.data.unsqueeze(0)
    """
    def __init__(self, input_size: int, num_classes: int, device="cpu"):
        self.device = torch.device(device)
        self.model = NN(input_size, num_classes).to(self.device)
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.num_classes = num_classes

    def train_readout(self, training_pairs, n_epochs: int):
        print("\n Training the read out")
        self.model.train()

        pbar = tqdm(enumerate(range(n_epochs)), total=n_epochs)
        for epoch, _ in pbar:
            avg_loss = 0.0

            for s, l in training_pairs:
                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(s.to(self.device))  # (C,)

                # Original one-hot label shape: (1, 1, C)
                label = torch.zeros(1, 1, self.num_classes, device=self.device).float()
                label[0, 0, int(l)] = 1.0

                # Original loss call:
                loss = self.criterion(outputs.view(1, 1, -1), label)
                avg_loss += loss.data.item()

                loss.backward()
                self.optimizer.step()

            pbar.set_description_str(
                "Epoch: %d/%d, Loss: %.4f"
                % (epoch + 1, n_epochs, avg_loss / len(training_pairs))
            )

    def test_readout(self, test_pairs):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for s, label in test_pairs:
                outputs = self.model(s.to(self.device))  # (C,)
                _, predicted = torch.max(outputs.data.unsqueeze(0), 1)  # (1,)
                total += 1
                correct += int(predicted.item() == int(label))

        return 100.0 * correct / total if total else 0.0

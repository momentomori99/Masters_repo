import torch
import torch.nn as nn
from tqdm import tqdm

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.linear_1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        return out


class Readout:
    def __init__(self, input_size, num_classes):
        self.num_classes = num_classes
        

        self.model = NN(input_size, num_classes).to("cpu")
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)

    def train_readout(self, training_pairs, n_epochs):
        self.model.train()
        pbar = tqdm(range(n_epochs), desc="Training readout")
        for epoch in pbar:
            avg_loss = 0.0

            for s, l in training_pairs:
                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(s)

                # One-hot label target
                target = torch.zeros(1, 1, self.num_classes).float()
                target[0, 0, int(l)] = 1.0

                # Compute loss (match your original logic)
                loss = self.criterion(outputs.view(1, 1, -1), target)
                avg_loss += loss.item()

                # Backprop + step
                loss.backward()
                self.optimizer.step()

            pbar.set_description_str(
                f"Epoch: {epoch+1}/{n_epochs}, Loss: {avg_loss / len(training_pairs):.4f}"
            )
    def test_readout(self, test_pairs):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for s, label in test_pairs:
                outputs = self.model(s)

                # same logic as original code
                _, predicted = torch.max(outputs.data.unsqueeze(0), 1)

                total += 1
                correct += int(predicted == int(label))

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy
# readout_unsupervised.py
import random
import numpy as np
import torch


class Readout:
    """
    Nearest-centroid per class (prototype classifier).

    - train_readout: computes one centroid vector per class from labeled training pairs.
    - test_readout: predicts by nearest centroid (Euclidean distance by default).
    """

    def __init__(self, input_size, num_classes, seed, metric="euclidean"):
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.metric = metric  # "euclidean" or "cosine"

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.centroids = None  # shape: (C, D)

    def _to_vector(self, s):
        """
        Make sure s becomes a 1D float tensor of length D.
        Accepts torch.Tensor, numpy array, or Python list.
        """
        if not torch.is_tensor(s):
            s = torch.tensor(s)
        s = s.float().view(-1)

        # Optional: sanity check (won't crash training, but helps debugging)
        # if s.numel() != self.input_size:
        #     raise ValueError(f"Expected input_size={self.input_size}, got {s.numel()}")

        return s

    def train_readout(self, training_pairs, n_epochs):
        """
        Computes class centroids from training_pairs = [(s, label), ...].

        n_epochs is accepted for interface-compatibility but not used.
        """
        sums = torch.zeros(self.num_classes, self.input_size, dtype=torch.float32)
        counts = torch.zeros(self.num_classes, dtype=torch.long)

        for s, label in training_pairs:
            x = self._to_vector(s)
            y = int(label)

            if 0 <= y < self.num_classes:
                sums[y] += x
                counts[y] += 1

        # Build centroids
        centroids = torch.zeros(self.num_classes, self.input_size, dtype=torch.float32)
        for c in range(self.num_classes):
            if counts[c].item() > 0:
                centroids[c] = sums[c] / float(counts[c].item())
            else:
                # If a class is missing in training, keep zero centroid.
                centroids[c] = torch.zeros(self.input_size, dtype=torch.float32)

        # If using cosine, normalize centroids once (and normalize samples at test time)
        if self.metric == "cosine":
            centroids = self._normalize_rows(centroids)

        self.centroids = centroids

    def _normalize_rows(self, M, eps=1e-8):
        norms = torch.norm(M, dim=1, keepdim=True)
        return M / (norms + eps)

    def _predict_one(self, s):
        """
        Returns predicted class index for a single sample vector s.
        """
        if self.centroids is None:
            raise RuntimeError("Readout not trained: call train_readout(...) first.")

        x = self._to_vector(s)

        if self.metric == "cosine":
            x = x / (torch.norm(x) + 1e-8)
            # cosine similarity = dot(x, centroid) since both normalized
            sims = torch.mv(self.centroids, x)  # shape: (C,)
            pred = int(torch.argmax(sims).item())
            return pred

        # Euclidean distance
        diffs = self.centroids - x.unsqueeze(0)          # (C, D)
        dists = torch.norm(diffs, dim=1)                 # (C,)
        pred = int(torch.argmin(dists).item())
        return pred

    def test_readout(self, test_pairs):
        """
        Returns accuracy in percent.
        """
        if self.centroids is None:
            raise RuntimeError("Readout not trained: call train_readout(...) first.")

        correct = 0
        total = 0

        for s, label in test_pairs:
            y = int(label)
            pred = self._predict_one(s)
            total += 1
            correct += int(pred == y)

        return 100.0 * correct / total if total else 0.0

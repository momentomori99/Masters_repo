import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =====================================================
#  Surrogate-gradient spike function
# =====================================================
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_th, gamma=0.5, sigma=0.3):
        """
        Forward: hard step: z = 1 if v >= v_th else 0
        Backward: Gaussian surrogate derivative around threshold.
        """
        ctx.save_for_backward(v, v_th)
        ctx.gamma = gamma
        ctx.sigma = sigma
        out = (v >= v_th).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v, v_th = ctx.saved_tensors
        gamma = ctx.gamma
        sigma = ctx.sigma

        # Gaussian bump around threshold
        psi = gamma * torch.exp(-((v - v_th) ** 2) / (sigma ** 2))

        grad_v = grad_output * psi
        # v_th, gamma, sigma are treated as constants (no grad)
        return grad_v, None, None, None


spike_fn = SurrogateSpike.apply


# =====================================================
#  Recurrent LIF hidden layer (Brunel-like core)
# =====================================================
class LIFRecurrentLayer(nn.Module):
    def __init__(self, n_neurons, input_dim, alpha=0.9,
                 v_th=1.0, v_reset=0.0):
        super().__init__()
        self.n = n_neurons
        self.input_dim = input_dim
        self.alpha = alpha
        self.v_th = nn.Parameter(torch.tensor(v_th), requires_grad=False)
        self.v_reset = v_reset

        # Input weights and recurrent weights (trainable)
        self.W_in = nn.Parameter(0.1 * torch.randn(n_neurons, input_dim))
        self.W_rec = nn.Parameter(0.1 * torch.randn(n_neurons, n_neurons))

    def forward(self, x_seq):
        """
        x_seq: (T, B, input_dim)
        Returns:
            v_seq: (T+1, B, n_neurons)
            z_seq: (T,   B, n_neurons)
        """
        T, B, D = x_seq.size()
        device = x_seq.device

        v = torch.zeros(B, self.n, device=device)
        v_seq = [v]
        z_seq = []

        for t in range(T):
            x_t = x_seq[t]           # (B, D)
            I_in = x_t @ self.W_in.T # (B, n)
            I_rec = z_seq[-1] @ self.W_rec.T if z_seq else torch.zeros_like(v)

            # LIF update without noise:
            v = self.alpha * v + I_in + I_rec

            # Spike
            z = spike_fn(v, self.v_th)

            # Subtractive reset
            v = v - z * (self.v_th - self.v_reset)

            v_seq.append(v)
            z_seq.append(z)

        v_seq = torch.stack(v_seq, dim=0)  # (T+1, B, n)
        z_seq = torch.stack(z_seq, dim=0)  # (T,   B, n)
        return v_seq, z_seq


# =====================================================
#  LIF readout layer (3 spiking neurons = 3 classes)
# =====================================================
class LIFReadout(nn.Module):
    def __init__(self, n_out, n_hidden, alpha=0.9,
                 v_th=1.0, v_reset=0.0):
        super().__init__()
        self.n_out = n_out
        self.alpha = alpha
        self.v_th = nn.Parameter(torch.tensor(v_th), requires_grad=False)
        self.v_reset = v_reset

        self.W_out = nn.Parameter(0.1 * torch.randn(n_out, n_hidden))

    def forward(self, z_hidden_seq):
        """
        z_hidden_seq: (T, B, n_hidden)
        Returns:
            v_seq: (T+1, B, n_out)
            z_seq: (T,   B, n_out)
        """
        T, B, N = z_hidden_seq.size()
        device = z_hidden_seq.device

        v = torch.zeros(B, self.n_out, device=device)
        v_seq = [v]
        z_seq = []

        for t in range(T):
            z_h = z_hidden_seq[t]      # (B, N)
            I_in = z_h @ self.W_out.T  # (B, n_out)

            v = self.alpha * v + I_in
            z = spike_fn(v, self.v_th)
            v = v - z * (self.v_th - self.v_reset)

            v_seq.append(v)
            z_seq.append(z)

        v_seq = torch.stack(v_seq, dim=0)  # (T+1, B, n_out)
        z_seq = torch.stack(z_seq, dim=0)  # (T,   B, n_out)
        return v_seq, z_seq


# =====================================================
#  Full SNN classifier for Iris
# =====================================================
class SNNIrisClassifier(nn.Module):
    def __init__(self, n_hidden=64, T=20, input_dim=4, n_classes=3):
        super().__init__()
        self.T = T
        self.hidden = LIFRecurrentLayer(n_hidden, input_dim)
        self.readout = LIFReadout(n_classes, n_hidden)

    def forward(self, x_batch):
        """
        x_batch: (B, input_dim) static features.
        We repeat them over T time steps as constant input.
        Returns:
            logits: (B, n_classes)
            z_hidden_seq: (T, B, n_hidden)
            z_readout_seq: (T, B, n_classes)
        """
        B, D = x_batch.size()
        device = x_batch.device

        # Repeat static input over time
        x_seq = x_batch.unsqueeze(0).repeat(self.T, 1, 1)  # (T, B, D)

        v_h_seq, z_h_seq = self.hidden(x_seq)
        v_r_seq, z_r_seq = self.readout(z_h_seq)

        # Use average spike count over time as "logits"
        rates = z_r_seq.mean(dim=0)  # (B, n_classes)

        return rates, z_h_seq, z_r_seq


# =====================================================
#  Data prep: load Iris, standardize, train/test split
# =====================================================
def load_iris_torch(test_size=0.3, batch_size=32, seed=0):
    iris = load_iris()
    X = iris.data       # (150, 4)
    y = iris.target     # (150,)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =====================================================
#  Training & evaluation
# =====================================================
def train_and_eval(
    n_epochs=100,
    n_hidden=64,
    T=20,
    lr=1e-3,
    batch_size=32
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, test_loader = load_iris_torch(batch_size=batch_size)

    model = SNNIrisClassifier(n_hidden=n_hidden, T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits, z_h, z_r = model(xb)

            # Cross-entropy on readout firing rates
            loss = criterion(logits, yb)

            # Optional regularization to keep firing rates reasonable
            rate_reg = z_h.mean() + z_r.mean()
            loss = loss + 1e-3 * rate_reg

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_examples += xb.size(0)

        train_loss = total_loss / total_examples
        train_acc = total_correct / total_examples

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss = {train_loss:.4f} | "
                  f"train acc = {train_acc:.3f}")

    # ---- Evaluation ----
    model.eval()
    def eval_loader(loader):
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits, _, _ = model(xb)
                preds = logits.argmax(dim=1)
                total_correct += (preds == yb).sum().item()
                total_examples += xb.size(0)
        return total_correct / total_examples

    train_acc = eval_loader(train_loader)
    test_acc = eval_loader(test_loader)

    print("\nFinal performance:")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")

    return model


if __name__ == "__main__":
    train_and_eval(
        n_epochs=250,  # bump this up if you want
        n_hidden=64,
        T=20,
        lr=1e-3,
        batch_size=32
    )

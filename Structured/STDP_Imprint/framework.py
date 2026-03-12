import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'BindsNet'))

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre

from tools.feature_encoding import encode_feature_map
from tools.feature_encoding import spikes_to_binned_counts

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)


class STDPImprintFramework:
    def __init__(self, neurons_per_class, num_classes, time, dt, seed,
                 nu_stdp=(1e-4, 1e-2), norm_stdp=None, w_input=10.0,
                 intensity=600, g=5, eta=0.6, epsilon=0.3):
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.device = "cpu"

        self.time = int(time)
        self.dt = float(dt)
        self.bin_ms = 50
        self.intensity = intensity

        self.num_classes = num_classes
        self.neurons_per_class = neurons_per_class
        self.N_E = neurons_per_class * num_classes
        self.N_I = max(self.N_E // 4, 1)
        self.n_neurons = self.N_E + self.N_I

        self.class_assignments = torch.arange(self.N_E) // neurons_per_class

        self.g = g
        self.eta = eta
        self.epsilon = epsilon
        self.w_E = 1
        self.w_ext = 1
        self.w_input = w_input

        self.mean_w_EE = self.w_E
        self.mean_w_EI = self.w_E
        self.mean_w_IE = -self.g * self.w_E
        self.mean_w_II = -self.g * self.w_E
        self.std_w = 0.05

        self.theta = 20.0
        self.tau_m = 20.0
        self.tau_s = self.tau_m / 1000.0

        self.v_th = self.theta / (self.tau_s * self.w_ext)
        self.rate_ext = self.eta * self.v_th

        self.nu_stdp = nu_stdp
        self.norm_stdp = norm_stdp

    def build_network(self):
        print("Building network...")

        self.network = Network(dt=self.dt)

        self.neurons_E = LIFNodes(
            n=self.N_E, tau=self.tau_m, rest=0.0, reset=0.0,
            thresh=self.theta, refrac=1, traces=True, tc_trace=20.0,
        )
        self.neurons_I = LIFNodes(
            n=self.N_I, tau=self.tau_m, rest=0.0, reset=0.0,
            thresh=self.theta, refrac=1, traces=True, tc_trace=20.0,
        )
        self.network.add_layer(self.neurons_E, name="E")
        self.network.add_layer(self.neurons_I, name="I")

        self.noise_E = Input(n=self.N_E)
        self.noise_I = Input(n=self.N_I)
        self.network.add_layer(self.noise_E, name="noise_E")
        self.network.add_layer(self.noise_I, name="noise_I")

        # Dense random F→E with STDP
        self.mnist_in = Input(n=784, traces=True, tc_trace=20.0)
        self.network.add_layer(self.mnist_in, name="F")

        self.W_in = torch.rand(784, self.N_E) * 0.01
        norm_val = self.norm_stdp if self.norm_stdp is not None else self.w_input
        self.norm_stdp_val = norm_val

        self.connection_F_E = Connection(
            source=self.mnist_in, target=self.neurons_E, w=self.W_in,
            update_rule=PostPre, nu=self.nu_stdp,
            wmin=0.0, wmax=self.w_input * 2, norm=norm_val,
        )
        self.network.add_connection(self.connection_F_E, source="F", target="E")
        print(f"STDP norm (F→E): {norm_val:.2f}")

        # E/I recurrent connections
        mask_EE = torch.bernoulli(torch.full((self.N_E, self.N_E), self.epsilon))
        mask_EI = torch.bernoulli(torch.full((self.N_E, self.N_I), self.epsilon))
        mask_IE = torch.bernoulli(torch.full((self.N_I, self.N_E), self.epsilon))
        mask_II = torch.bernoulli(torch.full((self.N_I, self.N_I), self.epsilon))

        W_EE = mask_EE * torch.normal(self.mean_w_EE, self.std_w, size=(self.N_E, self.N_E))
        W_EI = mask_EI * torch.normal(self.mean_w_EI, self.std_w, size=(self.N_E, self.N_I))
        W_IE = mask_IE * torch.normal(self.mean_w_IE, self.std_w, size=(self.N_I, self.N_E))
        W_II = mask_II * torch.normal(self.mean_w_II, self.std_w, size=(self.N_I, self.N_I))

        self.network.add_connection(
            Connection(source=self.neurons_E, target=self.neurons_E, w=W_EE),
            source="E", target="E")
        self.network.add_connection(
            Connection(source=self.neurons_E, target=self.neurons_I, w=W_EI),
            source="E", target="I")
        self.network.add_connection(
            Connection(source=self.neurons_I, target=self.neurons_E, w=W_IE),
            source="I", target="E")
        self.network.add_connection(
            Connection(source=self.neurons_I, target=self.neurons_I, w=W_II),
            source="I", target="I")
        self.network.add_connection(
            Connection(source=self.noise_E, target=self.neurons_E,
                       w=self.w_ext * torch.eye(self.N_E)),
            source="noise_E", target="E")
        self.network.add_connection(
            Connection(source=self.noise_I, target=self.neurons_I,
                       w=self.w_ext * torch.eye(self.N_I)),
            source="noise_I", target="I")

        T = int(self.time / self.dt)
        self.mon_E = Monitor(self.neurons_E, state_vars=["s"], time=T)
        self.mon_I = Monitor(self.neurons_I, state_vars=["s"], time=T)
        self.network.add_monitor(self.mon_E, name="E_spikes")
        self.network.add_monitor(self.mon_I, name="I_spikes")

        self.network.learning = False
        print(f"Network built: N_E={self.N_E}, N_I={self.N_I}, "
              f"{self.neurons_per_class} neurons/class")

    def enable_stdp(self):
        self.network.learning = True

    def disable_stdp(self):
        self.network.learning = False

    def run(self, feat_spikes):
        feat_spikes = feat_spikes.to(self.device)
        encoder = PoissonEncoder(time=1, dt=self.dt)

        for t in range(self.time):
            rates_XE = torch.ones(self.N_E) * self.rate_ext
            rates_XI = torch.ones(self.N_I) * self.rate_ext
            spikes_XE = encoder(rates_XE)
            spikes_XI = encoder(rates_XI)
            feat_t = feat_spikes[t:t+1]
            self.network.run(
                inputs={
                    "noise_E": spikes_XE.unsqueeze(0),
                    "noise_I": spikes_XI.unsqueeze(0),
                    "F": feat_t,
                },
                time=1,
            )

        E_spikes = self.mon_E.get("s")
        I_spikes = self.mon_I.get("s")
        E_spike_counts = E_spikes.squeeze(1).sum(0)
        I_spike_counts = I_spikes.squeeze(1).sum(0)

        self.network.reset_state_variables()
        self.mon_E.reset_state_variables()
        self.mon_I.reset_state_variables()

        return E_spike_counts, I_spike_counts, E_spikes, I_spikes

    def run_stdp_training(self, dataset, n_samples_per_neuron):
        self.W_in_before = self.connection_F_E.w.detach().clone()

        class_indices = {}
        for i in range(len(dataset)):
            label = dataset[i]["label"]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        self.enable_stdp()

        total = self.N_E * n_samples_per_neuron
        pbar = tqdm(total=total, desc="STDP Imprint Training")

        for neuron_j in range(self.N_E):
            c = self.class_assignments[neuron_j].item()
            indices = class_indices.get(c, [])
            if not indices:
                pbar.update(n_samples_per_neuron)
                continue

            for _ in range(n_samples_per_neuron):
                idx = random.choice(indices)
                sample = dataset[idx]
                feature_map = sample["feature_map"]
                feat_spikes = encode_feature_map(
                    feature_map, self.time, self.dt, self.intensity
                )

                single_mask = torch.zeros(self.N_E, dtype=torch.bool)
                single_mask[neuron_j] = True
                w_saved = self.connection_F_E.w.data[:, ~single_mask].clone()
                self.run(feat_spikes)
                self.connection_F_E.w.data[:, ~single_mask] = w_saved

                pbar.update(1)

        pbar.close()
        self.disable_stdp()

        self.W_in_after = self.connection_F_E.w.detach().clone()
        delta = (self.W_in_after - self.W_in_before).abs().mean().item()
        print(f"STDP imprint training complete. Mean |ΔW|: {delta:.6f}")

    def run_stimulation(self, dataset, examples, shuffle=True):
        pbar = tqdm(range(examples),
                    desc=f"Stimulating network: (0 / {examples})")
        pairs = []

        for i, index in enumerate(pbar):
            if shuffle:
                index = random.choice(range(len(dataset)))
            sample = dataset[index]
            label = sample["label"]
            feature_map = sample["feature_map"]
            feat_spikes = encode_feature_map(
                feature_map, self.time, self.dt, self.intensity
            )

            E_spike_counts, I_spike_counts, E_spikes, I_spikes = self.run(feat_spikes)
            binned_E = spikes_to_binned_counts(
                E_spikes, bin_ms=self.bin_ms, dt=self.dt, time=self.time
            )
            features = binned_E.flatten().float()
            pairs.append((features, label))
            pbar.set_description_str(
                f"Stimulating network: ({i+1} / {examples})"
            )

        return pairs

    def plot_class_responses(self, dataset, classes_to_show=None):
        if classes_to_show is None:
            classes_to_show = list(range(self.num_classes))

        class_indices = {}
        for i in range(len(dataset)):
            label = dataset[i]["label"]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        n_samples = len(classes_to_show)
        npc = self.neurons_per_class
        nc = self.num_classes

        response_matrix = np.zeros((n_samples, self.N_E))
        images = []

        for row, c in enumerate(classes_to_show):
            idx = random.choice(class_indices[c])
            sample = dataset[idx]
            images.append(sample["feature_map"].squeeze().cpu().numpy())
            feat_spikes = encode_feature_map(
                sample["feature_map"], self.time, self.dt, self.intensity
            )
            E_spike_counts, _, _, _ = self.run(feat_spikes)
            response_matrix[row] = E_spike_counts.cpu().numpy()

        fig, axes = plt.subplots(
            n_samples, 2, figsize=(14, 2.2 * n_samples),
            gridspec_kw={'width_ratios': [1, 6]},
        )
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for row, c in enumerate(classes_to_show):
            axes[row, 0].imshow(images[row], cmap='gray')
            axes[row, 0].set_title(f'Input: {c}', fontsize=10)
            axes[row, 0].axis('off')

            counts = response_matrix[row]
            colors = []
            for j in range(self.N_E):
                assigned = self.class_assignments[j].item()
                if assigned == c:
                    colors.append('#e63946')
                else:
                    colors.append('#457b9d')

            axes[row, 1].bar(range(self.N_E), counts, color=colors, width=1.0)
            for boundary in range(1, nc):
                axes[row, 1].axvline(x=boundary * npc - 0.5,
                                     color='gray', linewidth=0.5, linestyle='--')

            axes[row, 1].set_xlim(-0.5, self.N_E - 0.5)
            axes[row, 1].set_ylabel('Spikes', fontsize=9)
            if row == n_samples - 1:
                tick_pos = [npc * c + npc // 2 for c in range(nc)]
                axes[row, 1].set_xticks(tick_pos)
                axes[row, 1].set_xticklabels(
                    [f'C{c}' for c in range(nc)], fontsize=8)
            else:
                axes[row, 1].set_xticks([])

        fig.suptitle("Neuron Responses by Input Class\n"
                     "(red = neurons assigned to the input class)",
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(_RESULTS_DIR, "class_responses.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()

    def plot_group_responses(self, dataset, classes_to_show=None):
        if classes_to_show is None:
            classes_to_show = list(range(self.num_classes))

        class_indices = {}
        for i in range(len(dataset)):
            label = dataset[i]["label"]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        n_samples = len(classes_to_show)
        npc = self.neurons_per_class
        nc = self.num_classes

        group_matrix = np.zeros((n_samples, nc))
        images = []

        for row, c in enumerate(classes_to_show):
            idx = random.choice(class_indices[c])
            sample = dataset[idx]
            images.append(sample["feature_map"].squeeze().cpu().numpy())
            feat_spikes = encode_feature_map(
                sample["feature_map"], self.time, self.dt, self.intensity
            )
            E_spike_counts, _, _, _ = self.run(feat_spikes)
            counts = E_spike_counts.cpu().numpy()
            for g in range(nc):
                group_matrix[row, g] = counts[g * npc:(g + 1) * npc].sum()

        fig, axes = plt.subplots(
            n_samples, 2, figsize=(10, 2.2 * n_samples),
            gridspec_kw={'width_ratios': [1, 4]},
        )
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for row, c in enumerate(classes_to_show):
            axes[row, 0].imshow(images[row], cmap='gray')
            axes[row, 0].set_title(f'Input: {c}', fontsize=10)
            axes[row, 0].axis('off')

            bars = group_matrix[row]
            winner = int(np.argmax(bars))
            colors = ['#e63946' if g == c else '#457b9d' for g in range(nc)]
            axes[row, 1].bar(range(nc), bars, color=colors, edgecolor='white',
                             linewidth=0.5)
            axes[row, 1].annotate(
                f'max: {winner}', xy=(winner, bars[winner]),
                xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            )
            axes[row, 1].set_xticks(range(nc))
            axes[row, 1].set_xticklabels([f'{g}' for g in range(nc)], fontsize=9)
            axes[row, 1].set_ylabel('Total spikes', fontsize=9)
            if row == 0:
                axes[row, 1].set_title('Neuron group', fontsize=10)

        fig.suptitle("Group Spike Totals by Input Class\n"
                     "(red = group assigned to the input class)",
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(_RESULTS_DIR, "group_responses.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()

    def plot_input_weights(self, title="Input Weights"):
        w = self.connection_F_E.w.detach().clone().cpu().numpy()
        npc = self.neurons_per_class
        nc = self.num_classes

        fig, axes = plt.subplots(
            nc, npc, figsize=(npc * 2, nc * 2),
            squeeze=False,
        )

        for c in range(nc):
            for k in range(npc):
                neuron_idx = c * npc + k
                rf = w[:, neuron_idx].reshape(28, 28)
                axes[c, k].imshow(rf, cmap='hot', interpolation='nearest')
                axes[c, k].axis('off')
                if k == 0:
                    axes[c, k].set_ylabel(f'Class {c}', fontsize=10,
                                          rotation=0, labelpad=40, va='center')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        safe_title = title.replace(' ', '_').lower()
        plt.savefig(os.path.join(_RESULTS_DIR, f"{safe_title}.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()

    def plot_weight_change(self, n_show=16):
        if not hasattr(self, 'W_in_before') or not hasattr(self, 'W_in_after'):
            print("No STDP training has been run yet.")
            return

        delta = (self.W_in_after - self.W_in_before).cpu().numpy()
        npc = self.neurons_per_class
        nc = self.num_classes

        fig, axes = plt.subplots(
            nc, npc, figsize=(npc * 2, nc * 2),
            squeeze=False,
        )

        vmax = np.abs(delta).max()
        for c in range(nc):
            for k in range(npc):
                neuron_idx = c * npc + k
                rf = delta[:, neuron_idx].reshape(28, 28)
                axes[c, k].imshow(rf, cmap='RdBu_r', interpolation='nearest',
                                  vmin=-vmax, vmax=vmax)
                axes[c, k].axis('off')
                if k == 0:
                    axes[c, k].set_ylabel(f'Class {c}', fontsize=10,
                                          rotation=0, labelpad=40, va='center')

        fig.suptitle("STDP Weight Change (ΔW)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(_RESULTS_DIR, "stdp_imprint_weight_change.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()

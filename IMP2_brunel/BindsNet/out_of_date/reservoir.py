from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor

import numpy as np
import torch
from tqdm import tqdm


class Reservoir:
    def __init__(self, n_neurons, time, dt):
        self.n_neurons = n_neurons
        self.time = time
        self.dt = dt
        self.device = "cpu"

    def build_reservoir(self):
        self.network = Network(dt=self.dt)
        inpt = Input(n=784, shape=(1, 28, 28))
        self.network.add_layer(inpt, name="I")
        output = LIFNodes(self.n_neurons, thresh=-52 + np.random.randn(self.n_neurons).astype(float))
        self.network.add_layer(output, name="O")

        C1 = Connection(source=inpt, target=output, w= 0.5 * torch.randn(inpt.n, output.n))
        C2 = Connection(source=output, target=output, w= 0.5 * torch.randn(output.n, output.n))

        self.network.add_connection(C1, source="I", target="O")
        self.network.add_connection(C2, source="O", target="O")

        # Monitors for visualizing activity
        self.spikes = {}
        for l in self.network.layers:
            self.spikes[l] = Monitor(self.network.layers[l], ["s"], time=self.time, device = self.device)
            self.network.add_monitor(self.spikes[l], name="%s_spikes" % l)

        self.voltage = {"O": Monitor(self.network.layers["O"], ["v"], time=self.time, device = self.device)}
        self.network.add_monitor(self.voltage["O"], name="O_voltage")






    def train_reservoir(self, dataset, examples, shuffle=True):
        n_total = len(dataset)
        n_iters = min(examples, n_total)
        print(f"Training the reservoir with {n_iters} examples")

        if shuffle:
            indices = torch.randperm(n_total)[:n_iters].tolist()
        else:
            indices = list(range(n_iters))

        training_pairs = []
        pbar = tqdm(indices, desc=f"Train progress: (0 / {n_iters})")


        for i, idx in enumerate(pbar):
            dataPoint = dataset[idx]

            datum = dataPoint["encoded_image"].view(int(self.time / self.dt), 1, 1, 28, 28).to(self.device)
            label = dataPoint["label"]

            pbar.set_description_str(f"Train progress: ({i+1} / {n_iters})")

            self.network.run(inputs={"I": datum}, time=self.time)

            # Store spikes from output layer "O"
            training_pairs.append((self.spikes["O"].get("s"), label))

            # Reset state variables between samples
            self.network.reset_state_variables()

        return training_pairs

    def train_reservoir_spike_counts(self, dataset, examples, shuffle=True):
        n_total = len(dataset)
        n_iters = min(examples, n_total)
        print(f"Training the reservoir with {n_iters} examples")

        if shuffle:
            indices = torch.randperm(n_total)[:n_iters].tolist()
        else:
            indices = list(range(n_iters))

        training_pairs = []
        pbar = tqdm(indices, desc=f"Train progress: (0 / {n_iters})")


        for i, idx in enumerate(pbar):
            dataPoint = dataset[idx]

            datum = dataPoint["encoded_image"].view(int(self.time / self.dt), 1, 1, 28, 28).to(self.device)
            label = dataPoint["label"]

            pbar.set_description_str(f"Train progress: ({i+1} / {n_iters})")

            self.network.run(inputs={"I": datum}, time=self.time)

            # Store spikes from output layer "O"
            s = self.spikes["O"].get("s")
            count = s.sum(0)
            training_pairs.append((count, label))

            # Reset state variables between samples
            self.network.reset_state_variables()

        return training_pairs

    def test_reservoir(self, dataset, examples, shuffle=False):
        n_total = len(dataset)
        n_iters = min(examples, n_total)

        # manual indices
        if shuffle:
            indices = torch.randperm(n_total)[:n_iters].tolist()
        else:
            indices = list(range(n_iters))

        test_pairs = []
        pbar = tqdm(indices, desc=f"Test progress: (0 / {n_iters})")

        for i, idx in enumerate(pbar):
            dataPoint = dataset[idx]

            datum = dataPoint["encoded_image"].view(int(self.time / self.dt), 1, 1, 28, 28).to(self.device)
            label = dataPoint["label"]

            pbar.set_description_str(f"Test progress: ({i+1} / {n_iters})")

            self.network.run(inputs={"I": datum}, time=self.time)

            # Store spikes from output layer "O"
            test_pairs.append((self.spikes["O"].get("s"), label))

            # Reset state variables between samples
            self.network.reset_state_variables()

        return test_pairs

    def test_reservoir_spike_counts(self, dataset, examples, shuffle=False):
        n_total = len(dataset)
        n_iters = min(examples, n_total)

        # manual indices
        if shuffle:
            indices = torch.randperm(n_total)[:n_iters].tolist()
        else:
            indices = list(range(n_iters))

        test_pairs = []
        pbar = tqdm(indices, desc=f"Test progress: (0 / {n_iters})")

        for i, idx in enumerate(pbar):
            dataPoint = dataset[idx]

            datum = dataPoint["encoded_image"].view(int(self.time / self.dt), 1, 1, 28, 28).to(self.device)
            label = dataPoint["label"]

            pbar.set_description_str(f"Test progress: ({i+1} / {n_iters})")

            self.network.run(inputs={"I": datum}, time=self.time)

            # Store spikes from output layer "O"
            s = self.spikes["O"].get("s")
            count = s.sum(0)
            test_pairs.append((count, label))

            # Reset state variables between samples
            self.network.reset_state_variables()

        return test_pairs
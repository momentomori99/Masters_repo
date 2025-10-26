from LIF_brunels_network import simulate_brunels_network
from brian2 import NeuronGroup, Synapses, Network, PoissonInput, StateMonitor, SpikeMonitor, PopulationRateMonitor

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

X = np.load('data/iris_X_rates.npy')
y = np.load('data/iris_y.npy')






i = 0
X_output = []
for observation in X:
    print(f"Observation {i}")
    _, _, spike_monitor_E, _, _, _ = simulate_brunels_network(observation)
    num_neurons_E = np.max(spike_monitor_E.i) + 1 if len(spike_monitor_E.i) > 0 else 0
    spike_counts = np.bincount(spike_monitor_E.i, minlength=num_neurons_E)
    X_output.append(spike_counts)
    print(f"Dimension of output matrix: {np.shape(X_output)}")

    i += 1


np.save('data/X_output2.npy', X_output)









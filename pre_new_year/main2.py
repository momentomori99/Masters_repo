from LIF_brunels_network import simulate_brunels_network
from brian2 import NeuronGroup, Synapses, Network, PoissonInput, StateMonitor, SpikeMonitor, PopulationRateMonitor

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import brian2 as b2
X = np.load('data/iris_X_rates.npy')
y = np.load('data/iris_y.npy')




i = 0
X_output = []
for observation in X:
    print("================================================")
    print(f"Observation: {i}")
    observation = observation * b2.Hz
    voltage_monitor_E, _, spike_monitor_E, _, _, _, _, _, _ = simulate_brunels_network(observation, g_strength=0.5, eta=0.90)
    n_E_neurons_recorded = voltage_monitor_E.v.shape[0]
    spike_counts = np.bincount(spike_monitor_E.i, minlength=n_E_neurons_recorded)
    X_output.append(spike_counts)
    

    i += 1

X_output = np.array(X_output)

np.save('data/X_output2.npy', X_output)









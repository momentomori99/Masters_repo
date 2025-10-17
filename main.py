from LIF_brunels_network import simulate_brunels_network
from brian2 import NeuronGroup, Synapses, Network, PoissonInput, StateMonitor, SpikeMonitor, PopulationRateMonitor

import matplotlib.pyplot as plt
import seaborn as sns

#oltage_monitor_E, voltage_monitor_I, spike_monitor_E, spike_monitor_I, rate_monitor_E, rate_monitor_I = simulate_brunels_network(2)

import numpy as np

# Count the number of spikes for each neuron in spike_monitor_E
# num_neurons_E = np.max(spike_monitor_E.i) + 1 if len(spike_monitor_E.i) > 0 else 0
# spike_counts = np.bincount(spike_monitor_E.i, minlength=num_neurons_E)

# # Save the spike counts array
# np.save('spike_monitor_E_counts.npy', spike_counts)

spike_counts = np.load('spike_monitor_E_counts.npy')
print(spike_counts)
print(len(spike_counts))

plt.plot(spike_counts)
plt.show()






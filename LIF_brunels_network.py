import brian2 as b2
from brian2 import NeuronGroup, Synapses, Network, PoissonInput, StateMonitor, SpikeMonitor, PopulationRateMonitor
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def simulate_brunels_network(input_data=None):


    
    np.random.seed(10061999) # Get same initial voltages for the network
    b2.seed(10061999) # Get the same connectivity and Poisson for the network

    # ----- Parameters ------
    sim_time = 1000. * b2.ms # simulation time

    w0 = 0.1 * b2.mV # synaptic weight strength
    g = 4 # Relative inhibitory strength g

    N_E = 4000 # number of excitatory neurons
    N_I = 500 # number of inhibitory neurons
    N_noise = 650 # number of noise neurons to each neuron in the network

    v_reset = +10. * b2.mV # reset potential
    v_rest = 0. * b2.mV # resting potential
    v_thresh = +20. * b2.mV # threshold potential
    abs_refractory_period = 2.0 * b2.ms # absolute refractory period
    membrane_time_scale = 20. * b2.ms # membrane time scale
    synaptic_delay = 1.5 * b2.ms # synaptic delay
    noise_rate = 13. * b2.Hz # noise rate
    noise_weight = w0 # noise weight. If you gonna pick a specific value, remember to *b2.mV to get the correct units
    # defining the postsyneaptic potential amplitudes 
    J_E = w0 
    J_I = -g * w0

    # ----- Input data -----

    feature_rates = []
    n_clusters = 0
    cluster_size = 100
    if input_data is not None:
        # Convert input_data into Hz magnitudes and sanitize
        max_rate_hz = 200.0
        if hasattr(input_data, 'units'):
            magnitudes = np.asarray(input_data / b2.Hz, dtype=float)
        else:
            magnitudes = np.asarray(input_data, dtype=float)
        magnitudes = np.nan_to_num(magnitudes, nan=0.0, posinf=max_rate_hz, neginf=0.0)
        magnitudes = np.clip(magnitudes, 0.0, max_rate_hz)
        input_quantities = magnitudes * b2.Hz

        # Normalize to a list of scalar rates (Quantities)
        if hasattr(input_quantities, 'ndim') and getattr(input_quantities, 'ndim', 0) > 0:
            feature_rates = [input_quantities[i] for i in range(int(input_quantities.shape[0]))]
        elif isinstance(input_quantities, (list, tuple, np.ndarray)):
            feature_rates = list(input_quantities)
        else:
            feature_rates = [input_quantities]

        n_clusters = len(feature_rates)
        print(f"Number of clusters: {n_clusters}")
        print(f"Cluster size of the network: {n_clusters * cluster_size}, network size: {N_E}")
        if n_clusters * cluster_size > N_E:
            print("OBS OBS cluster taken over whe whole excitory network")
  
   

    # ----- Dynamics -----
    lif_dynamics = """dv/dt = -(v-v_rest) / membrane_time_scale : volt (unless refractory)"""

    #Define the network 
    network = NeuronGroup(N_E + N_I, model=lif_dynamics, threshold="v > v_thresh", reset="v = v_reset", refractory=abs_refractory_period, method="linear")
    network.v = random.uniform(v_rest/b2.mV, high=v_thresh/b2.mV, size=(N_E+N_I))*b2.mV #Introduces random initial voltages between resting and threshold potential

    E_population = network[:N_E]
    I_population = network[N_E:]

    # ----- Define the synaptic connections -----
    E_synapses = Synapses(E_population, target=network, on_pre="v += J_E", delay=synaptic_delay)
    E_synapses.connect(p=0.1) # 10% of excitatory neurons are connected to the rest of the network
    I_synapses = Synapses(I_population, target=network, on_pre="v += J_I", delay=synaptic_delay)
    I_synapses.connect(p=0.1) # 10% of inhibitory neurons are connected to the rest of the network


    # ----- Background noise inputs -----
    noise_input = PoissonInput(target = network, target_var = "v", N=N_noise, rate=noise_rate, weight=noise_weight)

    # ----- Feature inputs -----
    if n_clusters > 0:
        input_weights = 1.5*w0
        N_input = 100 # Number of Poisson sources per target neuron in a cluster

        # Brian2 Subgroups require slicing, not arbitrary index arrays
        # Place clusters sequentially without overlap
        for k in range(n_clusters):
            start = k*cluster_size
            stop = start + cluster_size
            if stop > N_E:
                break
            E_cluster = E_population[start:stop]
            rate_k = feature_rates[k]
            # Ensure rate_k is a scalar Quantity (not an array); take magnitude if wrapped in array
            if hasattr(rate_k, 'shape') and getattr(rate_k, 'shape', ()) != ():
                rate_k = (np.asarray(rate_k / b2.Hz, dtype=float).reshape(-1)[0]) * b2.Hz
            feature_input = PoissonInput(target = E_cluster, target_var = "v", N=N_input, rate=rate_k, weight=input_weights)


    # ----- Collect the data of simualtion -----
    voltage_monitor_E = StateMonitor(E_population, variables="v", record=True)
    voltage_monitor_I = StateMonitor(I_population, variables="v", record=True)

    spike_monitor_E = SpikeMonitor(E_population)
    spike_monitor_I = SpikeMonitor(I_population)

    rate_monitor_E = PopulationRateMonitor(E_population)
    rate_monitor_I = PopulationRateMonitor(I_population)

    b2.run(sim_time)

    return voltage_monitor_E, voltage_monitor_I, spike_monitor_E, spike_monitor_I, rate_monitor_E, rate_monitor_I


if __name__ == "__main__":
    sample_input =np.array([1, 0.43, 0.3, 0.14])

    voltage_monitor_E, voltage_monitor_I, spike_monitor_E, spike_monitor_I, rate_monitor_E, rate_monitor_I = simulate_brunels_network(sample_input)

    # Find the neuron in E_population that fired the most
    n_E_neurons = voltage_monitor_E.v.shape[0]
    spike_counts = np.bincount(spike_monitor_E.i, minlength=n_E_neurons)
    # Identify maximum firing neuron and its spike count
    max_idx = np.argmax(spike_counts)
    max_spikes = spike_counts[max_idx]
    avg_spikes = np.mean(spike_counts)

    print(f"Neuron with max spikes (excitatory): {max_idx}")
    print(f"  Number of spikes: {max_spikes}")
    print(f"  Average spikes per E neuron: {avg_spikes:.2f}")
    # ----- Plot the results -----
    # Raster plot (spike monitor)
    import seaborn as sns

    sns.set(style="whitegrid", palette="muted", font_scale=1.15, rc={"axes.titlesize":18, "axes.labelsize":15})
    plt.figure(figsize=(12, 7))

    # Raster plot (spike monitor)
    ax1 = plt.subplot(2, 1, 1)
    sc = plt.scatter(spike_monitor_E.t / b2.ms, spike_monitor_E.i, c=spike_monitor_E.i, cmap="viridis", marker='.', s=8, alpha=0.8, edgecolors='none')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title('Spike Raster Plot', weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    cbar = plt.colorbar(sc, label="Neuron index", ax=ax1)
    cbar.set_alpha(1)
    cbar.update_normal(sc)  # ensures colorbar updates properly

    # Population rate plot
    ax2 = plt.subplot(2, 1, 2)
    times = rate_monitor_E.t / b2.ms
    rate = rate_monitor_E.smooth_rate(window='flat', width=5*b2.ms) / b2.Hz
    plt.plot(times, rate, color=sns.color_palette()[1], linewidth=2.2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')
    plt.title('Population Firing Rate', weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(pad=2.0)

    # Make axes more professional
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # Save the figure before showing it
    plt.savefig("No_input_700_noise.png", dpi=300, bbox_inches='tight')
    plt.show()
import brian2 as b2
from brian2 import NeuronGroup, Synapses, Network, PoissonInput, StateMonitor, SpikeMonitor, PopulationRateMonitor
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def simulate_brunels_network(input_data=None, g_strength=4.5):


    
    np.random.seed(10061999) # Get same initial voltages for the network
    b2.seed(10061999) # Get the same connectivity and Poisson for the network

    # ----- Parameters ------
    sim_time = 1000. * b2.ms # simulation time

    w0 = 1.0 * b2.mV # synaptic weight strength
    g = g_strength # Relative inhibitory strength g

    N_E = 4000 # number of excitatory neurons
    N_I = 1000 # number of inhibitory neurons
    N_noise = 80# number of noise neurons to each neuron in the network

    v_reset = +10. * b2.mV # reset potential
    v_rest = 0. * b2.mV # resting potential
    v_thresh = +20. * b2.mV # threshold potential
    abs_refractory_period = 2.0 * b2.ms # absolute refractory period
    membrane_time_scale = 20. * b2.ms # membrane time scale
    synaptic_delay = 1.5 * b2.ms # synaptic delay
    noise_rate = 10. * b2.Hz # noise rate
    noise_weight = w0 # noise weight. If you gonna pick a specific value, remember to *b2.mV to get the correct units
    # defining the postsyneaptic potential amplitudes 
    J_E = w0 
    J_I = -g * w0

    # ----- Input data -----

    feature_rates = []
    n_clusters = 0
    cluster_size = 100
    if input_data is not None:
        feature_rates = input_data

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
    E_synapses = Synapses(E_population, target=network, on_pre="v_post += J_E", delay=synaptic_delay)
    E_synapses.connect(p=0.1) # 10% of excitatory neurons are connected to the rest of the network
    I_synapses = Synapses(I_population, target=network, on_pre="v_post += J_I", delay=synaptic_delay)
    I_synapses.connect(p=0.1) # 10% of inhibitory neurons are connected to the rest of the network


    # ----- Background noise inputs -----
    noise_input = PoissonInput(target = network, target_var = "v", N=N_noise, rate=noise_rate, weight=noise_weight)

    # ----- Feature inputs -----
    if input_data is not None:

        feature1_input = PoissonInput(target = E_population[0:100], target_var = "v", N=100, rate=input_data[0], weight=1.5*w0)
        feature2_input = PoissonInput(target = E_population[100:200], target_var = "v", N=100, rate=input_data[1], weight=1.5*w0)
        feature3_input = PoissonInput(target = E_population[200:300], target_var = "v", N=100, rate=input_data[2], weight=1.5*w0)
        feature4_input = PoissonInput(target = E_population[300:400], target_var = "v", N=100, rate=input_data[3], weight=1.5*w0)
   

    # ----- Collect the data of simualtion -----
    voltage_monitor_E = StateMonitor(E_population, variables="v", record=True)
    voltage_monitor_I = StateMonitor(I_population, variables="v", record=True)

    spike_monitor_E = SpikeMonitor(E_population)
    spike_monitor_I = SpikeMonitor(I_population)

    rate_monitor_E = PopulationRateMonitor(E_population)
    rate_monitor_I = PopulationRateMonitor(I_population)

    b2.run(sim_time)

    cvs = []
    for n in range(min(200, int(np.max(spike_monitor_E.i)+1))):
        t = (spike_monitor_E.t[spike_monitor_E.i==n] / b2.ms).astype(float)
        if len(t) >= 3:
            isi = np.diff(t)
            if np.mean(isi) > 0:
                cvs.append(np.std(isi)/np.mean(isi))
    print("================================================")
    
    print(f"Mean CV: {np.mean(cvs)}")
    print(f"Std CV: {np.std(cvs)}")
    print(f"Median CV: {np.median(cvs)}")
    print("================================================")
    return voltage_monitor_E, voltage_monitor_I, spike_monitor_E, spike_monitor_I, rate_monitor_E, rate_monitor_I


if __name__ == "__main__":
    sample_input =np.array([ 0, 0,  0,  0]) * b2.Hz

    voltage_monitor_E, voltage_monitor_I, spike_monitor_E, spike_monitor_I, rate_monitor_E, rate_monitor_I = simulate_brunels_network(input_data=sample_input, g_strength=4.5)


 

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
    plt.savefig("sample_input2.png", dpi=300, bbox_inches='tight')
    plt.show()
import brian2 as b2
from brian2 import NeuronGroup, Synapses, Network, PoissonInput, StateMonitor, SpikeMonitor, PopulationRateMonitor
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def simulate_brunels_network(input_data=None):


    # ----- Input data -----
    if input_data is not None:
        input_poisson_rate = input_data*100. * b2.Hz
        N_input = 200
    else:
        input_poisson_rate = 0. * b2.Hz
        N_input = 0

    # ----- Parameters ------
    sim_time = 1000. * b2.ms # simulation time

    w0 = 0.1 * b2.mV # synaptic weight strength
    g = 4 # Relative inhibitory strength g

    N_E = 5000 # number of excitatory neurons
    N_I = 1000 # number of inhibitory neurons
    N_noise = 1000 # number of noise neurons

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


    # ----- External inputs -----
    noise_input = PoissonInput(target = network, target_var = "v", N=N_noise, rate=noise_rate, weight=noise_weight)
    input_input = PoissonInput(target = network, target_var = "v", N=N_input, rate=input_poisson_rate, weight=w0)


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
    voltage_monitor, spike_monitor, rate_monitor = simulate_brunels_network()

    # ----- Plot the results -----
    # Raster plot (spike monitor)
    import seaborn as sns

    sns.set(style="whitegrid", palette="muted", font_scale=1.15, rc={"axes.titlesize":18, "axes.labelsize":15})
    plt.figure(figsize=(12, 7))

    # Raster plot (spike monitor)
    ax1 = plt.subplot(2, 1, 1)
    sc = plt.scatter(spike_monitor.t / b2.ms, spike_monitor.i, c=spike_monitor.i, cmap="viridis", marker='.', s=8, alpha=0.8, edgecolors='none')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title('Spike Raster Plot', weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    cbar = plt.colorbar(sc, label="Neuron index", ax=ax1)
    cbar.set_alpha(1)
    cbar.update_normal(sc)  # ensures colorbar updates properly

    # Population rate plot
    ax2 = plt.subplot(2, 1, 2)
    times = rate_monitor.t / b2.ms
    rate = rate_monitor.smooth_rate(window='flat', width=5*b2.ms) / b2.Hz
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

    plt.show()
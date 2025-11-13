import brian2 as b2
from brian2 import NeuronGroup, Synapses, Network, PoissonInput, StateMonitor, SpikeMonitor, PopulationRateMonitor
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def _nu_th(v_thresh, v_rest, J_E, tau_m, K_ext):
    theta = (v_thresh - v_rest)                   # volt
    # return in Hz (float)
    return float(theta / (J_E * tau_m * K_ext) / b2.Hz)

def simulate_brunels_network(input_data=None, g_strength=4.5, eta=1.0, p_rec=0.1, use_ext_to_I=True, w0=0.1*b2.mV):


    # Reset Brian's magic network so multiple calls don't mix old/new objects
    b2.start_scope()
    
    np.random.seed(10061999) # Get same initial voltages for the network
    b2.seed(10061999) # Get the same connectivity and Poisson for the network

    # ==== Parameters ====
    sim_time = 1000. * b2.ms # simulation time
    g = g_strength # Relative inhibitory strength g

    N_E, N_I = 4000, 1000 # number of excitatory and inhibitory neurons

    v_reset, v_rest = +10. * b2.mV, 0. * b2.mV # reset potential and resting potential
    v_thresh = 20. * b2.mV # threshold potential
    abs_refractory_period = 2.0 * b2.ms # absolute refractory period
    tau_m = 20. * b2.ms # membrane time scale
    synaptic_delay = 1.0 * b2.ms # synaptic delay

    #Synaptic amplitudes
    J_E = w0 
    J_I = -g * w0


    # === NETWORK =====
    lif = "dv/dt = -(v - v_rest) / tau_m : volt (unless refractory)"
    network = NeuronGroup(N_E + N_I, model=lif, threshold="v > v_thresh", reset="v = v_reset", refractory=abs_refractory_period, method="linear")
    network.v = random.uniform(v_rest/b2.mV, high=v_thresh/b2.mV, size=(N_E+N_I)) * b2.mV
    E_population = network[:N_E]
    I_population = network[N_E:]



    # ==== Connectivity ====
    E_syn = Synapses(E_population, network, on_pre="v_post += J_E", delay=synaptic_delay)
    E_syn.connect(p=p_rec)
    I_syn = Synapses(I_population, network, on_pre="v_post += J_I", delay=synaptic_delay)
    I_syn.connect(p=p_rec)

    # ==== External drive  ====
    K_ext = int(p_rec * N_E)
    nu_th = _nu_th(v_thresh, v_rest, J_E, tau_m, K_ext) * b2.Hz
    nu_ext = eta * nu_th
    noise_weight = J_E

    # External poisson bombardment to E (And I if chosen)
    ext_E = PoissonInput(target = E_population, target_var = "v", N=K_ext, rate=nu_ext, weight=noise_weight)
    if use_ext_to_I:
        ext_I = PoissonInput(target = I_population, target_var = "v", N=K_ext, rate=nu_ext, weight=noise_weight)



    # ==== Input data ====


    feature_rates = []
    n_clusters = 0
    cluster_size = 100
    if input_data is not None:
        feature_rates = input_data

        n_clusters = len(feature_rates)
        # print(f"Number of clusters: {n_clusters}")
        # print(f"Cluster size of the network: {n_clusters * cluster_size}, network size: {N_E}")
        if n_clusters * cluster_size > N_E:
            print("OBS OBS cluster taken over whe whole excitory network")
  
   



    # ----- Feature inputs -----
    if input_data is not None:

        feature1_input = PoissonInput(target = E_population[0:100], target_var = "v", N=100, rate=input_data[0], weight=1.5*w0)
        feature2_input = PoissonInput(target = E_population[100:200], target_var = "v", N=100, rate=input_data[1], weight=1.5*w0)
        feature3_input = PoissonInput(target = E_population[200:300], target_var = "v", N=100, rate=input_data[2], weight=1.5*w0)
        feature4_input = PoissonInput(target = E_population[300:400], target_var = "v", N=100, rate=input_data[3], weight=1.5*w0)
   

    # ----- Collect the data of simualtion -----
    voltage_monitor_E = StateMonitor(E_population[400:], variables="v", record=True)
    voltage_monitor_I = StateMonitor(I_population, variables="v", record=True)

    spike_monitor_E = SpikeMonitor(E_population[400:])
    spike_monitor_I = SpikeMonitor(I_population)

    rate_monitor_E = PopulationRateMonitor(E_population[400:])
    rate_monitor_I = PopulationRateMonitor(I_population)

    b2.run(sim_time)


    burn_ms = 200.0
    t_all   = np.asarray(spike_monitor_E.t[:]/b2.ms, dtype=float)
    i_all   = np.asarray(spike_monitor_E.i[:], dtype=int)

    # Restrict to post burn-in spikes
    mask = t_all > burn_ms
    t_all = t_all[mask]; i_all = i_all[mask]

    # Build S for correlations (use a smaller bin to catch fast synchrony)
    bin_ms = 2.0
    T = float(sim_time/b2.ms) - burn_ms
    nbins = int(T/bin_ms)
    bins = np.linspace(burn_ms, burn_ms+T, nbins+1)

    if i_all.size == 0:
        # No spikes after burn-in: skip correlation/CV computation gracefully
        N = 0
        mean_paircorr = np.nan
        cv2s, frates = [], []
    else:
        N = min(200, int(np.max(i_all))+1)
        S = np.zeros((N, nbins))
        for n in range(N):
            t = t_all[i_all==n]
            S[n], _ = np.histogram(t, bins=bins)

        C = np.corrcoef(S)
        mean_paircorr = np.nanmean(C[np.triu_indices_from(C,1)])

        # CV2 (less rate-dependent than CV)
        def cv2(isi):
            return np.mean(2*np.abs(np.diff(isi))/(isi[1:]+isi[:-1]))

        cv2s, frates = [], []
        for n in range(N):
            t = t_all[i_all==n]
            if len(t) >= 4:
                isi = np.diff(t)
                if np.mean(isi)>0:
                    cv2s.append(cv2(isi))
            frates.append(len(t) / (T/1000.0))  # Hz
    print("================================================")
    print("Mean pairwise corr:", mean_paircorr)
    print("mean CV2:", np.mean(cv2s))
    print("mean firing rate:", np.mean(frates))
    print("================================================")
    return voltage_monitor_E, voltage_monitor_I, spike_monitor_E, spike_monitor_I, rate_monitor_E, rate_monitor_I, np.mean(cv2s), np.mean(frates), mean_paircorr


if __name__ == "__main__":
    sample_input =np.array([0, 0,  0, 0]) * b2.Hz

    # # Prepare g and eta grid
    # g_values = np.linspace(3.6, 6.0, 6)
    # eta_values = np.linspace(0.7, 1.4, 8)
    # cv_matrix = np.zeros((len(eta_values), len(g_values)))
    # rate_matrix = np.zeros((len(eta_values), len(g_values)))
    # annot_matrix = np.empty(cv_matrix.shape, dtype=object)

    # for gi, g in enumerate(g_values):
    #     for ei, eta in enumerate(eta_values):
    #         voltage_monitor_E, voltage_monitor_I, spike_monitor_E, spike_monitor_I, rate_monitor_E, rate_monitor_I, cv, mean_rate = simulate_brunels_network(
    #             input_data=sample_input, g_strength=g, eta=eta)
    #         cv_matrix[ei, gi] = cv
    #         rate_matrix[ei, gi] = mean_rate
    #         annot_matrix[ei, gi] = f"{cv:.2f}\n{mean_rate:.1f}"

    #         print(f"g: {g}, eta: {eta}, CV: {cv}, Mean rate: {mean_rate}")

    # # Plotting heatmap of CV (annotated with mean rate)
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # plt.figure(figsize=(8, 6))
    # ax = sns.heatmap(
    #     cv_matrix, 
    #     xticklabels=np.round(g_values, 2), 
    #     yticklabels=np.round(eta_values, 2),
    #     annot=annot_matrix, fmt="", cmap="viridis"
    # )
    # plt.xlabel("g")
    # plt.ylabel("eta")
    # plt.title("Heatmap of Mean CV (top) and Mean Rate (Hz, bottom)")
    # plt.tight_layout()
    # plt.show()

    voltage_monitor_E, voltage_monitor_I, spike_monitor_E, spike_monitor_I, rate_monitor_E, rate_monitor_I, _, _, _ = simulate_brunels_network(input_data=sample_input, g_strength=2, eta=0.94)


 

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
    plt.title('Spike Raster Plot (SR regime)', weight='bold')
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
    plt.savefig("output_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
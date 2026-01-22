import nest
nest.set_verbosity("M_ERROR")
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import time
class Brunel:
    def __init__(self, input=None, stdp = True, reset = True, N_neurons = 1000):
        """
        Initialize the Brunel model.
        Args:
            input: Input data to the network. ([1, N_features]numpy array)
            stdp: Whether to use STDP or not. (Boolean)
            reset: Whether to reset the NEST kernel or not. (Boolean)
        """
        if reset:
            nest.ResetKernel()  # Reset the NEST kernel
            print("Resetting NEST kernel")
        else:   
            print("NEST kernel is NOT reset")
        nest.print_time = False
        nest.overwrite_files = True
        seed = 100699
        nest.SetKernelStatus({"resolution": 0.1, "rng_seed": seed}) # Simulation resolution (ms) and random seed
        np.random.seed(seed)

        self.simtime = 1000.0  # Simulation time (ms)
        self.delay = 1.5  # Synaptic delay (ms)
        self.g = 4.0  # Relative inhibitory strength
        self.eta = 1.1  # External rate in units of threshold
        self.epsilon = 0.01  # Connection probability
        self.N_neurons = N_neurons  # Total number of neurons
        self.NI = self.N_neurons // 5  # Number of inhibitory neurons
        self.NE = self.N_neurons - self.NI  # Number of excitatory neurons (four times as many as inhibitory)
        self.N_rec = 50  # Number of recorded excitatory neurons

        # Connectivity parameters
        self.CE = int(self.epsilon * self.NE)  # number of excitatory synapses per neuron
        self.CI = int(self.epsilon * self.NI)  # number of inhibitory synapses per neuron
        self.C_tot = int(self.CI + self.CE)  # total number of synapses per neuron


        # LIF neuron parameters
        self.tauSyn = 0.5 # synaptic time constant in ms
        self.tauMem = 20.0  # time constant of membrane potential in ms
        self.CMem = 250.0 # capacitance in pF
        self.theta = 20.0  # membrane threshold potential in mV
        self.neuron_params = {"C_m": self.CMem, "tau_m": self.tauMem, "tau_syn_ex": self.tauSyn, "tau_syn_in": self.tauSyn, "t_ref": 2.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": self.theta}
        self.J = 0.1  # postsynaptic amplitude mV
        self.J_unit = self.ComputePSPnorm(self.tauMem, self.CMem, self.tauSyn) # [mV / pA]
        self.J_ex = self.J / self.J_unit  # amplitude of excitatory postsynaptic potential [pA]
        self.J_in = -self.g * self.J_ex  # amplitude of inhibitory postsynaptic potential [pA]
        self.stim_weight_scaler = 50
        self.stim_weight = self.stim_weight_scaler * self.J_ex

        # Threshold rate, external firing rate and converted spikes per second
        self.nu_th = (self.theta * self.CMem) / (self.J_ex * self.CE * np.exp(1.0) * self.tauMem * self.tauSyn)
        self.nu_ex = self.eta * self.nu_th 
        self.p_rate_scaler = 1.1
        self.p_rate = (1000.0 * self.nu_ex * self.CE) / self.p_rate_scaler# Multiply be 1000 to convert to Hz

        # Synapse parameters
        self.stdp = stdp
        self.stdp_params = {"weight": self.J_ex, "delay": self.delay, "lambda": 0.001, "alpha": 0.5,  "Wmax": 100.0}
        self.static_params = {"weight": self.J_ex, "delay": self.delay}
        self.inhibitory_params = {"weight": self.J_in, "delay": self.delay}
        self.bernoulli_conn = {"rule": "pairwise_bernoulli", "p": self.epsilon, "allow_autapses": False}
        self.stimulus_params = {"weight": self.stim_weight, "delay": self.delay}

        self.input = input
        if input is not None:
            self.procentage_to_group = 0.05
            self.n_features = np.shape(self.input)[1]
            self.group_size = int(self.N_neurons * self.procentage_to_group)
            self.feature_size = self.group_size * self.n_features
                

        
    def print_summary(self):
        summary = ""
        summary += "======== Quick Summary of some of the parameters ========\n"
        summary += f"Number of neurons: {self.N_neurons}\n"
        summary += f"Number of inhibitory neurons: {self.NI}\n"
        summary += f"Number of excitatory neurons: {self.NE}\n"
        summary += f"Relative inhibitory strength: {self.g}\n"
        summary += f"External rate in units of threshold: {self.eta}\n"
        summary += f"Connection probability: {self.epsilon}\n"
        summary += f"Simulation time: {self.simtime} ms\n"
        summary += "--------------------------------\n"
        summary += f"Number of features: {self.n_features}\n"
        summary += f"Procentage of total neurons to be grouped: {self.procentage_to_group * 100}%\n"
        summary += f"Group size: {self.group_size}\n"
        summary += f"Feature size: {self.feature_size}\n"
        summary += "--------------------------------\n"
        summary += f"Stimulus weight scaler: {self.stim_weight_scaler}\n"
        summary += f"STDP: {self.stdp}\n"
        summary += f"P rate scaler: {self.p_rate_scaler}\n"
        summary += f"P rate: {self.p_rate:.2f} Hz\n"
        print(summary)
        return summary
    
    def LambertWm1(self, x):
        return sp.lambertw(x, k=-1 if x < 0 else 0).real
    def ComputePSPnorm(self, tauMem, CMem, tauSyn):
        a = tauMem / tauSyn
        b = 1.0 / tauSyn - 1.0 / tauMem
        t_max = 1.0 / b * (-self.LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)
        return (np.exp(1.0) / (tauSyn * CMem * b) *
                ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b -
                 t_max * np.exp(-t_max / tauSyn)))

    def build_network(self):
        print("Building network...")

        # Creating nodes
        self.input_nodes = nest.Create("iaf_psc_alpha", self.feature_size, params=self.neuron_params)
        self.nodes_ex = nest.Create("iaf_psc_alpha", self.NE - self.feature_size, params=self.neuron_params)
        self.nodes_in = nest.Create("iaf_psc_alpha", self.NI, params=self.neuron_params)
        self.noise = nest.Create("poisson_generator", self.N_neurons, params={"rate": self.p_rate})
        self.espikes = nest.Create("spike_recorder")
        self.ispikes = nest.Create("spike_recorder")
        self.input_spikes = nest.Create("spike_recorder")

        # Random initial membrane potentials
        V_init = np.random.uniform(self.neuron_params["V_reset"], self.theta, size=self.N_neurons)
        nest.SetStatus(self.input_nodes + self.nodes_ex + self.nodes_in, [{"V_m": float(v)} for v in V_init])

        # Defining synapse models
        nest.CopyModel("static_synapse", "background", self.static_params)
        nest.CopyModel("static_synapse", "inhibitory", self.inhibitory_params)
        if self.stdp:
            nest.CopyModel("stdp_synapse", "excitatory_stdp", self.stdp_params)
        else:   
            nest.CopyModel("static_synapse", "excitatory_stdp", self.static_params)
        nest.CopyModel("static_synapse", "stimulus", self.stimulus_params)
        nest.CopyModel("static_synapse", "excitatory_static", self.static_params)

        # Connecting nodes
        nest.Connect(self.noise,self.input_nodes + self.nodes_ex + self.nodes_in, conn_spec = "one_to_one", syn_spec="background") # Background noise to all neurons : static synapse
        nest.Connect(self.nodes_ex, self.nodes_in, conn_spec=self.bernoulli_conn, syn_spec="excitatory_static") # E -> I : Static synapse
        nest.Connect(self.nodes_ex, self.nodes_ex, conn_spec=self.bernoulli_conn, syn_spec="excitatory_static") # E -> E : Static synapse

        start_ex = 0
        start_input = 0
        stop_ex = len(self.nodes_ex) // self.n_features
        stop_input = len(self.input_nodes) // self.n_features
        for i in range(self.n_features):
            nest.Connect(self.input_nodes[start_input:stop_input], self.nodes_ex[start_ex:stop_ex], conn_spec="all_to_all", syn_spec="excitatory_stdp") # Input -> E : STDP synapse
            start_ex = stop_ex
            stop_ex = start_ex + len(self.nodes_ex) // self.n_features
            start_input = stop_input
            stop_input = start_input + len(self.input_nodes) // self.n_features

        
        nest.Connect(self.input_nodes, self.nodes_in, conn_spec=self.bernoulli_conn, syn_spec="excitatory_static") # Input -> I : Static synapse
        nest.Connect(self.nodes_in, self.nodes_ex + self.nodes_in, conn_spec=self.bernoulli_conn, syn_spec="inhibitory") # I -> (E + I) : Static synapse
        
        # Connect all neurons to spike recorders 
        nest.Connect(self.nodes_ex, self.espikes)
        nest.Connect(self.nodes_in, self.ispikes)
        nest.Connect(self.input_nodes, self.input_spikes)
        

        if self.input is not None:
            self.feature_generators = [] 
            for i in range(self.n_features):
                gens = nest.Create("poisson_generator", self.group_size, params={"rate": 0.0})
                start = i * self.group_size
                stop = (i + 1) * self.group_size
                nest.Connect(gens, self.input_nodes[start:stop], conn_spec = "one_to_one", syn_spec="stimulus")
                self.feature_generators.append(gens)



    
    def give_input(self, input):
        if np.shape(input)[0] != 1 or not isinstance(input, np.ndarray):
            raise ValueError(f"Input data must be of shape (1, N_features). And it also has to be numpy array \\ Input gotten is {np.shape(input)} and type {type(input)}")

        # Create one poisson generator for each feature
        feature_rates = input.flatten().astype(float)
        for gens, r in zip(self.feature_generators, feature_rates):
            nest.SetStatus(gens, {"rate": r})
            

  
    def simulate(self):
        print("Simulation running...")
        nest.Simulate(self.simtime)
    

    def get_spike_vector_window(self, t_start, t_end):

        ex_ids = np.asarray(self.nodes_ex, dtype=np.int64)
        in_ids = np.asarray(self.nodes_in, dtype=np.int64)
        input_ids = np.asarray(self.input_nodes, dtype=np.int64)

        events_ex = self.espikes.events
        events_in = self.ispikes.events
        events_input = self.input_spikes.events
        
        times_ex = np.asarray(events_ex["times"], dtype=float)
        times_in = np.asarray(events_in["times"], dtype=float)
        times_input = np.asarray(events_input["times"], dtype=float)
        senders_ex = np.asarray(events_ex["senders"], dtype=np.int64)
        senders_in = np.asarray(events_in["senders"], dtype=np.int64)
        senders_input = np.asarray(events_input["senders"], dtype=np.int64)

        # window mask
        m_ex = (times_ex > t_start) & (times_ex <= t_end)
        m_in = (times_in > t_start) & (times_in <= t_end)
        m_input = (times_input > t_start) & (times_input <= t_end)  
        senders_ex_w = senders_ex[m_ex]
        senders_in_w = senders_in[m_in]
        senders_input_w = senders_input[m_input]

        spike_counts_ex = np.bincount(senders_ex_w - ex_ids[0], minlength=len(ex_ids))
        spike_counts_in = np.bincount(senders_in_w - in_ids[0], minlength=len(in_ids))
        spike_counts_input = np.bincount(senders_input_w - input_ids[0], minlength=len(input_ids))

    

        return spike_counts_ex, spike_counts_in, spike_counts_input


        
    def get_firing_rates_window(self, t_start, t_end):
        dt_s = (t_end - t_start) / 1000.0

        # Excitatory spikes in window
        ev_ex = self.espikes.events
        ev_in = self.ispikes.events
        ev_input = self.input_spikes.events

        times_ex = np.asarray(ev_ex["times"], dtype=float)
        times_in = np.asarray(ev_in["times"], dtype=float)
        times_input = np.asarray(ev_input["times"], dtype=float)

        senders_ex = np.asarray(ev_ex["senders"], dtype=np.int64)
        senders_in = np.asarray(ev_in["senders"], dtype=np.int64)
        senders_input = np.asarray(ev_input["senders"], dtype=np.int64)

        m_ex = (times_ex > t_start) & (times_ex <= t_end)
        m_in = (times_in > t_start) & (times_in <= t_end)
        m_input = (times_input > t_start) & (times_input <= t_end)

        n_spikes_ex = m_ex.sum()
        n_spikes_in = m_in.sum()
        n_spikes_input = m_input.sum()
        fr_ex = n_spikes_ex / dt_s / (self.NE - self.feature_size)
        fr_in = n_spikes_in / dt_s / self.NI
        fr_input = n_spikes_input / dt_s / self.feature_size

        return fr_ex, fr_in, fr_input


    def get_stdp_weights(self, bins=100, show_top_bottom=False, plot=True, return_weights=False, folder_name="weights", y = None, stimtime = None, filename = None):
        """
        Plot histogram of weights for synapses that use the 'excitatory_stdp' model.
        Also print the top 10 synapses with highest and lowest weights after simulation.
        Call this AFTER simulate().
        """
        all_neurons = self.nodes_ex + self.nodes_in + self.input_nodes

        # Get all STDP synapses (from E population to all neurons) that use this synapse model
        conns_input_E = nest.GetConnections(self.input_nodes, all_neurons, synapse_model="excitatory_stdp")
        conns_all = nest.GetConnections(all_neurons, all_neurons)
        w_all = np.asarray(conns_all.get("weight"), dtype=float)
        w_input_E = np.asarray(conns_input_E.get("weight"), dtype=float)

        # Calculate mean and std
        w_mean = np.mean(w_input_E)
        w_std = np.std(w_input_E)
        w_mean_all = np.mean(w_all)
        w_std_all = np.std(w_all)
        
        # --- Print top 10 highest and lowest weights with their connection details
        # Get source and target GIDs
        sources = np.asarray(conns_input_E.get('source'), dtype=int)
        targets = np.asarray(conns_input_E.get('target'), dtype=int)

        if show_top_bottom:
            # Compose a list of (weight, source, target)
            connections = list(zip(w_input_E, sources, targets))
            # Sort by weight
            sorted_conns = sorted(connections, key=lambda x: x[0])
            # Lowest 10
            print("Lowest 10 STDP weights (weight, source, target):")
            for w, s, t in sorted_conns[:10]:
                print(f"  {w:.4f} mV\tsource: {s}\ttarget: {t}")
            # Highest 10
            print("\nHighest 10 STDP weights (weight, source, target):")
            for w, s, t in sorted_conns[-10:]:
                print(f"  {w:.4f} mV\tsource: {s}\ttarget: {t}")
        
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(16, 5))
            
            # Histogram for w_input_E (STDP weights from input to E population)
            axs[0].hist(w_input_E, bins=bins)
            axs[0].set_xlabel("Weight (pA)")
            axs[0].set_ylabel("Count")
            axs[0].set_title("Histogram of STDP synaptic weights (input → all, after simulation)\n"
                             f"Mean: {w_mean:.3f} pA, Std: {w_std:.3f} pA, class: {y}, stimtime per sample: {stimtime} ms")
            axs[0].axvline(w_mean, color='r', linestyle=':', linewidth=2, label=f"Mean ({w_mean:.3f} pA)")
            axs[0].legend()

            # Histogram for w_all (all weights in the network)
            axs[1].hist(w_all, bins=bins, color='orange')
            axs[1].set_xlabel("Weight (pA)")
            axs[1].set_ylabel("Count")
            axs[1].set_title("Histogram of all synaptic weights (after simulation)\n"
                             f"Mean: {w_mean_all:.3f} pA, Std: {w_std_all:.3f} pA")
            axs[1].axvline(w_mean_all, color='r', linestyle=':', linewidth=2, label=f"Mean ({w_mean_all:.3f} pA)")
            axs[1].legend()

            plt.tight_layout()
            import time
            plt.savefig(f"data/{folder_name}/{filename}.png")
            #plt.show()
            plt.close()

        if return_weights:
            return w_input_E

    def plot_weight_matrix(self):
        
        all_nodes = self.nodes_ex + self.nodes_in + self.input_nodes
        all_gids = np.asarray(all_nodes, dtype=int)

        gid_to_idx = {gid: i for i, gid in enumerate(all_gids)}

        N = len(all_gids)
        A = np.zeros((N, N))

        # This now works
        conns = nest.GetConnections(all_nodes, all_nodes)

        sources = conns.get("source")
        targets = conns.get("target")
        weights = conns.get("weight")

        for s, t, w in zip(sources, targets, weights):
            A[gid_to_idx[s], gid_to_idx[t]] = w

        max_n = 600
        A_plot = A[:max_n, :max_n]

        plt.figure(figsize=(7, 6))
        im = plt.imshow(A_plot, aspect="auto")

        # --- Draw grid lines to separate squares ---
        # Set major ticks at every integer (since each neuron is a "square")
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, A_plot.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, A_plot.shape[0], 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=0.2, alpha=0.6)
        ax.tick_params(which='minor', bottom=False, left=False)

        plt.colorbar(im, label="weight")
        plt.xlabel("target neuron index")
        plt.ylabel("source neuron index")
        plt.title("Full synaptic weight matrix (cropped)")
        plt.tight_layout()
        plt.show()
                    

    def plot_spike_distribution(self, t0, t1, info_str = None):
        """Plot a distribution of spikes from all excitatory and inhibitory neurons."""
        events_ex = self.espikes.events
        events_in = self.ispikes.events
        events_input = self.input_spikes.events

        senders_all = np.concatenate([events_ex["senders"][(events_ex["times"] > t0) & (events_ex["times"] <= t1)] , events_in["senders"][(events_in["times"] > t0) & (events_in["times"] <= t1)] , events_input["senders"][(events_input["times"] > t0) & (events_input["times"] <= t1)]])
        senders_unique_all = np.unique(senders_all)
        senders_sorted_all = np.sort(senders_unique_all)

        senders_ex = events_ex["senders"][(events_ex["times"] > t0) & (events_ex["times"] <= t1)] 
        senders_in = events_in["senders"][(events_in["times"] > t0) & (events_in["times"] <= t1)] 
        senders_input = events_input["senders"][(events_input["times"] > t0) & (events_input["times"] <= t1)] 

        senders_unique_ex = np.unique(senders_ex)
        senders_unique_in = np.unique(senders_in)
        senders_unique_input = np.unique(senders_input)

        senders_sorted_ex = np.sort(senders_unique_ex)
        senders_sorted_in = np.sort(senders_unique_in)
        senders_sorted_input = np.sort(senders_unique_input)
        
        plt.figure(figsize=(15, 5))  # Changed plot size here

        for sender in senders_unique_ex:
            num_spikes = np.sum(senders_ex == sender)
            sender_idx = np.where(senders_sorted_all == sender)[0][0]  # index among unique senders
            plt.vlines(sender_idx, 0, num_spikes, label="Excitatory", color='skyblue', alpha=0.8)
        
        for sender in senders_unique_in:
            num_spikes = np.sum(senders_in == sender)
            sender_idx = np.where(senders_sorted_all == sender)[0][0]  # index among unique senders
            plt.vlines(sender_idx, 0, num_spikes, label="Inhibitory", color='salmon', alpha=0.8)
        
        for sender in senders_unique_input:
            num_spikes = np.sum(senders_input == sender)
            sender_idx = np.where(senders_sorted_all == sender)[0][0]  # index among unique senders
            plt.vlines(sender_idx, 0, num_spikes, label="Input", color='mediumseagreen', alpha=0.8)



        plt.xlabel("Neuron index")
        plt.ylabel("Number of spikes")
        plt.title(f"Distribution of spikes from all excitatory neurons \n Extra info: {info_str}")
        plt.tight_layout()
        plt.grid(1)
        plt.savefig(f"data/spike_distribution/spike_distribution_{time.time()}.png")
        plt.show()
        #plt.close()




    def plot_raster(self):
        """Plot a raster plot of spikes from all excitatory and inhibitory neurons."""
        # Get spike events from both recorders
        events_ex = self.espikes.events
        events_in = self.ispikes.events
        events_input = self.input_spikes.events
        
        
        plt.figure(figsize=(10, 6))
        
        # Plot excitatory spikes in blue
        if len(events_ex["times"]) > 0:
            plt.scatter(events_ex["times"], events_ex["senders"], s=1, color='C0', alpha=0.6, label='Excitatory')
        
        # Plot inhibitory spikes in orange/red
        if len(events_in["times"]) > 0:
            plt.scatter(events_in["times"], events_in["senders"], s=1, color='C1', alpha=0.6, label='Inhibitory')
        
        if len(events_input["times"]) > 0:
            plt.scatter(events_input["times"], events_input["senders"], s=1, color='C2', alpha=0.6, label='Input')
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron GID")
        plt.title("Raster plot of all neurons (Excitatory: blue, Inhibitory: orange)")
        plt.xlim([0, self.simtime])
        plt.legend(markerscale=6, loc='upper right')
        plt.tight_layout()
        #plt.savefig(f"data/raster_plot_{time.time()}.png")
        plt.show()




if __name__ == "__main__":
    # Example run and the order
    brunel = Brunel()           # 1
    brunel.build_network()      # 2
    brunel.simulate()           # 3

    # These can you run any order
    brunel.get_stdp_weights()   
    brunel.get_spike_vector()   
    brunel.get_average_firing() 
    brunel.get_stdp_weights()   
    brunel.plot_raster()        


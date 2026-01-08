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
        self.g = 4.5  # Relative inhibitory strength
        self.eta = 1.0  # External rate in units of threshold
        self.epsilon = 0.1  # Connection probability
        self.N_neurons = N_neurons  # Total number of neurons
        self.NI = self.N_neurons // 5  # Number of inhibitory neurons
        self.NE = self.N_neurons - self.NI  # Number of excitatory neurons (four times as many as inhibitory)
        self.N_rec = 50  # Number of recorded excitatory neurons

        # Connectivity parameters
        self.CE = int(self.epsilon * self.NE)  # number of excitatory synapses per neuron
        self.CI = int(self.epsilon * self.NI)  # number of inhibitory synapses per neuron
        self.C_tot = int(self.CI + self.CE)  # total number of synapses per neuron

        print("CE", self.CE)
        print("CI", self.CI)
        print("C_tot", self.C_tot)

        # LIF neuron parameters
        self.tauSyn = 0.5 # synaptic time constant in ms
        self.tauMem = 20.0  # time constant of membrane potential in ms
        self.CMem = 250.0 # capacitance in pF
        self.theta = 20.0  # membrane threshold potential in mV
        self.neuron_params = {"C_m": self.CMem, "tau_m": self.tauMem, "tau_syn_ex": self.tauSyn, "tau_syn_in": self.tauSyn, "t_ref": 2.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": self.theta}
        self.J = 0.1  # postsynaptic amplitude in mV
        self.J_unit = self.ComputePSPnorm(self.tauMem, self.CMem, self.tauSyn) # [mV / pA]
        self.J_ex = self.J / self.J_unit  # amplitude of excitatory postsynaptic potential [pA]
        self.J_in = -self.g * self.J_ex  # amplitude of inhibitory postsynaptic potential [pA]
        self.stim_weight = 80 * self.J_ex

        # Threshold rate, external firing rate and converted spikes per second
        self.nu_th = (self.theta * self.CMem) / (self.J_ex * self.CE * np.exp(1.0) * self.tauMem * self.tauSyn)
        self.nu_ex = self.eta * self.nu_th 
        self.p_rate = (1000.0 * self.nu_ex * self.CE) / 1.1# Multiply be 1000 to convert to Hz

        # Synapse parameters
        self.stdp = stdp
        self.stdp_params = {"weight": self.J_ex, "delay": self.delay, "lambda": 0.001, "alpha": 0.5,  "Wmax": 30.0}
        self.static_params = {"weight": self.J_ex, "delay": self.delay}
        self.inhibitory_params = {"weight": self.J_in, "delay": self.delay}
        self.bernoulli_conn = {"rule": "pairwise_bernoulli", "p": self.epsilon, "allow_autapses": False}
        self.stimulus_params = {"weight": self.stim_weight, "delay": self.delay}

        self.input = input
        if input is not None:
            
            self.n_features = np.shape(self.input)[1]
            self.group_size = int(self.N_neurons * 0.05)
            self.feature_size = self.group_size * self.n_features
                

        
    def print_summary(self):
        print("======== Quick Summary of some of the parameters =======")
        print(f"Number of neurons: {self.N_neurons}")
        print(f"Number of inhibitory neurons: {self.NI}")
        print(f"Number of excitatory neurons: {self.NE}")
        #print(f"Number of recorded excitatory neurons: {self.N_rec}")
        print(f"Relative inhibitory strength: {self.g}")
        print(f"External rate in units of threshold: {self.eta}")
        print(f"Connection probability: {self.epsilon}")
        print(f"Simulation time: {self.simtime}")
        #print(f"Synaptic delay: {self.delay}")
        print("--------------------------------")
        print(f"Number of features: {self.n_features}")
        print(f"Group size: {self.group_size}")
        print(f"Feature size: {self.feature_size}")
    
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
        self.nodes_ex = nest.Create("iaf_psc_alpha", self.NE, params=self.neuron_params)
        self.nodes_in = nest.Create("iaf_psc_alpha", self.NI, params=self.neuron_params)
        noise = nest.Create("poisson_generator", self.N_neurons, params={"rate": self.p_rate})
        self.espikes = nest.Create("spike_recorder")
        self.ispikes = nest.Create("spike_recorder")

        # Random initial membrane potentials
        V_init = np.random.uniform(self.neuron_params["V_reset"], self.theta, size=self.N_neurons)
        nest.SetStatus(self.nodes_ex + self.nodes_in, [{"V_m": float(v)} for v in V_init])

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
        nest.Connect(noise, self.nodes_ex + self.nodes_in, conn_spec = "one_to_one", syn_spec="background") # Background noise to all neurons : static synapse
        nest.Connect(self.nodes_ex, self.nodes_in, conn_spec=self.bernoulli_conn, syn_spec="excitatory_static") # E -> I : Static synapse
        nest.Connect(self.nodes_ex, self.nodes_ex, conn_spec=self.bernoulli_conn, syn_spec="excitatory_stdp") # E -> E : Static synapse

        nest.Connect(self.nodes_in, self.nodes_ex + self.nodes_in, conn_spec=self.bernoulli_conn, syn_spec="inhibitory") # I -> (E + I) : Static synapse
        
        # Connect all neurons to spike recorders 
        nest.Connect(self.nodes_ex, self.espikes)
        nest.Connect(self.nodes_in, self.ispikes)

        if self.input is not None:
            self.feature_generators = [] 
            for i in range(self.n_features):
                gens = nest.Create("poisson_generator", self.group_size, params={"rate": 0.0})
                start = i * self.group_size
                stop = (i + 1) * self.group_size
                nest.Connect(gens, self.nodes_ex[start:stop], conn_spec = "one_to_one", syn_spec="stimulus")
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
    

    def get_spike_vector_window(self, t_start, t_end, N=200):
        """
        Spike counts per excitatory neuron for spikes with t_start < time <= t_end,
        excluding the first N excitatory neurons.
        """
        ex_ids = np.asarray(self.nodes_ex, dtype=np.int64)

        events_ex = self.espikes.events
        times = np.asarray(events_ex["times"], dtype=float)
        senders = np.asarray(events_ex["senders"], dtype=np.int64)

        # window mask
        m = (times > t_start) & (times <= t_end)
        senders_w = senders[m]

        spike_counts_ex = np.bincount(senders_w - ex_ids[0], minlength=len(ex_ids))
        return spike_counts_ex[N:]

    def get_spike_vector(self, N=200):
        """
        Returns a vector of spike counts for each excitatory neuron recorded in espikes,
        EXCLUDING the first N excitatory neurons. (Inhibitory neurons are ignored.)
        """
        # Get excitatory neuron IDs
        ex_ids = np.asarray(self.nodes_ex, dtype=np.int64)
        
        # Get spike sender arrays for excitatory neurons only
        events_ex = self.espikes.events
        senders_ex = np.asarray(events_ex["senders"], dtype=np.int64)
        
        # Count spikes for each excitatory neuron using np.bincount
        spike_counts_ex = np.bincount(senders_ex - ex_ids[0], minlength=len(ex_ids))
        
        # Exclude the first N excitatory neurons
        spike_counts_ex_trunc = spike_counts_ex[N:]
        
        return spike_counts_ex_trunc



        
    def get_firing_rates_window(self, t_start, t_end):
        dt_s = (t_end - t_start) / 1000.0

        # Excitatory spikes in window
        ev_ex = self.espikes.events
        times_ex = np.asarray(ev_ex["times"], dtype=float)
        senders_ex = np.asarray(ev_ex["senders"], dtype=np.int64)

        m_ex = (times_ex > t_start) & (times_ex <= t_end)
        n_spikes_ex = m_ex.sum()
        fr_ex = n_spikes_ex / dt_s / self.NE

        # Inhibitory spikes in window
        ev_in = self.ispikes.events
        times_in = np.asarray(ev_in["times"], dtype=float)
        m_in = (times_in > t_start) & (times_in <= t_end)
        n_spikes_in = m_in.sum()
        fr_in = n_spikes_in / dt_s / self.NI



        return fr_ex, fr_in


    def get_stdp_weights(self, bins=100, show_top_bottom=False, plot=True, return_weights=False, folder_name="weights"):
        """
        Plot histogram of weights for synapses that use the 'excitatory_stdp' model.
        Also print the top 10 synapses with highest and lowest weights after simulation.
        Call this AFTER simulate().
        """
        all_neurons = self.nodes_ex + self.nodes_in

        # Get all STDP synapses (from E population to all neurons) that use this synapse model
        conns_E = nest.GetConnections(self.nodes_ex, all_neurons, synapse_model="excitatory_stdp")
        w_E = np.asarray(conns_E.get("weight"), dtype=float)

        # Calculate mean and std
        w_mean = np.mean(w_E)
        w_std = np.std(w_E)
        
        # --- Print top 10 highest and lowest weights with their connection details
        # Get source and target GIDs
        sources = np.asarray(conns_E.get('source'), dtype=int)
        targets = np.asarray(conns_E.get('target'), dtype=int)

        if show_top_bottom:
            # Compose a list of (weight, source, target)
            connections = list(zip(w_E, sources, targets))
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
            plt.figure(figsize=(8, 5))
            plt.hist(w_E, bins=bins)
            plt.xlabel("Weight (mV)")
            plt.ylabel("Count")
            plt.title("Histogram of STDP synaptic weights (after simulation)\n"
                    f"Mean: {w_mean:.3f} mV, Std: {w_std:.3f} mV")
            # Draw a vertical dotted line at the mean
            plt.axvline(w_mean, color='r', linestyle=':', linewidth=2, label=f"Mean ({w_mean:.3f} mV)")
            # Annotate mean and std in the top right corner
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"data/{folder_name}/stdp_weights_{time.time()}.png")
            #plt.show()

        if return_weights:
            return w_E

    


    
    def plot_raster(self):
        """Plot a raster plot of spikes from all excitatory and inhibitory neurons."""
        # Get spike events from both recorders
        events_ex = self.espikes.events
        events_in = self.ispikes.events
        
        
        plt.figure(figsize=(10, 6))
        
        # Plot excitatory spikes in blue
        if len(events_ex["times"]) > 0:
            plt.scatter(events_ex["times"], events_ex["senders"], s=1, color='C0', alpha=0.6, label='Excitatory')
        
        # Plot inhibitory spikes in orange/red
        if len(events_in["times"]) > 0:
            plt.scatter(events_in["times"], events_in["senders"], s=1, color='C1', alpha=0.6, label='Inhibitory')
        
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


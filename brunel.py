import nest
import numpy as np
import matplotlib.pyplot as plt

class Brunel:
    def __init__(self):
        nest.ResetKernel()  # Reset the NEST kernel
        nest.resolution = 0.1 # Simulation resolution (ms)
        self.simtime = 1000.0  # Simulation time (ms)
        self.delay = 1.5  # Synaptic delay (ms)
        self.g = 5.0  # Relative inhibitory strength
        self.eta = 1.0  # External rate in units of threshold
        self.epsilon = 0.1  # Connection probability
        self.N_neurons = 1000  # Total number of neurons
        self.NI = self.N_neurons // 5  # Number of inhibitory neurons
        self.NE = self.N_neurons - self.NI  # Number of excitatory neurons (four times as many as inhibitory)
        self.N_rec = 50  # Number of recorded excitatory neurons

        # Connectivity parameters
        self.CE = int(self.epsilon * self.NE)  # number of excitatory synapses per neuron
        self.CI = int(self.epsilon * self.NI)  # number of inhibitory synapses per neuron
        self.C_tot = int(self.CI + self.CE)  # total number of synapses per neuron

        # LIF neuron parameters
        self.tauMem = 20.0  # time constant of membrane potential in ms
        self.theta = 20.0  # membrane threshold potential in mV
        self.neuron_params = {"C_m": 1.0, "tau_m": self.tauMem, "t_ref": 2.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": self.theta}
        self.J = 0.1  # postsynaptic amplitude in mV
        self.J_ex = self.J  # amplitude of excitatory postsynaptic potential
        self.J_in = -self.g * self.J_ex  # amplitude of inhibitory postsynaptic potential

        # Threshold rate, external firing rate and converted spikes per second
        self.nu_th = 1000.0 * self.theta / (self.J * self.CE * self.tauMem)  # Hz
        self.nu_ex = self.eta * self.nu_th # Hz
        self.p_rate = self.nu_ex * self.CE # Hz

        # Synapse parameters
        self.stdp_params = {"weight": self.J_ex, "delay": self.delay, "lambda": 0.01, "alpha": 1.0,  "Wmax": 1.0}
        self.static_params = {"weight": self.J_ex, "delay": self.delay}
        self.inhibitory_params = {"weight": self.J_in, "delay": self.delay}
        self.bernoulli_conn = {"rule": "pairwise_bernoulli", "p": self.epsilon}

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

    def build_network(self):
        # Creating nodes
        self.nodes_ex = nest.Create("iaf_psc_alpha", self.NE, params=self.neuron_params)
        self.nodes_in = nest.Create("iaf_psc_alpha", self.NI, params=self.neuron_params)
        noise = nest.Create("poisson_generator", self.N_neurons, params={"rate": self.p_rate})
        self.espikes = nest.Create("spike_recorder")
        self.ispikes = nest.Create("spike_recorder")

        # Defining synapse models
        nest.CopyModel("static_synapse", "background", self.static_params)
        nest.CopyModel("static_synapse", "inhibitory", self.inhibitory_params)
        nest.CopyModel("stdp_synapse", "excitatory_stdp", self.stdp_params)

        # Connecting nodes
        nest.Connect(noise, self.nodes_ex + self.nodes_in, conn_spec = "one_to_one", syn_spec="background")
        nest.Connect(self.nodes_ex, self.nodes_ex + self.nodes_in, conn_spec=self.bernoulli_conn, syn_spec="excitatory_stdp")
        nest.Connect(self.nodes_in, self.nodes_ex + self.nodes_in, conn_spec=self.bernoulli_conn, syn_spec="inhibitory")
        
        # Connect all neurons to spike recorders 
        nest.Connect(self.nodes_ex, self.espikes)
        nest.Connect(self.nodes_in, self.ispikes)


    def simulate(self):
        nest.Simulate(self.simtime)

    def get_stdp_weights(self, bins=100):
        """
        Plot histogram of weights for synapses that use the 'excitatory_stdp' model.
        Call this AFTER simulate().
        """
        all_neurons = self.nodes_ex + self.nodes_in

        # Get all STDP synapses (from E population to all neurons) that use this synapse model
        conns_E = nest.GetConnections(self.nodes_ex, all_neurons, synapse_model="excitatory_stdp")
        w_E = np.asarray(conns_E.get("weight"), dtype=float)

        plt.figure(figsize=(8, 5))
        plt.hist(w_E, bins=bins)
        plt.xlabel("Weight (mV)")
        plt.ylabel("Count")
        plt.title("Histogram of STDP synaptic weights (after simulation)")
        plt.tight_layout()
        plt.show()

    
    def plot_raster(self):
        """Plot a raster plot of spikes from all excitatory and inhibitory neurons."""
        # Get spike events from both recorders
        events_ex = self.espikes.events
        events_in = self.ispikes.events
        print(events_ex)
        
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
        plt.show()




if __name__ == "__main__":
    brunel = Brunel()
    brunel.build_network()
    brunel.simulate()
    brunel.get_stdp_weights()
    #brunel.plot_raster()
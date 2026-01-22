import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.learning import PostPre

class Brunel:
    def __init__(self, n_neurons, time, dt, mnist_input=True):
        self.time = int(time)                                           # Simulation time per sample [ms]
        self.dt = float(dt)                                             # Time step [ms]     

        self.n_neurons = int(n_neurons)                                 # Total number of neurons
        self.N_E = int(0.8 * self.n_neurons)                            # Number of excitatory neurons
        self.N_I = self.n_neurons - self.N_E                            # Number of inhibitory neurons
        self.N_noise = self.n_neurons                        # Number of noise neurons
        # Connectivity/synapse parameters
        self.epsilon = 0.1                                              # Connection probability [ ]
        self.g = 5.0                                                  # Relative inhibitory strength [ ]
        self.eta = 0.6
        self.C_noise = int(self.N_noise * self.epsilon)

        self.w_E = 2.0                                                  # (Excitatory ->) synapse weight [ ]
        self.w_noise = 0.5                                              # (Noise ->) synapse weight [ ]
        if mnist_input:
            self.w_input = 40.0
        else:
            self.w_input = 0.0

        self.J = 0.1                                                    # Voltage amplitude jump [mV]
        self.J_E = self.w_E * self.J                                    # Excitatory voltage amplitude jump [mV]
        self.J_I = -self.g * self.J_E                                   # Inhinbitory voltage amplitude jump [mV]
        self.J_input = self.w_input * self.J                            # Input voltage amplitude jump [mV]
        self.J_noise = self.w_noise * self.J                            # Noise voltage amplitude jump [mV]
        self.J_std = 0.1 * abs(self.J)
        # Single neuron paramerers
        self.theta = 20.0                                               # Membrane threshold potential [mV]
        self.tau_m = 20.0 
        self.tau_s = self.tau_m / 1000.0                                           

    
        self.v_th = self.theta/(self.J_noise*self.C_noise*self.tau_s)   # Threshold rate [Hz]
        self.v_ext = self.eta * self.v_th 



        self.network = None
        self.neurons_E = None
        self.neurons_I = None
        self.mnist_in = None

        self.mon_E = None
        self.mon_I = None
        
        self.seed = 10061999                    # My birthday:)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)

    def get_configuration_info(self):
        print("======== Quick Summary of the parameters ========")

        # Simulation parameters
        print("----- Simulation Parameters -----")
        print(f"Simulation time per sample [ms]: {self.time}")
        print(f"Time step [ms]: {self.dt}")

        # Network structure
        print("\n----- Network Structure -----")
        print(f"Total number of neurons: {self.n_neurons}")
        print(f"Number of excitatory neurons: {self.N_E}")
        print(f"Number of inhibitory neurons: {self.N_I}")
        print(f"Number of noise neurons: {self.N_noise}")

        # Connectivity/synapse parameters
        print("\n----- Connectivity/Synapse Parameters -----")
        print(f"Connection probability: {self.epsilon}")
        print(f"Relative inhibitory strength (g): {self.g}")
        print(f"eta (external rate multiplier): {self.eta}")
        print(f"C_noise (noise connections): {self.C_noise}")
        print(f"(Excitatory ->) synapse weight: {self.w_E}")
        print(f"(Input ->) synapse weight: {self.w_input}")
        print(f"(Noise ->) synapse weight: {self.w_noise}")

        print(f"Voltage amplitude jump (J): {self.J}")
        print(f"Excitatory voltage amplitude jump (J_E): {self.J_E}")
        print(f"Inhibitory voltage amplitude jump (J_I): {self.J_I}")
        print(f"Input voltage amplitude jump (J_input): {self.J_input}")
        print(f"Noise voltage amplitude jump (J_noise): {self.J_noise}")
        print(f"Standard deviation of jump (J_std): {self.J_std}")

        # Neuron parameters
        print("\n----- Neuron Parameters -----")
        print(f"Membrane threshold potential (theta) [mV]: {self.theta}")
        print(f"Membrane time constant (tau_m): {self.tau_m}")
        print(f"Synaptic time constant (tau_s): {self.tau_s}")

        # Rates
        print("\n----- Rates -----")
        print(f"Threshold rate (v_th): {self.v_th}")
        print(f"External rate (v_ext): {self.v_ext}")

        # Miscellaneous
        print("\n----- Seed -----")
        print(f"Seed: {self.seed}")

        print("--------------------------------")
        
        print("--------------------------------")

        

    def build_brunel(self):
        print("Building network...")

        self.network = Network(dt=self.dt)

        # Excitatory and Inhibitory Neurons
        self.neurons_E = LIFNodes(n=self.N_E, tau=self.tau_m, rest=0.0, reset=0.0, thresh=self.theta, refrac=1, traces=True, tc_trace=20.0)
        self.neurons_I = LIFNodes(n=self.N_I, tau=self.tau_m, rest=0.0, reset=0.0, thresh=self.theta, refrac=1, traces=True, tc_trace=20.0)
        self.network.add_layer(self.neurons_E, name="E")
        self.network.add_layer(self.neurons_I, name="I")

        # Noise Inputs
        self.noise = Input(n=self.N_noise)
        self.network.add_layer(self.noise, name="noise")
        p_noise_E, p_noise_I = self.C_noise / self.N_noise, self.C_noise / self.N_noise


        # Input (MNIST) to Excitatory Neurons
        input_indices = torch.randint(0, 784, (self.N_E,)) # Which input neurons feeds excitatory neuron j
        self.W = torch.zeros(784, self.N_E)
        for j in range(self.N_E):
            i = input_indices[j]
            self.W[i, j] = float(self.J_input)

        self.mnist_in = Input(n=784, traces=True, tc_trace=20.0)
        self.network.add_layer(self.mnist_in, name="MNIST")
        #self.W = float(self.J_input) * torch.rand(784, self.N_E)/ np.sqrt(784)
        connection_mnist_E = Connection(source=self.mnist_in, target=self.neurons_E, w=self.W)
        self.network.add_connection(connection_mnist_E, source="MNIST", target="E")

        # each excitatory neuron has exactly one input
        #assert torch.all((self.W != 0).sum(dim=0) == 1) # Sanity check

        # input neurons may have multiple outputs
        


        # Bernoulli masks for sparse connectivity
        mask_EE = torch.bernoulli(torch.full((self.N_E, self.N_E), self.epsilon))
        mask_EI = torch.bernoulli(torch.full((self.N_E, self.N_I), self.epsilon))
        mask_IE = torch.bernoulli(torch.full((self.N_I, self.N_E), self.epsilon))
        mask_II = torch.bernoulli(torch.full((self.N_I, self.N_I), self.epsilon))
        mask_noise_E = torch.bernoulli(torch.full((self.N_noise, self.N_E), p_noise_E))
        mask_noise_I = torch.bernoulli(torch.full((self.N_noise, self.N_I), p_noise_I))

        # Weights
        W_EE = mask_EE * torch.normal(self.J_E, self.J_std, size=(self.N_E, self.N_E)).clamp(min=0.0)   
        W_EI = mask_EI * torch.normal(self.J_E, self.J_std, size=(self.N_E, self.N_I)).clamp(min=0.0)   
        W_IE = mask_IE * torch.normal(self.J_I, self.J_std, size=(self.N_I, self.N_E)).clamp(max=0.0)
        W_II = mask_II * torch.normal(self.J_I, self.J_std, size=(self.N_I, self.N_I)).clamp(max=0.0)
        W_noise_E = mask_noise_E * self.J_noise
        W_noise_I = mask_noise_I * self.J_noise

        self.network.add_connection(Connection(self.neurons_E, self.neurons_E, w=W_EE), source="E", target="E")
        self.network.add_connection(Connection(self.neurons_E, self.neurons_I, w=W_EI), source="E", target="I")
        self.network.add_connection(Connection(self.neurons_I, self.neurons_E, w=W_IE), source="I", target="E")
        self.network.add_connection(Connection(self.neurons_I, self.neurons_I, w=W_II), source="I", target="I")
        self.network.add_connection(Connection(self.noise, self.neurons_E, w=W_noise_E), source="noise", target="E")
        self.network.add_connection(Connection(self.noise, self.neurons_I, w=W_noise_I), source="noise", target="I")

        # Monitors
        T = int(self.time / self.dt)
        self.mon_E = Monitor(self.neurons_E, state_vars=["s"], time=T)
        self.mon_I = Monitor(self.neurons_I, state_vars=["s"], time=T)
        self.network.add_monitor(self.mon_E, name="E_spikes")
        self.network.add_monitor(self.mon_I, name="I_spikes")

        print("Network built successfully")
        return self

    def run_one_sample(self, dataset, target):


        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample["encoded_image"]
            label = sample["label"]
            if label == target:
                break

        E_spike_counts, I_spike_counts, E_spikes, I_spikes = self.run(image)


        self.plot_raster(E_spikes, I_spikes, "Excitatory raster", "Inhibitory raster")
        self.plot_rate_distribution(E_spike_counts, I_spike_counts, "Histogram of Excitatory Neuron Firing Rates", "Histogram of Inhibitory Neuron Firing Rates")
        self.plot_spike_distribution(E_spike_counts, I_spike_counts, "Distribution of Excitatory and Inhibitory Neuron Spikes")
     
        
        
        


    def stimulate_brunel(self, dataset, examples=500, shuffle=True):

        # create index list of the samples to train on
        n_total = len(dataset)
        n_iters = min(examples, n_total)
        indices = torch.randperm(n_total)[:n_iters].tolist() if shuffle else list(range(n_iters))

        pbar = tqdm(indices, desc=f"Train progress: (0 / {n_iters})")
        pairs = []
        for i, index in enumerate(pbar):
            sample = dataset[index]
            image = sample["encoded_image"]
            label = sample["label"]
            pbar.set_description_str(f"Train progress: ({i+1} / {n_iters})")
            features_E, features_I, E_spikes, I_spikes = self.run(image)
            binned_E = self._spikes_to_binned_counts(E_spikes, bin_ms=50)
            binned_E_flat = binned_E.flatten()
            features = binned_E_flat.float()
            #features = torch.cat([features_E.float(), features_I.float()])

            CV_E = self._calculate_CV(E_spikes)
            CV_I = self._calculate_CV(I_spikes)
            print(f"CV_E: {CV_E}, CV_I: {CV_I}")
           

            pairs.append((features, label))
        
        return pairs
            
    # Helper methods:
    def run(self, image):
        
        T = image.shape[0] # number of time steps

        mnist_spikes = image.view(T, 1, 784).to("cpu")

        # External Poisson drive for full window
        p = self.v_ext * self.dt / 1000.0
        spikes = (torch.rand(T, self.N_noise) < p).float()
        self.network.run(inputs={"noise": spikes, "MNIST": mnist_spikes}, time=self.time)

        # Get spikes from monitors
        E_spikes = self.mon_E.get("s") # shape (T, 1, N_E)
        I_spikes = self.mon_I.get("s") # shape (T, 1, N_I)

        E_spike_counts = E_spikes.squeeze(1).sum(0) # shape (N_E,)
        I_spike_counts = I_spikes.squeeze(1).sum(0) # shape (N_I,)

        # Reset state variables
        self.network.reset_state_variables()
        self.mon_E.reset_state_variables()
        self.mon_I.reset_state_variables()


        return E_spike_counts, I_spike_counts, E_spikes, I_spikes

    def _spikes_to_binned_counts(self, E_spikes, bin_ms = 50):
        s = E_spikes.squeeze(1) if E_spikes.dim() == 3 else E_spikes
        s = np.array(s)
        s = s.astype(int)

        bin_steps = int(round(bin_ms / self.dt))
        N_bins = self.time // bin_steps

        T, N = s.shape
        trim_T = N_bins * bin_steps
        s = s[:trim_T]  # Now shape (N_bins * bin_steps, N)
        s_binned = s.reshape(N_bins, bin_steps, N)
        binned_counts = s_binned.sum(axis=1)

        return torch.tensor(binned_counts, dtype=torch.float32)

    def _calculate_CV(self, spikes):
        # spikes shape: (T, 1, N) -> squeeze to (T, N)
        spikes = spikes.squeeze(1).cpu()
        T, N = spikes.shape
        
        cv_list = []
        
        for i in range(N):
            # Find indices where spike == 1
            spike_times = torch.where(spikes[:, i] == 1)[0].float()
            
            if len(spike_times) < 3:
                # Need at least 3 spikes to get 2 intervals for a standard deviation
                continue
                
            # Calculate ISIs in milliseconds
            intervals = (spike_times[1:] - spike_times[:-1]) * self.dt
            
            # Brunel CV calculation
            cv_i = torch.std(intervals) / torch.mean(intervals)
            cv_list.append(cv_i.item())
        
        return np.mean(cv_list) if cv_list else 0.0

        

        
                           
    
    def plot_raster(self, E_spikes, I_spikes, title_excitatory, title_inhibitory):
        sE = E_spikes.squeeze(1)
        sI = I_spikes.squeeze(1)

        tE, nE = torch.where(sE > 0)
        tI, nI = torch.where(sI > 0)

        fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        ax[0].scatter(tE.cpu(), nE.cpu(), color="slategrey", marker=".", s=10, edgecolors="none", alpha=0.8)
        ax[0].set_title(title_excitatory)
        ax[0].set_ylabel("Neuron idx")
        ax[1].scatter(tI.cpu(), nI.cpu(), color="slategrey", marker=".", s=10, edgecolors="none", alpha=0.8)
        ax[1].set_title(title_inhibitory)
        ax[1].set_xlabel("Time (timestep)")
        ax[1].set_ylabel("Neuron idx")
        ax[0].grid(True, linestyle="--", alpha=0.6)
        ax[1].grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show(block=True)
        plt.close()
    
    def plot_rate_distribution(self, E_spike_counts, I_spike_counts, title_excitatory, title_inhibitory):
        neuron_rates_E = E_spike_counts / (self.time / 1000.0) #Hz to spikes/sec
        avr_rate_E = neuron_rates_E.mean()
        neuron_rates_I = I_spike_counts / (self.time / 1000.0) #Hz to spikes/sec
        avr_rate_I = neuron_rates_I.mean()

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].hist(neuron_rates_E.cpu().numpy(), bins=100, color='skyblue', alpha=0.7)
        ax[0].set_title(title_excitatory)
        ax[0].axvline(avr_rate_E, color='salmon', linestyle='--', linewidth=2, label=f"Average rate: {avr_rate_E:.2f} Hz")
        ax[0].legend()
        ax[1].hist(neuron_rates_I.cpu().numpy(), bins=100, color='salmon', alpha=0.7)
        ax[1].set_title(title_inhibitory)
        ax[1].axvline(avr_rate_I, color='skyblue', linestyle='--', linewidth=2, label=f"Average rate: {avr_rate_I:.2f} Hz")
        ax[1].legend()
        plt.xlabel('Firing rate (spikes/ms)')
        plt.ylabel('Number of neurons')
        plt.tight_layout()
        plt.show(block=True)
        plt.close()
    
    def plot_spike_distribution(self, E_spike_counts, I_spike_counts, title):
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)


        E_flat = E_spike_counts.flatten()
        I_flat = I_spike_counts.flatten()

        for i, val in enumerate(E_flat):
            num_spikes = int(val)
            ax.vlines(i, 0, num_spikes, color='skyblue', alpha= 0.8)
        for i, val in enumerate(I_flat):
            i = len(E_flat) + i
            num_spikes = int(val)
            ax.vlines(i, 0, num_spikes, color='salmon', alpha= 0.8)

        ax.set_title(title)
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Number of spikes")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        plt.show(block=True)
        plt.close()

        


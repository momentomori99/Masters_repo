import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import math

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.learning import PostPre
from bindsnet.encoding import PoissonEncoder

class Brunel:
    def __init__(self, n_neurons, time, dt, mnist_input=True, self_tuning=True, g=4, eta=1.0):
        self.time = int(time)                                           # Simulation time per sample [ms]
        self.dt = float(dt)                                             # Time step [ms]     

        self.n_neurons = int(n_neurons)                                 # Total number of neurons
        self.N_E = int(0.8 * self.n_neurons)                            # Number of excitatory neurons
        self.N_I = self.n_neurons - self.N_E                            # Number of inhibitory neurons

        # Connectivity/synapse parameters
        self.epsilon = 0.01                                              # Connection probability [ ]
        self.g = g                                                  # Relative inhibitory strength [ ]
        self.eta = eta
        self.self_tuning = self_tuning

        self.w_E = 1.0                                                  # (Excitatory ->) synapse weight [ ]
        self.w_ext = 1.0                                              # (Noise ->) synapse weight [ ]
        if mnist_input:
            self.w_input = 40.0
        else:
            self.w_input = 0.0

        self.J = 0.001                                                    # Voltage amplitude jump [mV]
        #self.J_E = self.w_E * self.J                                    # Excitatory voltage amplitude jump [mV]
        #self.J_I = -self.g * self.J_E                                   # Inhinbitory voltage amplitude jump [mV]
        self.J_input = self.w_input * self.J                            # Input voltage amplitude jump [mV]
        #self.J_noise = self.w_noise * self.J                            # Noise voltage amplitude jump [mV]
        #self.J_std = 0.1 * abs(self.J)
        # Single neuron paramerers

        self.mean_w_EE = self.w_E 
        self.mean_w_EI = self.w_E
        self.mean_w_IE = -self.g * self.w_E
        self.mean_w_II = -self.g * self.w_E
        self.std_w_EE = 0.1
        self.std_w_EI = 0.1
        self.std_w_IE = 0.1
        self.std_w_II = 0.1

        self.theta = 20.0                                               # Membrane threshold potential [mV]
        self.tau_m = 20.0 
        self.tau_s = self.tau_m / 1000.0                                           

    
        #self.v_th = self.theta/(self.J_noise*self.C_noise*self.tau_s)   # Threshold rate [Hz]
        #self.v_ext = self.eta * self.v_th 

        self.v_th = self.theta / (self.tau_s * self.w_ext) 
        self.rate_ext = self.eta * self.v_th
        print(f"v_th: {self.v_th}, rate_ext: {self.rate_ext}")
        #self.rate_ext = 250.0 * self.eta # Hz per afferent



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
        self.neurons_E = LIFNodes(n=self.N_E, tau=self.tau_m, rest=0.0, reset=0.0, thresh=self.theta, refrac=1)
        self.neurons_I = LIFNodes(n=self.N_I, tau=self.tau_m, rest=0.0, reset=0.0, thresh=self.theta, refrac=1)
        self.network.add_layer(self.neurons_E, name="E")
        self.network.add_layer(self.neurons_I, name="I")

        # Noise Inputs
        self.noise_E = Input(n=self.N_E)
        self.noise_I = Input(n=self.N_I)
        self.network.add_layer(self.noise_E, name="noise_E")
        self.network.add_layer(self.noise_I, name="noise_I")

        connection_noise_E = Connection(source=self.noise_E, target=self.neurons_E, w=self.w_ext * torch.eye(self.N_E))
        connection_noise_I = Connection(source=self.noise_I, target=self.neurons_I, w=self.w_ext * torch.eye(self.N_I))



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

        #pos_E, rows_E, cols_E = self.make_grid_positions(self.N_E, device="cpu")
        #pos_I, rows_I, cols_I = self.make_grid_positions(self.N_I, device="cpu")
        pos_all, rows, cols = self.make_grid_positions(self.n_neurons)
        pos_E = pos_all[:self.N_E]
        pos_I = pos_all[self.N_E: self.N_E + self.N_I]


        sigma_EE = 2.5
        sigma_EI = 2.5
        sigma_IE = 2.5
        sigma_II = 2.5

        #mask_EE, P_EE = self.distance_mask_2d(pos_E, pos_E, self.epsilon, sigma_EE, device="cpu")
        #mask_EI, P_EI = self.distance_mask_2d(pos_E, pos_I, self.epsilon, sigma_EI, device="cpu")
        #mask_IE, P_IE = self.distance_mask_2d(pos_I, pos_E, self.epsilon, sigma_IE, device="cpu")
        #mask_II, P_II = self.distance_mask_2d(pos_I, pos_I, self.epsilon, sigma_II, device="cpu")
        mask_EE, P_EE = self.distance_mask_2d_toroidal(pos_E, pos_E, self.epsilon, sigma_EE, rows, cols, device="cpu")
        mask_EI, P_EI = self.distance_mask_2d_toroidal(pos_E, pos_I, self.epsilon, sigma_EI, rows, cols, device="cpu")
        mask_IE, P_IE = self.distance_mask_2d_toroidal(pos_I, pos_E, self.epsilon, sigma_IE, rows, cols, device="cpu")
        mask_II, P_II = self.distance_mask_2d_toroidal(pos_I, pos_I, self.epsilon, sigma_II, rows, cols, device="cpu")

        # Weights
        W_EE = mask_EE * torch.normal(self.mean_w_EE, self.std_w_EE, size=(self.N_E, self.N_E))
        W_EI = mask_EI * torch.normal(self.mean_w_EI, self.std_w_EI, size=(self.N_E, self.N_I))
        W_IE = mask_IE * torch.normal(self.mean_w_IE, self.std_w_IE, size=(self.N_I, self.N_E))
        W_II = mask_II * torch.normal(self.mean_w_II, self.std_w_II, size=(self.N_I, self.N_I))

        connection_EE = Connection(source=self.neurons_E, target=self.neurons_E, w=W_EE)
        connection_EI = Connection(source=self.neurons_E, target=self.neurons_I, w=W_EI)
        self.connection_IE = Connection(source=self.neurons_I, target=self.neurons_E, w=W_IE)
        self.connection_II = Connection(source=self.neurons_I, target=self.neurons_I, w=W_II)

        self.W_IE_base = self.connection_IE.w.clone()
        self.W_II_base = self.connection_II.w.clone()
        self.g_base = self.g

        self.network.add_connection(connection_noise_E, source="noise_E", target="E")
        self.network.add_connection(connection_noise_I, source="noise_I", target="I")
        self.network.add_connection(connection_EE, source="E", target="E")
        self.network.add_connection(connection_EI, source="E", target="I")
        self.network.add_connection(self.connection_IE, source="I", target="E")
        self.network.add_connection(self.connection_II, source="I", target="I")


        # Monitors
        T = int(self.time / self.dt)
        self.mon_E = Monitor(self.neurons_E, state_vars=["s"], time=T)
        self.mon_I = Monitor(self.neurons_I, state_vars=["s"], time=T)
        self.network.add_monitor(self.mon_E, name="E_spikes")
        self.network.add_monitor(self.mon_I, name="I_spikes")

        print("Network built successfully")




        # positions (store for later viz)
        self.pos_E, self.rows_E, self.cols_E = pos_E, rows, cols
        self.pos_I, self.rows_I, self.cols_I = pos_I, rows, cols

        # masks + probability matrices (store for later viz)
        self.mask_EE, self.P_EE = mask_EE, P_EE
        self.mask_EI, self.P_EI = mask_EI, P_EI
        self.mask_IE, self.P_IE = mask_IE, P_IE
        self.mask_II, self.P_II = mask_II, P_II

        # store sigmas too if you want them in plot titles
        self.sigma_EE = sigma_EE
        self.sigma_EI = sigma_EI
        self.sigma_IE = sigma_IE
        self.sigma_II = sigma_II

        return self

    def run_one_sample(self, dataset, target):


        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample["encoded_image"]
            label = sample["label"]
            if label == target:
                break

        E_spike_counts, I_spike_counts, E_spikes, I_spikes = self.run(image, self.rate_ext)

        CV_E = self._calculate_CV(E_spikes)
        rho_mean_E = self._calculate_rho_mean(E_spikes, bin_ms=10.0)
        neuron_rates_E = E_spike_counts / (self.time / 1000.0) #Hz to spikes/sec
        rate_E = neuron_rates_E.mean()
        print(f"CV_E: {CV_E}, rho_mean_E: {rho_mean_E}")
        print(f"g: {self.g}, eta: {self.eta}")  


        self.plot_raster(E_spikes, I_spikes, "Excitatory raster", "Inhibitory raster")
        self.plot_rate_distribution(E_spike_counts, I_spike_counts, "Histogram of Excitatory Neuron Firing Rates", "Histogram of Inhibitory Neuron Firing Rates")
        self.plot_spike_distribution(E_spike_counts, I_spike_counts, "Distribution of Excitatory and Inhibitory Neuron Spikes")
        self.plot_spikecount_grid_E(E_spike_counts, title="E spike counts (2D grid)")
        
        
     
        
        
        


    def stimulate_brunel(self, dataset, examples=500, shuffle=True):

        CV_list, rho_mean_list, rate_list, g_list, eta_list = [], [], [], [], []

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
            features_E, features_I, E_spikes, I_spikes = self.run(image, self.rate_ext)
            binned_E = self._spikes_to_binned_counts(E_spikes, bin_ms=50)
            binned_E_flat = binned_E.flatten()
            features = binned_E_flat.float()
            #features = torch.cat([features_E.float(), features_I.float()])

            CV_E = self._calculate_CV(E_spikes)
            CV_list.append(CV_E)
            CV_I = self._calculate_CV(I_spikes)
            rho_mean_E = self._calculate_rho_mean(E_spikes, bin_ms=10.0)
            rho_mean_list.append(rho_mean_E)
            rho_mean_I = self._calculate_rho_mean(I_spikes, bin_ms=10.0)
            neuron_rates_E = features_E / (self.time / 1000.0) #Hz to spikes/sec
            rate_E = neuron_rates_E.mean()
            rate_list.append(rate_E)
            g_list.append(self.g)
            eta_list.append(self.eta)
            print(f"CV_E: {CV_E}, rho_mean_E: {rho_mean_E}")
            print(f"g: {self.g}, eta: {self.eta}")
            #self.plot_raster(E_spikes, I_spikes, f"Excitatory raster \n CV: {CV_E:.2f}, rho_mean: {rho_mean_E:.4f}, rate: {rate_E:.2f}", f"Inhibitory raster \n CV: {CV_I:.2f}, rho_mean: {rho_mean_I:.4f}")
            self.plot_spikecount_grid_E(features_E, title="E spike counts (2D grid)")
            self.plot_spike_distribution(features_E, features_I, "Distribution of Excitatory and Inhibitory Neuron Spikes")

            pairs.append((features, label))

            # Logic for self tuning network towards AI regime.
            if self.self_tuning:
                self._self_tune(CV_E, rho_mean_E, rate_E)
            
        
        return pairs, CV_list, rho_mean_list, rate_list, g_list, eta_list
            
    # Helper methods:
    def run(self, image, noise_rate=150):

        mnist_spikes = image.view(self.time, 1, 784).to("cpu") # (T, 1, 784)

        encoder = PoissonEncoder(time=1)
        for t in range(self.time):
            rates_XE = torch.ones(self.N_E) * self.rate_ext
            rates_XI = torch.ones(self.N_I) * self.rate_ext
            spikes_XE = encoder(rates_XE)
            spikes_XI = encoder(rates_XI)

            mnist_t = mnist_spikes[t:t+1] # (1, 1, 784)
            self.network.run(inputs={"noise_E": spikes_XE.unsqueeze(0), "noise_I": spikes_XI.unsqueeze(0), "MNIST": mnist_t}, time=1)

        E_spikes = self.mon_E.get("s") # shape (T, 1, N_E)
        I_spikes = self.mon_I.get("s") # shape (T, 1, N_I)
        E_spike_counts = E_spikes.squeeze(1).sum(0) # shape (N_E,)
        I_spike_counts = I_spikes.squeeze(1).sum(0) # shape (N_I,)
        
        # T = image.shape[0] # number of time steps

        # mnist_spikes = image.view(T, 1, 784).to("cpu")

        # # External Poisson drive for full window
        # p = self.v_ext * self.dt / 1000.0
        # spikes = (torch.rand(T, self.N_noise) < p).float()
        # self.network.run(inputs={"noise": spikes, "MNIST": mnist_spikes}, time=self.time)

        # # Get spikes from monitors
        # E_spikes = self.mon_E.get("s") # shape (T, 1, N_E)
        # I_spikes = self.mon_I.get("s") # shape (T, 1, N_I)

        # E_spike_counts = E_spikes.squeeze(1).sum(0) # shape (N_E,)
        # I_spike_counts = I_spikes.squeeze(1).sum(0) # shape (N_I,)

        # Reset state variables
        self.network.reset_state_variables()
        self.mon_E.reset_state_variables()
        self.mon_I.reset_state_variables()


        return E_spike_counts, I_spike_counts, E_spikes, I_spikes




        self.network.reset_state_variables()
        self.mon_E.reset_state_variables()
        self.mon_I.reset_state_variables()

    def set_g(self, new_g):
        scale = float(new_g / self.g_base)

        with torch.no_grad():
            self.connection_IE.w.copy_(self.W_IE_base * scale)
            self.connection_II.w.copy_(self.W_II_base * scale)
        self.g = new_g

    
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

    def _calculate_rho_mean(self, spikes: torch.Tensor, bin_ms: float = 5.0) -> float:
        # spikes -> (T, N)
        spikes = spikes.squeeze(1) if spikes.dim() == 3 else spikes
        spikes = spikes.detach().float().cpu()

        T, N = spikes.shape
        if N < 2 or T < 2:
            return 0.0

        # bin in time
        bin_steps = max(1, int(round(bin_ms / self.dt)))
        n_bins = T // bin_steps
        if n_bins < 2:
            return 0.0

        x = spikes[: n_bins * bin_steps].reshape(n_bins, bin_steps, N).sum(dim=1)  # (n_bins, N)

        # population activity
        r_t = x.mean(dim=1)  # (n_bins,)

        var_r = torch.var(r_t, unbiased=False)
        var_i = torch.var(x, dim=0, unbiased=False)
        mean_var_i = torch.mean(var_i)

        if mean_var_i.item() <= 1e-12:
            return 0.0

        return float((var_r / mean_var_i).item())

    def _self_tune(self, CV_value, rho_mean_value, rate):

        CV_low, CV_high = 0.8, 1.2
        rho_high = 0.015

        rate_low = 2.0
        rate_mid = 20.0
        rate_high = 80.0

        eta_min, eta_max = 0.5, 15.0
        g_min, g_max = 1.0, 15.0

        if rho_mean_value > rho_high:
            
            # update g with bounds
            new_g = min(max(self.g + 0.1, g_min), g_max)
            self.set_g(new_g)

            if rate > rate_low:
                # update eta with bounds
                new_eta = min(max(self.eta - 0.02, eta_min), eta_max)
                self.eta = new_eta
                self.rate_ext = self.v_th * self.eta
            return 

        if CV_value < CV_low:
            if rate > rate_high:
                # too active + too regular
                new_eta = min(max(self.eta - 0.02, eta_min), eta_max)
                self.eta = new_eta
                self.rate_ext = self.v_th * self.eta

            elif rate < rate_low:
                # too quiet
                new_eta = min(max(self.eta + 0.02, eta_min), eta_max)
                self.eta = new_eta
                self.rate_ext = self.v_th * self.eta

                new_g = min(max(self.g - 0.05, g_min), g_max)
                self.set_g(new_g)

            else:
                new_eta = min(max(self.eta - 0.02, eta_min), eta_max)
                self.eta = new_eta
                self.rate_ext = self.v_th * self.eta
            return

        return







                           
    
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
        #plt.savefig(f"BindsNet/results/self_tuning/test1/raster_plot_{time.time()}.png")
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



    # ==================== Introducing spatial dynamics ===============================

    def make_grid_positions(self, N: int, rows: int | None = None, cols: int | None = None, device="cpu"):
        if rows is None or cols is None:
            cols = int(math.ceil(math.sqrt(N)))
            rows = int(math.ceil(N / cols))

        # (rows*cols, 2) coordinates, then keep first N
        rr, cc = torch.meshgrid(
            torch.arange(rows, device=device),
            torch.arange(cols, device=device),
            indexing="ij"
        )
        pos = torch.stack([rr.flatten(), cc.flatten()], dim=1).float()  # (rows*cols, 2)
        pos = pos[:N]  # (N, 2)

    
        return pos, rows, cols

    
    def distance_mask_2d(self, pos_pre: torch.Tensor,
                     pos_post: torch.Tensor,
                     epsilon: float,
                     sigma: float,
                     device="cpu"):
        """
        pos_pre:  (N_pre, 2)
        pos_post: (N_post, 2)
        Returns a Bernoulli mask of shape (N_pre, N_post)
        using Gaussian distance-based probabilities, scaled to ~epsilon mean.
        """
        # pairwise squared distances: (N_pre, N_post)
        diff = pos_pre[:, None, :] - pos_post[None, :, :]
        d2 = (diff ** 2).sum(dim=2)

        # unscaled kernel
        P = torch.exp(-d2 / (2.0 * sigma * sigma))  # (N_pre, N_post)

        # if square, remove self connections
        if pos_pre.shape[0] == pos_post.shape[0]:
            P.fill_diagonal_(0.0)

        # scale to match desired average epsilon (approx)
        meanP = P.mean()
        if meanP.item() > 1e-12:
            P = P * (epsilon / meanP)
        P = P.clamp(0.0, 1.0)

        mask = torch.bernoulli(P).to(device)
        return mask, P
        
    def distance_mask_2d_toroidal(self, pos_pre, pos_post, epsilon, sigma, rows, cols, device="cpu"):
        # pos_* are (N, 2) with (row, col)

        drow = torch.abs(pos_pre[:, None, 0] - pos_post[None, :, 0])
        dcol = torch.abs(pos_pre[:, None, 1] - pos_post[None, :, 1])

        # wrap-around distances
        drow = torch.minimum(drow, rows - drow)
        dcol = torch.minimum(dcol, cols - dcol)

        d2 = drow**2 + dcol**2

        P = torch.exp(-d2 / (2.0 * sigma * sigma))

        if pos_pre.shape[0] == pos_post.shape[0]:
            P.fill_diagonal_(0.0)

        meanP = P.mean()
        if meanP.item() > 1e-12:
            P = P * (epsilon / meanP)

        P = P.clamp(0.0, 1.0)
        mask = torch.bernoulli(P).to(device)
        return mask, P

    def plot_positions(self,pos: torch.Tensor, title="Neuron positions"):
        p = pos.detach().cpu().numpy()
        plt.figure(figsize=(5,5))
        plt.scatter(p[:,1], p[:,0], s=8)  # x=col, y=row
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("x (col)")
        plt.ylabel("y (row)")
        plt.tight_layout()
        plt.show(block=True)
        plt.close()

    def plot_EI_positions(self):
        """
        Plot excitatory and inhibitory neuron positions
        on the same 2D toroidal grid.
        """
        pE = self.pos_E.detach().cpu().numpy()
        pI = self.pos_I.detach().cpu().numpy()

        plt.figure(figsize=(5, 5))

        # Excitatory neurons
        plt.scatter(
            pE[:, 1], pE[:, 0],
            s=8, c="tab:blue", label="Excitatory", alpha=0.8
        )

        # Inhibitory neurons
        plt.scatter(
            pI[:, 1], pI[:, 0],
            s=12, c="tab:red", label="Inhibitory", alpha=0.9
        )

        plt.gca().invert_yaxis()
        plt.xlabel("x (col)")
        plt.ylabel("y (row)")
        plt.title("E / I neuron positions on toroidal lattice")
        plt.legend(markerscale=1.5)
        plt.tight_layout()
        plt.show(block=True)
        plt.close()

    def plot_probability_matrix(self, P: torch.Tensor, title="Connection probability P_ij", max_n=400):
        M = P.detach().cpu()
        n0 = min(max_n, M.shape[0])
        n1 = min(max_n, M.shape[1])
        plt.figure(figsize=(6,5))
        plt.imshow(M[:n0, :n1].numpy(), aspect="auto")
        plt.title(title + f" (cropped {n0}x{n1})")
        plt.xlabel("post neuron index")
        plt.ylabel("pre neuron index")
        plt.colorbar(label="P_ij")
        plt.tight_layout()
        plt.show(block=True)
        plt.close()

    
    def plot_outgoing_connections(self, mask: torch.Tensor, pos_post: torch.Tensor, pre_idx: int, title=None):
        """
        mask: (N_pre, N_post) 0/1
        pos_post: (N_post, 2)
        """
        m = mask[pre_idx].detach().cpu()  # (N_post,)
        p = pos_post.detach().cpu()
        connected = (m > 0)

        plt.figure(figsize=(5,5))
        plt.scatter(p[:,1], p[:,0], s=8, alpha=0.2, label="all post")
        plt.scatter(p[connected,1], p[connected,0], s=15, alpha=0.9, label="connected")
        plt.gca().invert_yaxis()
        plt.title(title or f"Outgoing connections from pre neuron {pre_idx}")
        plt.xlabel("x (col)")
        plt.ylabel("y (row)")
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)
        plt.close()


    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    def plot_spikecount_grid(self,counts: torch.Tensor, pos: torch.Tensor, title="Spike count heatmap"):
        counts = counts.detach().cpu().float()
        pos = pos.detach().cpu().float()

        # infer grid size from positions
        rows = int(pos[:, 0].max().item()) + 1
        cols = int(pos[:, 1].max().item()) + 1

        grid = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(pos.shape[0]):
            r = int(pos[i, 0].item())
            c = int(pos[i, 1].item())
            grid[r, c] = counts[i]

        plt.figure(figsize=(6, 5))
        plt.imshow(grid.numpy(), aspect="equal")
        plt.title(title)
        plt.xlabel("x (col)")
        plt.ylabel("y (row)")
        plt.colorbar(label="spikes")
        plt.tight_layout()
        plt.show(block=True)
        plt.close()



    def plot_positions_E(self):
        self.plot_positions(self.pos_E, title="E positions")

    def plot_positions_I(self):
        self.plot_positions(self.pos_I, title="I positions")

    def plot_probability_EE(self, max_n=400):
        self.plot_probability_matrix(self.P_EE, title=f"P_EE (sigma={self.sigma_EE})", max_n=max_n)

    def plot_mask_EE(self, max_n=400):
        self.plot_probability_matrix(self.mask_EE.float(), title="mask_EE", max_n=max_n)

    def plot_outgoing_EE(self, pre_idx=0):
        self.plot_outgoing_connections(self.mask_EE, self.pos_E, pre_idx=pre_idx,
                                    title=f"EE outgoing (pre={pre_idx})")

    def plot_spikecount_grid_E(self, E_counts, title="E spike counts"):
        self.plot_spikecount_grid(E_counts, self.pos_E, title=title)






        


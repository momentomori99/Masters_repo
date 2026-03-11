import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import math
import random


from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder

from tools.spatial_tools import make_mixed_EI_lattice
from tools.spatial_tools import distance_mask_2d_toroidal
from tools.build_W_in import build_tiled_gaussian_W_in
from tools.build_W_in import build_pixel_gaussian_W_in
from tools.other import sample_per_neuron_param
from tools.feature_encoding import encode_feature_map
from tools.feature_encoding import spikes_to_binned_counts

#metrics
from tools.metrics import calculate_CV
from tools.metrics import calculate_rate

#visualization
from visualization.visualizations import plot_raster
from visualization.visualizations import plot_rate_distribution
from visualization.visualizations import plot_spike_distribution
from visualization.visualizations_spatial import plot_EI_positions
from visualization.visualizations_spatial import plot_outgoing_connections
from visualization.visualizations_spatial import plot_spikecount_grid



class Framework:
    def __init__(self, n_neurons, time, dt, seed, log_normal=False, heterogeneity=False, mnist_input=True, self_tuning=True, spatial=True, convolution=True, input_channels=4, g=4, eta=1.0, sigma_input = 1, sigma_network = 1, epsilon = 0.3, intensity=430):
        self.seed = seed
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        self.device = "cpu"
        
        # Simulation time parameters
        self.time = int(time)                                           
        self.dt = float(dt)  
        self.bin_ms = 50
        # Methods
        self.heterogeneity = heterogeneity  
        self.intensity = intensity
        self.self_tuning = self_tuning
        self.spatial = spatial
        self.log_normal = log_normal
        self.mnist_input = mnist_input
        self.convolution = convolution

        
        self.sigma_log = 0.8
        self.mu_log = math.log(1) - 0.5*self.sigma_log**2
        if self.spatial == False and self.convolution == True:
            raise ValueError("Cannot have Spatial is set to False and convolution to True.")
        if self.convolution:
            self.input_channels = 4
        else:
            self.input_channels = 1
        
        # Neuron Network parameters
        self.n_neurons = int(n_neurons) 
        self.frac_E = 0.8                               
        self.N_E = int(self.frac_E * self.n_neurons)                         
        self.N_I = self.n_neurons - self.N_E  

        # Brunel parameters
        self.g = g                                                 
        self.eta = eta   
        

        # Connectivity/synapse parameters
        self.epsilon = epsilon 
        self.sigma_input = sigma_input                                           
        self.sigma_network = sigma_network
        self.w_E = 1                                                 
        self.w_ext = 1  
        if mnist_input:
            self.w_input = 10.0
        else:
            self.w_input = 0.0

        self.mean_w_EE = self.w_E 
        self.mean_w_EI = self.w_E
        self.mean_w_IE = -self.g * self.w_E
        self.mean_w_II = -self.g * self.w_E
        self.std_w_EE = 0.05
        self.std_w_EI = 0.05
        self.std_w_IE = 0.05
        self.std_w_II = 0.05

        # Sinlge neuron parameters
        self.theta_base = 20.0
        self.tau_m_base = 20.0
        if self.heterogeneity:
            self.theta_E = sample_per_neuron_param(self.N_E, base=self.theta_base, rel_std=0.20, min_val=5.0, max_val=40.0, device=self.device)
            self.tau_m_E = sample_per_neuron_param(self.N_E, base=self.tau_m_base, rel_std=0.25, min_val=5.0, max_val=60.0, device=self.device)
            self.theta_I = sample_per_neuron_param(self.N_I, base=self.theta_base, rel_std=0.20, min_val=5.0, max_val=40.0, device=self.device)
            self.tau_m_I = sample_per_neuron_param(self.N_I, base=self.tau_m_base, rel_std=0.25, min_val=5.0, max_val=60.0, device=self.device)
        else:
            self.theta_E = self.theta_base
            self.tau_m_E = self.tau_m_base
            self.theta_I = self.theta_base
            self.tau_m_I = self.tau_m_base
        self.tau_s = self.tau_m_base / 1000.0                                         

        # Noise paramters
        self.v_th = self.theta_base / (self.tau_s * self.w_ext) 
        self.rate_ext = self.eta * self.v_th

    
    def build_network(self):
        print("Building network...")

        self.network = Network(dt=self.dt)

        self.neurons_E = LIFNodes(n=self.N_E, tau=self.tau_m_E, rest=0.0, reset=0.0, thresh=self.theta_E, refrac=1, traces=True, tc_trace = 20.0)
        self.neurons_I = LIFNodes(n=self.N_I, tau=self.tau_m_I, rest=0.0, reset=0.0, thresh=self.theta_I, refrac=1, traces=True, tc_trace = 20.0)
        self.network.add_layer(self.neurons_E, name="E")
        self.network.add_layer(self.neurons_I, name="I")

        # Noise Inputs
        self.noise_E = Input(n=self.N_E)
        self.noise_I = Input(n=self.N_I)
        self.network.add_layer(self.noise_E, name="noise_E")
        self.network.add_layer(self.noise_I, name="noise_I")

        if self.spatial:
            self.pos_all, self.pos_E, self.pos_I, self.rows, self.cols = make_mixed_EI_lattice(self.n_neurons, frac_E=self.frac_E, device=self.device)

            self.K = self.input_channels # number of feature maps (4 with gabor, 1 without)
            self.Hf = self.rows_f = 28
            self.Wf = self.cols_f = 28 
            self.D_in = self.K * self.Hf * self.Wf
            self.feat_in = Input(n=self.D_in, traces=True, tc_trace=20.0)
            self.network.add_layer(self.feat_in, name="F")
            self.W_in = build_tiled_gaussian_W_in(self.pos_E, self.N_E, self.rows, self.cols, self.K, self.Hf, self.Wf, self.sigma_input, margin = 0.5, w_input=self.w_input)
            self.connection_F_E = Connection(source=self.feat_in, target=self.neurons_E, w=self.W_in)
            self.network.add_connection(self.connection_F_E, source="F", target="E")

            self.mask_EE = distance_mask_2d_toroidal(self.pos_E, self.pos_E, self.epsilon, self.sigma_network, self.rows, self.cols, device=self.device)
            self.mask_EI = distance_mask_2d_toroidal(self.pos_E, self.pos_I, self.epsilon, self.sigma_network, self.rows, self.cols, device=self.device)
            self.mask_IE = distance_mask_2d_toroidal(self.pos_I, self.pos_E, self.epsilon, self.sigma_network, self.rows, self.cols, device=self.device)
            self.mask_II = distance_mask_2d_toroidal(self.pos_I, self.pos_I, self.epsilon, self.sigma_network, self.rows, self.cols, device=self.device)
        
        else:
            input_indices = torch.randint(0, 784, (self.N_E,)) # Which input neurons feeds excitatory neuron j
            self.W_in = torch.zeros(784, self.N_E)
            for j in range(self.N_E):
                i = input_indices[j]
                self.W_in[i, j] = float(self.w_input)
            self.mnist_in = Input(n=784, traces=True, tc_trace=20.0)
            self.network.add_layer(self.mnist_in, name="F")
            self.connection_F_E = Connection(source=self.mnist_in, target=self.neurons_E, w=self.W_in)
            self.network.add_connection(self.connection_F_E, source="F", target="E")
            self.mask_EE = torch.bernoulli(torch.full((self.N_E, self.N_E), self.epsilon))
            self.mask_EI = torch.bernoulli(torch.full((self.N_E, self.N_I), self.epsilon))
            self.mask_IE = torch.bernoulli(torch.full((self.N_I, self.N_E), self.epsilon))
            self.mask_II = torch.bernoulli(torch.full((self.N_I, self.N_I), self.epsilon))

        if self.log_normal:
            log_samples = torch.normal(self.mu_log, self.sigma_log, size=(self.N_E, self.N_E))
            weights = torch.exp(log_samples)
            self.W_EE = self.mask_EE * weights
            self.W_EI = self.mask_EI * torch.normal(self.mean_w_EI, self.std_w_EI, size=(self.N_E, self.N_I))
            self.W_IE = self.mask_IE * torch.normal(self.mean_w_IE, self.std_w_IE, size=(self.N_I, self.N_E))
            self.W_II = self.mask_II * torch.normal(self.mean_w_II, self.std_w_II, size=(self.N_I, self.N_I))
        else:
            self.W_EE = self.mask_EE * torch.normal(self.mean_w_EE, self.std_w_EE, size=(self.N_E, self.N_E))
            self.W_EI = self.mask_EI * torch.normal(self.mean_w_EI, self.std_w_EI, size=(self.N_E, self.N_I))
            self.W_IE = self.mask_IE * torch.normal(self.mean_w_IE, self.std_w_IE, size=(self.N_I, self.N_E))
            self.W_II = self.mask_II * torch.normal(self.mean_w_II, self.std_w_II, size=(self.N_I, self.N_I))

        connection_EE = Connection(source=self.neurons_E, target=self.neurons_E, w=self.W_EE.clone())
        connection_EI = Connection(source=self.neurons_E, target=self.neurons_I, w=self.W_EI)
        connection_IE = Connection(source=self.neurons_I, target=self.neurons_E, w=self.W_IE)
        connection_II = Connection(source=self.neurons_I, target=self.neurons_I, w=self.W_II)
        connection_noise_E = Connection(source=self.noise_E, target=self.neurons_E, w=self.w_ext * torch.eye(self.N_E))
        connection_noise_I = Connection(source=self.noise_I, target=self.neurons_I, w=self.w_ext * torch.eye(self.N_I))

        self.network.add_connection(connection_EE, source="E", target="E")
        self.network.add_connection(connection_EI, source="E", target="I")
        self.network.add_connection(connection_IE, source="I", target="E")
        self.network.add_connection(connection_II, source="I", target="I")
        self.network.add_connection(connection_noise_E, source="noise_E", target="E")
        self.network.add_connection(connection_noise_I, source="noise_I", target="I")

        # Monitors
        T = int(self.time / self.dt)
        self.mon_E = Monitor(self.neurons_E, state_vars=["s"], time=T)
        self.mon_I = Monitor(self.neurons_I, state_vars=["s"], time=T)
        self.network.add_monitor(self.mon_E, name="E_spikes")
        self.network.add_monitor(self.mon_I, name="I_spikes")

        print("Network built successfully")


    

    def run_one_sample(self, dataset, target):
        
        seed = 42
        random.seed(seed)
        idx = random.randrange(len(dataset))        
        while dataset[idx]["label"] != target:
            idx = random.randrange(len(dataset))

        sample = dataset[idx]
        label = sample["label"]
        feature_map = sample["feature_map"]
        feat_spikes = encode_feature_map(feature_map, self.time, self.dt, self.intensity)

        E_spike_counts, I_spike_counts, E_spikes, I_spikes = self.run(feat_spikes)

        #plot_raster(E_spikes, I_spikes, "Excitatory raster", "Inhibitory raster")
        #plot_rate_distribution(self.time, E_spike_counts, I_spike_counts, "Excitatory rate distribution", "Inhibitory rate distribution")
        #plot_spike_distribution(E_spike_counts, I_spike_counts, "Excitatory and inhibitory spike distribution")
        #plot_EI_positions(self.pos_E, self.pos_I)
        #plot_outgoing_connections(self.mask_EE, self.pos_E, 450, "Outgoing connections from excitatory neuron 450")
        plot_spikecount_grid(E_spike_counts, self.pos_E, "Excitatory spike count heatmap")

    
    def run_stimulation(self, dataset, examples, shuffle=True):
        

        pbar = tqdm(range(examples), desc=f"Stimulating network: (0 / {examples})")
        pairs = []

        for i, index in enumerate(pbar):
            if shuffle:
                index = random.choice(range(len(dataset)))
            sample = dataset[index]
            label = sample["label"]
            feature_map = sample["feature_map"]
            feat_spikes = encode_feature_map(feature_map, self.time, self.dt, self.intensity)

            E_spike_counts, I_spike_counts, E_spikes, I_spikes = self.run(feat_spikes)
            binned_E = spikes_to_binned_counts(E_spikes, bin_ms=self.bin_ms, dt=self.dt, time=self.time) # (N_bins, N_E)
            binned_E_flat = binned_E.flatten() # (N_bins * N_E,)
            features = binned_E_flat.float() # the features are the binned counts of the excitatory spikes in the network 
            pairs.append((features, label))
            pbar.set_description_str(f"Stimulating network: ({i+1} / {examples})")
            

        return pairs


    def run(self, feat_spikes):
        feat_spikes = feat_spikes.to(self.device)
        encoder = PoissonEncoder(time=1, dt=self.dt)

        for t in range(self.time):
            rates_XE = torch.ones(self.N_E) * self.rate_ext
            rates_XI = torch.ones(self.N_I) * self.rate_ext
            spikes_XE = encoder(rates_XE)
            spikes_XI = encoder(rates_XI)
            feat_t = feat_spikes[t:t+1] # (1, 1, D_in)
            self.network.run(inputs={"noise_E": spikes_XE.unsqueeze(0), "noise_I": spikes_XI.unsqueeze(0), "F": feat_t}, time=1)

        E_spikes = self.mon_E.get("s") # shape (T, 1, N_E)
        I_spikes = self.mon_I.get("s") # shape (T, 1, N_I)
        E_spike_counts = E_spikes.squeeze(1).sum(0) # shape (N_E,)
        I_spike_counts = I_spikes.squeeze(1).sum(0) # shape (N_I,)

        # Reset state variables
        self.network.reset_state_variables()
        self.mon_E.reset_state_variables()
        self.mon_I.reset_state_variables()

        return E_spike_counts, I_spike_counts, E_spikes, I_spikes



    def get_configuration_info(self):
        print("======== Quick Summary of the parameters ========")

        # Methods
        print("\n----- Methods -----")
        print(f"Heterogeneity: {self.heterogeneity}")
        print(f"Intensity: {self.intensity}")
        print(f"Self tuning: {self.self_tuning}")
        print(f"MNIST input: {self.mnist_input}")

        # Simulation parameters
        print("----- Simulation Parameters -----")
        print(f"Simulation time per sample [ms]: {self.time}")
        print(f"Time step [ms]: {self.dt}")

        # Network structure
        print("\n----- Network Structure -----")
        print(f"Total number of neurons: {self.n_neurons}")
        print(f"Number of excitatory neurons: {self.N_E}")
        print(f"Number of inhibitory neurons: {self.N_I}")

        # Connectivity/synapse parameters
        print("\n----- Connectivity/Synapse Parameters -----")
        print(f"Connection probability: {self.epsilon}")
        print(f"Relative inhibitory strength (g): {self.g}")
        print(f"eta (external rate multiplier): {self.eta}")
        print(f"(Excitatory ->) synapse weight: {self.w_E}")
        print(f"(Input ->) synapse weight: {self.w_input}")
        print(f"(Noise ->) synapse weight: {self.w_ext}")

       

        # Neuron parameters
        print("\n----- Neuron Parameters -----")
        print(f"Membrane threshold potential (theta) [mV]: {self.theta}")
        print(f"Membrane time constant (tau_m) [ms]: {self.tau_m}")

        # Rates
        print("\n----- Rates -----")
        print(f"Threshold rate (v_th): {self.v_th}")
        print(f"External rate (v_ext): {self.rate_ext}")

        # Miscellaneous
        print("\n----- Seed -----")
        print(f"Seed: {self.seed}")

        print("--------------------------------")
        
        print("--------------------------------")
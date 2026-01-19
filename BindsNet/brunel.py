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
    def __init__(self, n_neurons, time, dt):
        self.n_neurons = int(n_neurons)         # Total number of neurons
        self.time = int(time)                   # Simulation time per sample [ms]
        self.dt = float(dt)                     # Time step [ms]
        self.N_E = int(0.8 * self.n_neurons)    # Number of excitatory neurons
        self.N_I = self.n_neurons - self.N_E    # Number of inhibitory neurons
        self.epsilon = 0.1                      # Connection probability
        self.g = 5.0                            # Relative inhibitory strength
        self.w_e = 2.0                          # Excitatory weight
        self.w_I = -self.g * self.w_e           # Inhibitory weight
        self.w_mnist = 5.0                     # MNIST weight
        self.w_std = 0.1                        # STD of weights
        self.network = None
        self.neurons_E = None
        self.neurons_I = None
        self.mnist_in = None
        self.X_E = None
        self.X_I = None
        self.mon_E = None
        self.mon_I = None
        self.rate_ext = 80.0                   # External rate per synapse [Hz]
        self.w_ext = 4.0                       # External weight 
        
        self.seed = 10061999                    # My birthday:)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)

    def get_configuration_info(self):
        print("======== Quick Summary of the parameters ========")
        print(f"number of neurons: {self.n_neurons}")
        print(f"number of excitatory neurons: {self.N_E}")
        print(f"number of inhibitory neurons: {self.N_I}")
        print(f"connection probability: {self.epsilon}")
        print(f"inhibitory strength: {self.g}")
        print(f"external weight: {self.w_ext}")
        print(f"external rate: {self.rate_ext}")
        print(f"excitatory weight: {self.w_e}")
        print(f"MNIST weight: {self.w_mnist}")
        print(f"standard weight: {self.w_std}")
        print(f"external rate: {self.rate_ext}")
        print(f"time: {self.time}")
        print(f"dt: {self.dt}")
        print("--------------------------------")

        

    def build_brunel(self):
        print("Building network...")

        self.network = Network(dt=self.dt)

        # Excitatory and Inhibitory Neurons
        self.neurons_E = LIFNodes(n=self.N_E, tau=20.0, rest=0.0, reset=0.0, thresh=20.0, refrac=1, traces=True, tc_trace=20.0)
        self.neurons_I = LIFNodes(n=self.N_I, tau=20.0, rest=0.0, reset=0.0, thresh=20.0, refrac=1, traces=True, tc_trace=20.0)
        self.network.add_layer(self.neurons_E, name="E")
        self.network.add_layer(self.neurons_I, name="I")

        # Noise Inputs
        self.noise_E = Input(n=self.N_E)
        self.noise_I = Input(n=self.N_I)
        self.network.add_layer(self.noise_E, name="noise_E")
        self.network.add_layer(self.noise_I, name="noise_I")

        # Connecting noise to neurons
        connection_noise_E = Connection(source=self.noise_E, target=self.neurons_E, w=self.w_ext * torch.eye(self.N_E))
        connection_noise_I = Connection(source=self.noise_I, target=self.neurons_I, w=self.w_ext * torch.eye(self.N_I))
        self.network.add_connection(connection_noise_E, source="noise_E", target="E")
        self.network.add_connection(connection_noise_I, source="noise_I", target="I")

        # Input (MNIST) to Excitatory Neurons
        self.mnist_in = Input(n=784, traces=True, tc_trace=20.0)
        self.network.add_layer(self.mnist_in, name="MNIST")
        W = float(self.w_mnist) * torch.rand(784, self.N_E)/ np.sqrt(784)
        connection_mnist_E = Connection(source=self.mnist_in, target=self.neurons_E, w=W)
        self.network.add_connection(connection_mnist_E, source="MNIST", target="E")

        # Bernoulli masks for sparse connectivity
        mask_EE = torch.bernoulli(torch.full((self.N_E, self.N_E), self.epsilon))
        mask_EI = torch.bernoulli(torch.full((self.N_E, self.N_I), self.epsilon))
        mask_IE = torch.bernoulli(torch.full((self.N_I, self.N_E), self.epsilon))
        mask_II = torch.bernoulli(torch.full((self.N_I, self.N_I), self.epsilon))

        # Weights
        W_EE = mask_EE * torch.normal(self.w_e, self.w_std, size=(self.N_E, self.N_E)).clamp(min=0.0)   
        W_EI = mask_EI * torch.normal(self.w_e, self.w_std, size=(self.N_E, self.N_I)).clamp(min=0.0)   
        W_IE = mask_IE * torch.normal(self.w_I, self.w_std, size=(self.N_I, self.N_E)).clamp(max=0.0)
        W_II = mask_II * torch.normal(self.w_I, self.w_std, size=(self.N_I, self.N_I)).clamp(max=0.0)

        self.network.add_connection(Connection(self.neurons_E, self.neurons_E, w=W_EE), source="E", target="E")
        self.network.add_connection(Connection(self.neurons_E, self.neurons_I, w=W_EI), source="E", target="I")
        self.network.add_connection(Connection(self.neurons_I, self.neurons_E, w=W_IE), source="I", target="E")
        self.network.add_connection(Connection(self.neurons_I, self.neurons_I, w=W_II), source="I", target="I")

        # Monitors
        T = int(self.time / self.dt)
        self.mon_E = Monitor(self.neurons_E, state_vars=["s"], time=T)
        self.mon_I = Monitor(self.neurons_I, state_vars=["s"], time=T)
        self.network.add_monitor(self.mon_E, name="E_spikes")
        self.network.add_monitor(self.mon_I, name="I_spikes")

        print("Network built successfully")
        return self


    def stimulate_brunel(self, dataset, examples=500, shuffle=True, plot=False):

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
            pairs.append((features_E, label))

        if plot:
            plt.figure(figsize=(15, 5))
            plot_spikes({"E": E_spikes, "I": I_spikes})
            plt.show(block=True)
        
        return pairs
            
    # Helper methods:
    def run(self, image):
        
        T = image.shape[0] # number of time steps
        mnist_spikes = image.view(T, 1, 784).to("cpu")

        # External Poisson drive for full window
        p = float(np.clip(self.rate_ext * (self.dt / 1000.0), 0.0, 1.0)) # probability of a spike in each timestep
        noise_E_spikes = (torch.rand(T, 1, self.N_E) < p).float() # shape (T, 1, N_E)
        noise_I_spikes = (torch.rand(T, 1, self.N_I) < p).float() # shape (T, 1, N_I)

        self.network.run(inputs={"MNIST": mnist_spikes, "noise_E": noise_E_spikes, "noise_I": noise_I_spikes}, time=self.time)

        # Get spikes from monitors
        E_spikes = self.mon_E.get("s") # shape (T, 1, N_E)
        I_spikes = self.mon_I.get("s") # shape (T, 1, N_I)

        E_spike_counts = E_spikes.squeeze(1).sum(0) # shape (N_E,)
        I_spike_counts = I_spikes.squeeze(1).sum(0) # shape (N_I,)

        # Reset state variables
        self.network.reset_state_variables()

        return E_spike_counts, I_spike_counts, E_spikes, I_spikes

        


        


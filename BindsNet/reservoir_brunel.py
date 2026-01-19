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


class Reservoir:
    """
    Brunel-like E/I reservoir that can be driven by:
      - MNIST spike-encoded input (784 channels)
      - external Poisson drive to E and I via X_E and X_I

    After calling build_reservoir_brunel(), you can call:
      - train_reservoir(dataset, ...)
      - test_reservoir(dataset, ...)
      - debug_raster(dataset, idx, ...)
    """

    def __init__(self, n_neurons, time, dt, device="cpu"):
        self.n_neurons = int(n_neurons)
        self.time = int(time)          # ms
        self.dt = float(dt)            # ms
        self.device = device

        # E/I split (Brunel-style: ~80/20)
        self.N_E = int(0.8 * self.n_neurons)
        self.N_I = self.n_neurons - self.N_E

        # Network objects (set in build)
        self.net = None
        self.neurons_E = None
        self.neurons_I = None
        self.mnist_in = None
        self.X_E = None
        self.X_I = None

        # Monitors (set in build)
        self.mon_E = None
        self.mon_I = None

        # Readout selection: "E", "I", or "E+I"
        self.readout_from = "E"

        # External drive params (set in build)
        self.rate_ext = None
        self.w_ext = None

        print(f"number of excitatory neurons: {self.N_E}")
        print(f"number of inhibitory neurons: {self.N_I}")

    # -------------------------
    # helpers
    # -------------------------

    
    def _poisson_spikes(self, n, rate_hz):
        """
        Bernoulli approximation to Poisson:
        returns shape (T, 1, n) where T = time/dt
        """
        T = int(self.time / self.dt)
        p = float(np.clip(rate_hz * (self.dt / 1000.0), 0.0, 1.0))
        return (torch.rand(T, 1, n, device=self.device) < p).float()

    @staticmethod
    def _counts_from_spikes(s):
        """
        s: spike tensor (T, n) or (T, 1, n)
        returns: counts (n,)
        """
        if s.dim() == 3:
            s = s.squeeze(1)  # (T, n)
        return s.sum(0)

    def _extract_features(self):
        """
        Uses the monitors to produce one feature vector per sample.
        """
        sE = self.mon_E.get("s")
        sI = self.mon_I.get("s")

        cE = self._counts_from_spikes(sE)  # (N_E,)
        cI = self._counts_from_spikes(sI)  # (N_I,)

        return cE, cI

        # if self.readout_from == "E":
        #     return cE
        # if self.readout_from == "I":
        #     return cI
        # if self.readout_from == "E+I":
        #     return torch.cat([cE, cI], dim=0)

        # raise ValueError('readout_from must be "E", "I", or "E+I".')

    # -------------------------
    # build
    # -------------------------
    def build_reservoir_brunel(
        self,
        conn_prob=0.1,
        w_e=1.0,
        g=4.0,
        w_ext=10.0,
        rate_ext=150.0,
        std_w=0.1,
        w_mnist=0.1,
        readout_from="E",
        seed=None,
    ):
        """
        Builds a Brunel-like E/I network with:
          - recurrent sparse random E/I connectivity
          - external Poisson drive via X_E -> E and X_I -> I (diagonal)
          - MNIST input layer (784) projecting to E

        Does NOT run simulation; train/test methods handle that.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.readout_from = readout_from
        self.rate_ext = float(rate_ext)
        self.w_ext = float(w_ext)

        self.net = Network(dt=self.dt)
        


        # --- E/I neuron layers ---
        self.neurons_E = LIFNodes(
            n=self.N_E, 
            tau=20.0, 
            rest=0.0, 
            reset=0.0, 
            thresh=20.0, 
            refrac=1,
            traces=True,
            tc_trace=20.0
        )
        self.neurons_I = LIFNodes(
            n=self.N_I, tau=20.0, rest=0.0, reset=0.0, thresh=20.0, refrac=1
        )
        self.net.add_layer(self.neurons_E, name="E")
        self.net.add_layer(self.neurons_I, name="I")

        # --- external drive inputs (Poisson spikes will be fed in run()) ---
        self.X_E = Input(n=self.N_E)
        self.X_I = Input(n=self.N_I)
        self.net.add_layer(self.X_E, name="X_E")
        self.net.add_layer(self.X_I, name="X_I")

        # Strong diagonal external connections (like supervisor code)
        conn_XE = Connection(source=self.X_E, target=self.neurons_E, w=self.w_ext * torch.eye(self.N_E))
        conn_XI = Connection(source=self.X_I, target=self.neurons_I, w=self.w_ext * torch.eye(self.N_I))
        self.net.add_connection(conn_XE, source="X_E", target="E")
        self.net.add_connection(conn_XI, source="X_I", target="I")

        # --- MNIST input layer -> E ---
        self.mnist_in = Input(n=784,
                      traces=True,
                      tc_trace=20.0)

        self.net.add_layer(self.mnist_in, name="MNIST")

        # Random dense MNIST->E weights (you can sparsify later)
        W_MNIST_E = float(w_mnist) * torch.rand(784, self.N_E)
        conn_MNIST_E = Connection(source=self.mnist_in, target=self.neurons_E, w=W_MNIST_E, update_rule=PostPre, nu=(1e-6, 1e-6), wmin = 0.0, wmax=1.0)
        self.net.add_connection(conn_MNIST_E, source="MNIST", target="E")

        #store handle so we can inspect the weights later
        self.conn_MNIST_E = conn_MNIST_E

        # --- recurrent sparse E/I connectivity ---
        def bernoulli_mask(shape, p):
            return torch.bernoulli(torch.full(shape, p))

        # Means and signs
        mean_w_EE = +w_e #
        mean_w_EI = +w_e
        mean_w_IE = -g * w_e # this is just w_I
        mean_w_II = -g * w_e

        # masks
        m_EE = bernoulli_mask((self.N_E, self.N_E), conn_prob)
        m_EI = bernoulli_mask((self.N_E, self.N_I), conn_prob)
        m_IE = bernoulli_mask((self.N_I, self.N_E), conn_prob)
        m_II = bernoulli_mask((self.N_I, self.N_I), conn_prob)

        # weights
        W_EE = m_EE * torch.normal(mean_w_EE, std_w, size=(self.N_E, self.N_E))
        W_EI = m_EI * torch.normal(mean_w_EI, std_w, size=(self.N_E, self.N_I))
        W_IE = m_IE * torch.normal(mean_w_IE, std_w, size=(self.N_I, self.N_E))
        W_II = m_II * torch.normal(mean_w_II, std_w, size=(self.N_I, self.N_I))

        self.net.add_connection(Connection(self.neurons_E, self.neurons_E, w=W_EE), source="E", target="E")
        self.net.add_connection(Connection(self.neurons_E, self.neurons_I, w=W_EI), source="E", target="I")
        self.net.add_connection(Connection(self.neurons_I, self.neurons_E, w=W_IE), source="I", target="E")
        self.net.add_connection(Connection(self.neurons_I, self.neurons_I, w=W_II), source="I", target="I")

        # --- monitors for E/I spikes over full window ---
        T = int(self.time / self.dt)
        self.mon_E = Monitor(self.neurons_E, state_vars=["s"], time=T, device=self.device)
        self.mon_I = Monitor(self.neurons_I, state_vars=["s"], time=T, device=self.device)
        self.net.add_monitor(self.mon_E, "E_spikes")
        self.net.add_monitor(self.mon_I, "I_spikes")

        # move to device if needed (BindsNET is sometimes CPU-only depending on install)
        # We'll still keep tensors on self.device for inputs.
        return self

    def set_learning(self, flag: bool):
        self.net.learning = flag

    # -------------------------
    # run one sample
    # -------------------------
    def run_one(self, encoded_image):
        """
        encoded_image can be:
        (T, 1, 1, 28, 28)
        (T, 1, 28, 28)
        (T, 1, 784)
        Returns: feature vector (counts)
        """
        assert self.net is not None, "Call build_reservoir_brunel() first."

        # Convert to (T, 1, 784)
        if encoded_image.dim() == 5:
            # (T, 1, 1, 28, 28)
            T = encoded_image.shape[0]
            mnist_spikes = encoded_image.view(T, 1, 784).to(self.device)

        elif encoded_image.dim() == 4:
            # (T, 1, 28, 28)
            T = encoded_image.shape[0]
            mnist_spikes = encoded_image.view(T, 1, 784).to(self.device)

        elif encoded_image.dim() == 3 and encoded_image.shape[-1] == 784:
            # (T, 1, 784)
            mnist_spikes = encoded_image.to(self.device)

        else:
            raise ValueError(f"Unexpected encoded_image shape: {tuple(encoded_image.shape)}")

        # External Poisson drive for full window
        xE = self._poisson_spikes(self.N_E, self.rate_ext)
        xI = self._poisson_spikes(self.N_I, self.rate_ext)

        self.net.run(inputs={"MNIST": mnist_spikes, "X_E": xE, "X_I": xI}, time=self.time)

        features_E, features_I = self._extract_features()
        sE = self.mon_E.get("s").squeeze()
        sI = self.mon_I.get("s").squeeze()
        
        self.net.reset_state_variables()
        return features_E, features_I, sE, sI

    # -------------------------
    # train/test: return (features, label) pairs
    # -------------------------

    def run_analysis(self, dataset, examples=100, label=1):
        n_total = len(dataset)
        n_iters = min(examples, n_total)
        indices = list(range(n_iters))
        pbar = tqdm(indices, desc=f"Analysis progress: (0 / {n_iters})")
        for i, idx in enumerate(pbar):
            dp = dataset[idx]
            x = dp["encoded_image"]
            y = dp["label"]
            if y == label:
                pbar.set_description_str(f"Analysis progress: ({i+1} / {n_iters})")
                feat, feat_I, sE, sI = self.run_one(x)
                plt.figure(figsize=(15, 5))
                x_flat = x.sum(dim=0).flatten()
                plot_spikes({"E": sE, "I": sI})
                #plt.show(block=True)
                #plt.savefig(f"BindsNet/results/raster_plots/label{y}_raster_plot_index{idx}.png")
                plt.close()
                
            
                
                for i, val in enumerate(x_flat):
                    num_spikes = int(val)
                    plt.vlines(i, 0, num_spikes, color='skyblue', alpha= 0.8)
                for i, val in enumerate(feat):
                    i = len(x_flat) + i
                    num_spikes = int(val)
                    plt.vlines(i, 0, num_spikes, color='salmon', alpha= 0.8)
                for i, val in enumerate(feat_I):
                    i = len(x_flat) + len(feat) + i
                    num_spikes = int(val)
                    plt.vlines(i, 0, num_spikes, color='mediumseagreen', alpha= 0.8)
                plt.xlabel("Neuron index")
                plt.ylabel("Number of spikes")
                plt.title(f"Input and output spike distributions \n Label: {y}")
                plt.tight_layout()
                plt.grid(1)
                #plt.savefig(f"BindsNet/results/spike_distribution/label{y}_spike_distribution_index{idx}.png")
                plt.show(block=True)
                plt.close()
            else:
                pass
                
            


    def train_reservoir(self, dataset, examples=500, shuffle=True, plot=False):


        n_total = len(dataset)
        n_iters = min(examples, n_total)
        indices = torch.randperm(n_total)[:n_iters].tolist() if shuffle else list(range(n_iters))

        training_pairs = []
        pbar = tqdm(indices, desc=f"Train progress: (0 / {n_iters})")

        for i, idx in enumerate(pbar):
            dp = dataset[idx]
            x = dp["encoded_image"]
            y = dp["label"]
            pbar.set_description_str(f"Train progress: ({i+1} / {n_iters})")

            feat, feat_I, sE, sI = self.run_one(x)
            training_pairs.append((feat.detach().cpu(), y))


            if plot:
                plt.figure(figsize=(15, 5))
                x_flat = x.sum(dim=0).flatten()
                plot_spikes({"E": sE, "I": sI})
                plt.show(block=True)
                
            
                
                for i, val in enumerate(x_flat):
                    num_spikes = int(val)
                    plt.vlines(i, 0, num_spikes, color='skyblue', alpha= 0.8)
                for i, val in enumerate(feat):
                    i = len(x_flat) + i
                    num_spikes = int(val)
                    plt.vlines(i, 0, num_spikes, color='salmon', alpha= 0.8)
                for i, val in enumerate(feat_I):
                    i = len(x_flat) + len(feat) + i
                    num_spikes = int(val)
                    plt.vlines(i, 0, num_spikes, color='mediumseagreen', alpha= 0.8)
                plt.xlabel("Neuron index")
                plt.ylabel("Number of spikes")
                plt.title("Input and output spike distributions")
                plt.tight_layout()
                plt.grid(1)
                plt.show(block=True)
                #input("Press Enter to continue...")  # Prevents plot from closing immediately
            
         



        return training_pairs

    def test_reservoir(self, dataset, examples=500, shuffle=False):
        assert self.net is not None, "Call build_reservoir_brunel() first."


        n_total = len(dataset)
        n_iters = min(examples, n_total)
        indices = torch.randperm(n_total)[:n_iters].tolist() if shuffle else list(range(n_iters))

        test_pairs = []
        pbar = tqdm(indices, desc=f"Test progress: (0 / {n_iters})")

        for i, idx in enumerate(pbar):
            dp = dataset[idx]
            x = dp["encoded_image"]
            y = dp["label"]
            pbar.set_description_str(f"Test progress: ({i+1} / {n_iters})")

            feat = self.run_one(x)
            test_pairs.append((feat.detach().cpu(), y))

        return test_pairs

    # -------------------------
    # debugging: plot rasters for one MNIST sample
    # -------------------------
    def debug_raster(self, dataset, idx=0, title=None):
        """
        Runs one sample and plots spikes for E and I over the window.
        """
        assert self.net is not None, "Call build_reservoir_brunel() first."

        dp = dataset[idx]
        x = dp["encoded_image"]
        y = dp["label"]

        # run once (but don't reset until after we grab spike tensors)
        if x.dim() == 5:
            T = x.shape[0]
            mnist_spikes = x.view(T, 1, 784).to(self.device)
        else:
            mnist_spikes = x.to(self.device)

        xE = self._poisson_spikes(self.N_E, self.rate_ext)
        xI = self._poisson_spikes(self.N_I, self.rate_ext)

        self.net.run(inputs={"MNIST": mnist_spikes, "X_E": xE, "X_I": xI}, time=self.time)

        sE = self.mon_E.get("s")
        sI = self.mon_I.get("s")

        # print quick activity stats
        cE = self._counts_from_spikes(sE)
        cI = self._counts_from_spikes(sI)
        print(f"[debug] label={int(y)} | total E spikes={int(cE.sum().item())} | nonzero E neurons={(cE>0).sum().item()}")
        print(f"[debug] label={int(y)} | total I spikes={int(cI.sum().item())} | nonzero I neurons={(cI>0).sum().item()}")

        plt.ioff()
        plot_spikes({"E": sE, "I": sI})
        plt.title(title if title else f"Raster | idx={idx} | label={int(y)}")
        plt.show()

        self.net.reset_state_variables()




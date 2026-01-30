from input_data import Data
from brunel import Brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch

# General parameters
n_neurons=2500
n_epochs=20
examples_stdp_train = 1000
examples_train=250
examples_test=250
time=50 # the temporal bins are 50ms, so this shoudl be minimum 50ms
dt=1.0
intensity=600

stdp = True
mnist_input = True
self_tuning = False
reset = True

g = 5
eta = 0.6
sigma = 1
epsilon = 0.3

# =============================== Data ===============================
data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)
train_dataset, test_dataset = data.load_MNIST()

#=============================== Brunel ===============================
brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt, mnist_input=True, self_tuning=False, eta=0.6, g=5.0)
brunel.build_brunel()


# =============================== Training ===============================
brunel.train_stdp(train_dataset, examples=examples_stdp_train, shuffle=True)
conn = brunel.network.connections[("MNIST", "E")]
#conn.update_rule = None       # disables STDP updates
conn.nu = (0.0, 0.0) 

# =============================== Testing ===============================
training_pairs, CV_list, rho_mean_list, rate_list, g_list, eta_list = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True)
test_pairs, CV_test_list, rho_mean_test_list, rate_test_list, g_test_list, eta_test_list = brunel.stimulate_brunel(test_dataset, examples=examples_test, shuffle=False)

print("mean CV: ", np.mean(CV_list))
print("mean rho_mean: ", np.mean(rho_mean_list))
print("mean rate: ", np.mean(rate_list))
print("mean g: ", np.mean(g_list))
print("mean eta: ", np.mean(eta_list))

feature_dim = training_pairs[0][0].numel()
readout = Readout(input_size=feature_dim, num_classes=10)
readout.train_readout(training_pairs, n_epochs=n_epochs)
acc = readout.test_readout(test_pairs)
print(f"Accuracy: {acc:.2f}%")




def normalize_input_weights(self, decay=1e-5, target_sum=None):
    conn = self.network.connections[("MNIST", "E")]
    with torch.no_grad():
        W = conn.w

        # --- very small decay ---
        W.mul_(1.0 - decay)

        # --- column-wise normalization (per E neuron) ---
        col_sum = W.sum(dim=0, keepdim=True) + 1e-12

        if target_sum is None:
            target_sum = col_sum.mean()

        W.mul_(target_sum / col_sum)

        # safety clamp
        W.clamp_(0.0, conn.wmax)

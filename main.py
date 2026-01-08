
from preprocessing import Preprocessing
from brunel import Brunel
from readout import Readout
import numpy as np
from tqdm import tqdm
import time
import nest
import os
from datetime import datetime

from sklearn.datasets import load_iris

def freeze_stdp(brunel):
    all_neurons = brunel.nodes_ex + brunel.nodes_in
    conns = nest.GetConnections(source=brunel.nodes_ex, target=all_neurons, synapse_model="excitatory_stdp")
    conns.set({"lambda": 0.0})

def reset_state(brunel):
    nest.SetStatus(brunel.nodes_ex + brunel.nodes_in, {"V_m": 0.0})





iris = load_iris()
X_iris = iris.data
y_iris = iris.target


preprocessing = Preprocessing()
#dataset_info, X, y = preprocessing.import_iris_dataset()
dataset_info, X, y = preprocessing.import_moon_dataset(plot=False)
#dataset_info, X, y = preprocessing.import_circles_dataset(plot=False)

# readout = Readout(X_brunel, y)
# readout.cross_validation_pca()




# Prepare the filename and string to save
now = datetime.now()
filename = f"data/info_{now.strftime('%Y%m%d_%H%M%S')}.txt"











#spike_vector = brunel.get_spike_vector()
#brunel.get_average_firing(200)
#brunel.get_stdp_weights(show_top_bottom=False)
#brunel.plot_raster()


# spike_matrix = []

# from tqdm import tqdm
# for i, sample in enumerate(tqdm(X, desc="Processing samples")):
#     x = sample.reshape(1, -1)
#     brunel = Brunel(input=x, stdp=True, reset=True)
#     brunel.build_network()
#     brunel.simulate()
#     spike_vector = brunel.get_spike_vector()
#     spike_matrix.append(spike_vector)
#     print(f"Sample {i}: {x} with shape {x.shape}")

# spike_matrix = np.array(spike_matrix)
# np.save("data/spike_matrix.npy", spike_matrix)

##########################

brunel = Brunel(input=X[0].reshape(1, -1), stdp=True, reset=True, N_neurons=5000)
summary = brunel.print_summary()
#brunel.build_network()

# brunel.simtime = 200.0
# rng = np.random.RandomState(0)
# n_epochs = 2

# for ep in range(n_epochs):
#     print(f"Epoch {ep+1} of {n_epochs}")
#     #brunel.get_stdp_weights(bins=100, show_top_bottom=False, plot=True, return_weights=False, folder_name="")
#     idx = rng.permutation(len(X))
#     for j in tqdm(idx, desc="Samples"):
#         brunel.give_input(X[j].reshape(1, -1))
#         brunel.simulate()
#         reset_state(brunel)
#         #brunel.get_stdp_weights(bins=100, show_top_bottom=False, plot=True, return_weights=False, folder_name="weights_training")

# #brunel.get_stdp_weights(bins=100, show_top_bottom=False, plot=True, return_weights=False, folder_name="")
# freeze_stdp(brunel)
# brunel.simtime = 1000.0


# spike_matrix = []
# for sample in tqdm(X, desc="Processing samples"):
#     print(f"Processing sample {sample}")
#     reset_state(brunel)
#     brunel.give_input(sample.reshape(1, -1))
#     t0 = nest.GetKernelStatus("biological_time")
#     brunel.simulate()
#     t1 = nest.GetKernelStatus("biological_time")

#     spike_vector = brunel.get_spike_vector_window(t0, t1)
#     spike_matrix.append(spike_vector)
#     firing_rate_ex, firing_rate_in = brunel.get_firing_rates_window(t0, t1)
    
#     with open("data/firing_rates.txt", "a") as f:
#         f.write("Sample:\n")
#         f.write(f"  Excitatory firing rate: {firing_rate_ex}\n")
#         f.write(f"  Inhibitory firing rate: {firing_rate_in}\n")
#         f.write("-" * 30 + "\n")
#         f.write(f"  Spike vector: {spike_vector}\n")
    






# #X_brunel = np.load("data/spike_matrix.npy")
# #spike_matrix = np.asarray(spike_matrix, dtype=float)
# #np.save("data/spike_matrix.npy", spike_matrix)
# spike_matrix = np.load("data/spike_matrix.npy")

readout = Readout(X, y)
summary_readout = readout.cross_validation_pca()


# Save to file
with open(filename, "w") as f:
    f.write("======== Dataset information: ========")
    f.write("\n")
    f.write(f"{dataset_info}")
    f.write("\n")
    f.write("\n")
    f.write("======== Brunel summary: ========")
    f.write("\n")
    f.write(summary)
    f.write("\n")
    f.write("\n")
    f.write("======== Readout summary: ========")
    f.write("\n")
    f.write(summary_readout)



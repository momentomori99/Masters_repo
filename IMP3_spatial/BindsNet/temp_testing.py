# from input_data import Data
from input_data_CNN import Data as Data_CNN
from input_data import Data

from visualizer import *
from brunel import Brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt


n_neurons=2500
n_epochs=20
examples_train=1
examples_test=500
time = 250
intensity = 300
dt = 1

stdp = False
mnist_input = True
self_tuning = False
reset = True
use_gabor = False
heterogeneity = False
g = 5
eta = 1
sigma = 1
epsilon = 0.3

data = Data_CNN(dt=dt, intensity=intensity, kernel_size=9, thetas_deg=(0, 45, 90, 135), use_gabor=use_gabor)
train_dataset, test_dataset = data.load_MNIST()



brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt, heterogeneity=heterogeneity, mnist_input=mnist_input, self_tuning=self_tuning, stdp = stdp, reset = reset, eta=eta, g=g, sigma=sigma, epsilon=epsilon, intensity=intensity)
brunel.build_brunel()

#brunel.plot_EI_positions()
brunel.plot_outgoing_connections(brunel.mask_EE, brunel.pos_E, 450)


#brunel.run_one_sample(train_dataset, 8)
#training_pairs, CV_list, rho_mean_list, rate_list, g_list, eta_list = brunel.stimulate_brunel(train_dataset, examples=examples_train)
#test_pairs, CV_test_list, rho_mean_test_list, rate_test_list, g_test_list, eta_test_list = brunel.stimulate_brunel(test_dataset, examples=examples_test)

# for i in range(100):
#     sample = train_dataset[i]
#     if sample["label"] == 3:

#         feature_map = sample["feature_map"]
#         feat_spikes = brunel.encode_feature_map(feature_map, time, dt, intensity)

#         #data.plot_feature_map(feature_map, title=f"Feature map for label {sample['label']}")

#         E_counts, I_counts, E_spikes, I_spikes = brunel.run(feat_spikes, brunel.rate_ext)
#         #brunel.plot_raster(E_spikes, I_spikes, f"Excitatory raster, label: {sample['label']}", f"Inhibitory raster, label: {sample['label']}")
#         brunel.plot_spikecount_grid_E(E_counts, title=f"E spike counts (2D grid) label: {sample['label']}")

#sample = train_dataset[0]
# state = True
# i = 0
# label = 5
# while state:
#     sample = train_dataset[i]
#     if sample["label"] == label:
#         feature_map = sample["feature_map"]
#         feat_spikes = data.encode_feature_map(feature_map, time, dt, intensity)
#         data.plot_feature_map(feature_map, title=f"Feature map for label {sample['label']}")

#         E_counts, I_counts, E_spikes, I_spikes = brunel.run(feat_spikes, brunel.rate_ext)
#         #brunel.plot_raster(E_spikes, I_spikes, "Excitatory raster", "Inhibitory raster")
#         brunel.plot_spikecount_grid_E(E_counts, title=f"E spike counts (2D grid) label: {sample['label']}")
#         state = False




# i = 0
# while True:
#     sample = train_dataset[i]
#     feature_map = sample["feature_map"]
#     data.plot_feature_map(feature_map, title=f"Feature map for label {sample['label']}")
#     i += 1




#train_dataset, test_dataset = data.load_MNIST()


#brunel.plot_EI_positions()
#brunel.plot_outgoing_connections(brunel.mask_EE, brunel.pos_E, 500)

#brunel.run_one_sample(test_dataset, 1)













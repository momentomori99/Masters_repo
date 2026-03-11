from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
import numpy as np
import matplotlib.pyplot as plt
from visualization.visualizations_readout import plot_tsne, plot_confusion_heatmap
import torch


# pramaters
n_neurons = 2500
n_epochs = 100
examples_train = 500
examples_test = 100

time = 100
dt = 1.0

intensity = 600
seed = 42

mnist_input = True
heterogeneity = True
self_tuning = False
spatial = True
convolution = False

g = 4
eta = 1.0
sigma_input = 1
sigma_network = 0.3
epsilon = 0.1

data_CNN = Data_CNN(dt=dt, intensity=intensity, kernel_size=9, thetas_deg=(0, 45, 90, 135), convolution=convolution)
train_dataset, test_dataset = data_CNN.load_MNIST()

framework = Framework(
    n_neurons=n_neurons,
    time=time,
    dt=dt,
    seed=seed,
    heterogeneity=heterogeneity,
    mnist_input=mnist_input,
    self_tuning=self_tuning,
    spatial=spatial,
    convolution=convolution,
    g=g,
    eta=eta,
    sigma_input=sigma_input,
    sigma_network=sigma_network,
    epsilon=epsilon,
    intensity=intensity,
)
framework.build_network()
#framework.run_one_sample(train_dataset, 0)





#print(framework.K * framework.Hf * framework.Wf)
print(framework.mask_EI)


# Define zoomed-in region (adjust as desired)
# input_start, input_end = 0, 100   # e.g., first 100 input indices
# neuron_start, neuron_end = 0, 100 # e.g., first 100 excitatory neuron indices

# zoomed_W_in = framework.W_in[input_start:input_end, neuron_start:neuron_end]

pos = framework.pos_E  # (N_E, 2)
rows, cols = framework.rows, framework.cols

order = torch.argsort(pos[:, 0] * cols + pos[:, 1])  # row-major ordering
M = framework.mask_EE[order][:, order]

plt.figure(figsize=(8,6))
plt.imshow(M.cpu(), aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Mask value')
plt.xlabel('Post E (sorted by position)')
plt.ylabel('Pre E (sorted by position)')
plt.show(block=True)









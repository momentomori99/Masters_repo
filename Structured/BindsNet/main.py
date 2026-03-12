from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
from tools.metrics import calculate_fisher_ratio
import numpy as np
import matplotlib.pyplot as plt
from visualization.visualizations_readout import plot_tsne, plot_confusion_heatmap


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
log_normal = False
stdp = True
stdp_samples = 100

g = 5
eta = 0.6
sigma_input = 1
sigma_network = 1
epsilon = 0.3

data_CNN = Data_CNN(dt=dt, intensity=intensity, kernel_size=9, thetas_deg=(0, 45, 90, 135), convolution=convolution)
train_dataset, test_dataset = data_CNN.load_MNIST()

framework = Framework(
    n_neurons=n_neurons,
    time=time,
    dt=dt,
    seed=seed,
    log_normal=log_normal,
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
    stdp=stdp,
)
framework.build_network()

if stdp:
    framework.plot_input_weights(title="Weights Before STDP")
    framework.run_stdp_training(train_dataset, n_samples=stdp_samples)
    framework.plot_input_weights(title="Weights After STDP")
    framework.plot_weight_change()

# pairs_train = framework.run_stimulation(train_dataset, examples_train)
# pairs_test = framework.run_stimulation(test_dataset, examples_test)

# feature_dim = pairs_train[0][0].numel()
# readout = Readout(input_size=feature_dim, num_classes=10, seed=seed)
# readout.train_readout(pairs_train, n_epochs=n_epochs)
# acc = readout.test_readout(pairs_test)
# print(f"Accuracy: {acc:.2f}%")

# fisher_J = calculate_fisher_ratio(pairs_test)
# print(f"Fisher ratio: {fisher_J:.4f}")

# plot_tsne(pairs_train, perplexity=30)
# plot_confusion_heatmap(pairs_train)



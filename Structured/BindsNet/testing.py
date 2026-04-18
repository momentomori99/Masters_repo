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

time = 1000
dt = 1.0

intensity = 64
seed = 42

mnist_input = True
heterogeneity = False
self_tuning = False
spatial = False
convolution = False

g = 5
eta = 0.6
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
framework.run_one_sample(train_dataset, 0)










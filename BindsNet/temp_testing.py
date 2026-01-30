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
time = 200
intensity = 1000
dt = 1

stdp = False
mnist_input = True
self_tuning = False
reset = True

g = 5
eta = 1.0
sigma = 1.0
epsilon = 0.3

data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)

train_dataset, test_dataset = data.load_MNIST()

brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt, mnist_input=mnist_input, self_tuning=self_tuning, stdp = stdp, reset = reset, eta=eta, g=g, sigma=sigma, epsilon=epsilon)
brunel.build_brunel()

brunel.plot_EI_positions()
#brunel.plot_outgoing_connections(brunel.mask_EE, brunel.pos_E, 500)

brunel.run_one_sample(test_dataset, 1)













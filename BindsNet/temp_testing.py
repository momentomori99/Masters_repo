from input_data import Data
from reservoir import Reservoir
from reservoir_brunel import Reservoir as Reservoir_brunel
from brunel import Brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch

import matplotlib.pyplot as plt


n_neurons=1000
n_epochs=500
examples_train=500
examples_test=500
time = 250
intensity = 64
dt = 1.0





data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)
train_dataset, test_dataset = data.load_MNIST()

brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt)
brunel.build_brunel()




training_pairs = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True, plot=True)
test_pairs = brunel.stimulate_brunel(test_dataset, examples=examples_test, shuffle=False, plot=False)



feature_dim = training_pairs[0][0].numel()
readout = Readout(input_size=feature_dim, num_classes=10)
readout.train_readout(training_pairs, n_epochs=n_epochs)

acc = readout.test_readout(test_pairs)
print(f"Accuracy: {acc:.2f}%")









from input_data import Data
from brunel import Brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch

# General parameters
n_neurons=1100
n_epochs=200
examples_train=500
examples_test=500
time=250
dt=1.0
intensity=320

# =============================== Data ===============================
data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)
train_dataset, test_dataset = data.load_MNIST()

# =============================== Brunel ===============================
brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt)
brunel.build_brunel()

training_pairs = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True)
test_pairs = brunel.stimulate_brunel(test_dataset, examples=examples_test, shuffle=False)

# =============================== Readout ===============================
feature_dim = training_pairs[0][0].numel()
readout = Readout(input_size=feature_dim, num_classes=10)
readout.train_readout(training_pairs, n_epochs=n_epochs)
acc = readout.test_readout(test_pairs)
print(f"Accuracy: {acc:.2f}%")




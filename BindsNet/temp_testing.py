from input_data import Data
from visualizer import *
from brunel import Brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt


n_neurons=800
n_epochs=20
examples_train=1
examples_test=500
time = 1000
intensity = 64
dt = 1







data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)
train_dataset, test_dataset = data.load_MNIST()

# res = Reservoir(n_neurons=n_neurons, time=time, dt=dt)

# res.build_reservoir()
# training_pairs = res.train_reservoir_spike_counts(train_dataset, examples=examples_train, shuffle=True)
# test_pairs = res.test_reservoir_spike_counts(test_dataset, examples=examples_test, shuffle=False)

brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt)
brunel.build_brunel()
#brunel.get_configuration_info()
brunel.run_one_sample(test_dataset, 5)
training_pairs = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True)



#visualizer = Visualizer(training_pairs)
#visualizer.plot_tsne(perplexity=30)
# visualizer.plot_pca(n_components=2)
# visualizer.plot_class_separation()
# visualizer.plot_confusion_proximity()








# training_pairs = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True, plot=True)
# test_pairs = brunel.stimulate_brunel(test_dataset, examples=examples_test, shuffle=False, plot=False)



#feature_dim = training_pairs[0][0].numel()
#readout = Readout(input_size=feature_dim, num_classes=10)
#readout2 = Readout2(input_size=feature_dim, num_classes=10)
#readout.train_readout(training_pairs, n_epochs=n_epochs)
#readout2.train_readout(training_pairs, n_epochs=n_epochs)

#acc = readout.test_readout(test_pairs)
#acc2 = readout2.test_readout(test_pairs)
#print(f"Accuracy: {acc:.2f}%")
#print(f"Accuracy: {acc2:.2f}%")








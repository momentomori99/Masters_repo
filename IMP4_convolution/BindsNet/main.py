from input_data_CNN import Data as Data_CNN
from brunel import Brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch
from visualizer import *

# General parameters
n_neurons=2500
n_epochs=100
examples_stdp_train = 1000
examples_train=500
examples_test=100
time=100 # the temporal bins are 50ms, so this should be minimum 50ms
dt=1.0
intensity=600
seed = 42

stdp = False
mnist_input = True
self_tuning = False
reset = True
heterogeneity = False

g = 5
eta = 0.9
sigma = 1
epsilon = 0.3

# =============================== Data ===============================
data_CNN = Data_CNN(dt=dt, intensity=intensity, kernel_size=9, thetas_deg=(0, 45, 90, 135))
train_dataset, test_dataset = data_CNN.load_MNIST()



#=============================== Brunel ===============================
brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt, heterogeneity=heterogeneity, mnist_input=True, self_tuning=False, eta=eta, g=g, intensity=intensity)
brunel.build_brunel()



# =============================== Testing ===============================
training_pairs, CV_list, rho_mean_list, rate_list, g_list, eta_list = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True)
test_pairs, CV_test_list, rho_mean_test_list, rate_test_list, g_test_list, eta_test_list = brunel.stimulate_brunel(test_dataset, examples=examples_test, shuffle=False)


#visualizers
#visualizer = Visualizer(training_pairs)
#visualizer.plot_tsne(perplexity=30)
#visualizer.plot_pca(n_components=2)
#visualizer.plot_pca(n_components=3)
#visualizer.plot_distance_matrix(metric='euclidean')
#visualizer.plot_class_separation()
#visualizer.plot_per_class_separation()
#visualizer.plot_confusion_proximity()   

print("mean CV: ", np.mean(CV_list))
print("mean rho_mean: ", np.mean(rho_mean_list))
print("mean rate: ", np.mean(rate_list))
print("mean g: ", np.mean(g_list))
print("mean eta: ", np.mean(eta_list))

feature_dim = training_pairs[0][0].numel()
readout = Readout(input_size=feature_dim, num_classes=10, seed=seed)


readout.train_readout(training_pairs, n_epochs=n_epochs)
acc = readout.test_readout(test_pairs)
print(f"Accuracy: {acc:.2f}%")



import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'BindsNet'))

from framework import STDPImprintFramework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
from tools.metrics import calculate_fisher_ratio

# ── Parameters ──
neurons_per_class = 20
num_classes = 10
n_epochs = 100
examples_train = 500
examples_test = 100

time = 100
dt = 1.0
intensity = 600
seed = 42

g = 5
eta = 0.2
epsilon = 0.1

nu_stdp = (1e-4, 1e-2)
n_samples_per_class = 10

# ── Data ──
data_CNN = Data_CNN(dt=dt, intensity=intensity, kernel_size=9,
                    thetas_deg=(0, 45, 90, 135), convolution=False)
train_dataset, test_dataset = data_CNN.load_MNIST()

# ── Build network ──
framework = STDPImprintFramework(
    neurons_per_class=neurons_per_class,
    num_classes=num_classes,
    time=time,
    dt=dt,
    seed=seed,
    nu_stdp=nu_stdp,
    intensity=intensity,
    g=g,
    eta=eta,
    epsilon=epsilon,
)
framework.build_network()

# ── STDP imprint training ──
framework.plot_input_weights(title="Weights Before STDP")
framework.run_stdp_training(train_dataset, n_samples_per_neuron=n_samples_per_class)
framework.plot_input_weights(title="Weights After STDP")
framework.plot_weight_change()

# ── Visualize class responses ──
framework.plot_class_responses(test_dataset, classes_to_show=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9])
framework.plot_group_responses(test_dataset, classes_to_show=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9])

# # ── Stimulation (frozen weights) ──
# pairs_train = framework.run_stimulation(train_dataset, examples_train)
# pairs_test = framework.run_stimulation(test_dataset, examples_test)

# # ── Readout ──
# feature_dim = pairs_train[0][0].numel()
# readout = Readout(input_size=feature_dim, num_classes=num_classes, seed=seed)
# readout.train_readout(pairs_train, n_epochs=n_epochs)
# acc = readout.test_readout(pairs_test)
# print(f"Accuracy: {acc:.2f}%")

# fisher_J = calculate_fisher_ratio(pairs_test)
# print(f"Fisher ratio: {fisher_J:.4f}")

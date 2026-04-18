from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
from tools.metrics import calculate_fisher_ratio
from tools.pca import apply_pca, plot_explained_variance
import numpy as np
import torch
import matplotlib.pyplot as plt
from visualization.visualizations_readout import plot_tsne, plot_confusion_heatmap, plot_rsa_heatmap, plot_tsne_rsa
from visualization.visualizations_spatial import plot_EI_positions, plot_outgoing_connections, plot_spikecount_grid



# pramaters
n_neurons = 1000
n_epochs = 50
examples_train = 500
examples_test = 200
pca = False

n_components = 60

time = 1000
dt = 1.0

intensity = 64
seed = 54

mnist_input = False
heterogeneity = False
self_tuning = False
spatial = False
convolution = False
log_normal = False
stdp = False
stdp_samples = 100

g = 5#5
eta = 1.5#0.6
sigma_input = 1
sigma_network = 1
epsilon = 0.5

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
    nu_stdp=(1e-6, 1e-4),
    norm_stdp=None,
)
framework.build_network()

if stdp:
    framework.plot_input_weights(title="Weights Before STDP")
    framework.run_stdp_training(train_dataset, n_samples=stdp_samples)
    framework.plot_input_weights(title="Weights After STDP")
    framework.plot_weight_change()

#plot_EI_positions(framework.pos_E, framework.pos_I)
#plot_outgoing_connections(framework.mask_EE, framework.pos_E, 505)
#plot_spikecount_grid(framework.E_spike_counts, framework.pos_E, "Excitatory spike count heatmap")


target_label = 0
framework.run_one_sample(train_dataset, target_label)

# pairs_train, CV_list, rho_mean_list, rate_list, g_list, eta_list = framework.run_stimulation(train_dataset, examples_train)
# pairs_test, *_ = framework.run_stimulation(test_dataset, examples_test)


# # Convert lists to numpy arrays for easier plotting
# CV_arr = np.array(CV_list)
# rho_arr = np.array(rho_mean_list)
# rate_arr = np.array(rate_list)
# g_arr = np.array(g_list)
# eta_arr = np.array(eta_list)

# print(f"Average CV: {CV_arr.mean():.4f}")
# print(f"Average rho mean: {rho_arr.mean():.4f}")


# steps = np.arange(len(CV_arr))

# fig, axs = plt.subplots(3, 2, figsize=(12, 10))
# axs = axs.flatten()

# axs[0].plot(steps, CV_arr)
# axs[0].set_title("CV (coefficient of variation)")
# axs[0].set_xlabel("Step")
# axs[0].set_ylabel("CV")

# axs[1].plot(steps, rho_arr)
# axs[1].set_title("Mean Population Firing Rate (rho mean)")
# axs[1].set_xlabel("Step")
# axs[1].set_ylabel("Rho Mean")

# axs[2].plot(steps, rate_arr)
# axs[2].set_title("Rate (Hz)")
# axs[2].set_xlabel("Step")
# axs[2].set_ylabel("Rate")

# axs[3].plot(steps, g_arr)
# axs[3].set_title("g (Inhibition/Excitation ratio)")
# axs[3].set_xlabel("Step")
# axs[3].set_ylabel("g")

# axs[4].plot(steps, eta_arr)
# axs[4].set_title("eta (External drive parameter)")
# axs[4].set_xlabel("Step")
# axs[4].set_ylabel("eta")

# # Hide the last subplot if not used
# axs[5].axis('off')

# plt.tight_layout()
# #plt.savefig("results/training_dynamics.png", dpi=150, bbox_inches='tight')
# plt.show(block=True)


# if pca:
#     pairs_train, pairs_test_pca = apply_pca(pairs_train, pairs_test, n_components)
#     plot_explained_variance(pairs_train)
#     plot_explained_variance(pairs_test)

# feature_dim = pairs_train[0][0].numel()
# readout = Readout(input_size=feature_dim, num_classes=10, seed=seed)
# readout.train_readout(pairs_train, n_epochs=n_epochs)
# acc = readout.test_readout(pairs_test)
# print(f"Accuracy: {acc:.2f}%")

# fisher_J = calculate_fisher_ratio(pairs_test)
# print(f"Fisher ratio: {fisher_J:.4f}")

# plot_tsne(pairs_train, perplexity=30)
# plot_confusion_heatmap(pairs_train)
# plot_rsa_heatmap(pairs_train)

# Visualize MNIST sample and its Gabor feature maps for the chosen label
# import torch.nn.functional as F
# import random as _rnd

# _rnd.seed(seed)
# gabor_idx = _rnd.randrange(len(train_dataset))
# while train_dataset[gabor_idx]["label"] != target_label:
#     gabor_idx = _rnd.randrange(len(train_dataset))

# sample = train_dataset[gabor_idx]
# image = sample["image"]  # (1, 28, 28)

# gabor_input = image.unsqueeze(0)  # (1, 1, 28, 28)
# W_gabor = data_CNN.W_gabor
# feat = F.conv2d(gabor_input, W_gabor, bias=None, stride=1, padding=W_gabor.shape[-1] // 2)
# feat = torch.relu(feat).squeeze(0)  # (4, 28, 28)

# thetas = [r"$0°$", r"$45°$", r"$90°$", r"$135°$"]
# fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# axes[0].imshow(image.squeeze(0).cpu(), cmap="gray")
# axes[0].set_title(f"Original (label {target_label})")
# axes[0].axis("off")

# for k in range(4):
#     axes[k + 1].imshow(feat[k].cpu(), cmap="gray")
#     axes[k + 1].set_title(f"Gabor {thetas[k]}")
#     axes[k + 1].axis("off")

# plt.suptitle(f"MNIST label {target_label} — Gabor feature maps", fontsize=14)
# plt.tight_layout()
# plt.show(block=True)


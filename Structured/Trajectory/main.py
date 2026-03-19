import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'BindsNet'))

import random
import numpy as np
import torch

from framework import TrajectoryFramework
from data.input_data_CNN import Data as Data_CNN

# ── Parameters ──────────────────────────────────────────────────────
n_neurons = 2500
time = 100
dt = 1.0
intensity = 600
seed = 42
bin_ms = 3

g = 5
eta = 0.6
sigma_input = 1
sigma_network = 1
epsilon = 0.3

# ── Seed ────────────────────────────────────────────────────────────
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ── Data ────────────────────────────────────────────────────────────
data_CNN = Data_CNN(
    dt=dt, intensity=intensity, kernel_size=9,
    thetas_deg=(0, 45, 90, 135), convolution=False,
)
train_dataset, test_dataset = data_CNN.load_MNIST()

# ── Build network ───────────────────────────────────────────────────
tfw = TrajectoryFramework(
    bin_ms=bin_ms,
    n_neurons=n_neurons,
    time=time,
    dt=dt,
    seed=seed,
    heterogeneity=True,
    mnist_input=True,
    self_tuning=False,
    spatial=True,
    convolution=False,
    log_normal=False,
    g=g,
    eta=eta,
    sigma_input=sigma_input,
    sigma_network=sigma_network,
    epsilon=epsilon,
    intensity=intensity,
    stdp=False,
)

# ── Build class index ───────────────────────────────────────────────
class_indices = {}
for i in range(len(test_dataset)):
    label = test_dataset[i]["label"]
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(i)

# ── Single-sample trajectory ────────────────────────────────────────
target_digit = 3

idx = random.choice(class_indices[target_digit])
trajectory, label = tfw.extract_trajectory(test_dataset, idx)
print(f"Trajectory shape: {trajectory.shape}  "
      f"({trajectory.shape[0]} bins x {trajectory.shape[1]} neurons)")

#tfw.plot_trajectory_3d(trajectory, label=label)
#tfw.plot_trajectory_2d(trajectory, label=label)
#tfw.save_trajectory_3d_frames(trajectory, label=label)

# ── Multi-sample trajectories ──────────────────────────────────────
classes_to_show = [0, 1, 9, 8, 5]
samples_per_class = 3

print(f"\nMulti-trajectory: classes={classes_to_show}, "
      f"{samples_per_class} samples each")

trajectories = []
labels = []
for c in classes_to_show:
    for _ in range(samples_per_class):
        idx = random.choice(class_indices[c])
        traj, lbl = tfw.extract_trajectory(test_dataset, idx)
        trajectories.append(traj)
        labels.append(lbl)

tfw.plot_multi_trajectory_3d(trajectories, labels)
#tfw.plot_multi_trajectory_2d(trajectories, labels)
#tfw.save_multi_trajectory_3d_frames(trajectories, labels)

print("Done.")

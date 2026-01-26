from input_data import Data
from brunel import Brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch

# General parameters
n_neurons=1000
n_epochs=50
examples_train=250
examples_test=250
time=500
dt=1.0
intensity=430

# =============================== Data ===============================
data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)
train_dataset, test_dataset = data.load_MNIST()

#=============================== Brunel ===============================
brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt, mnist_input=True, self_tuning=False, eta=0.6, g=4.0)
brunel.build_brunel()
training_pairs, CV_list, rho_mean_list, rate_list, g_list, eta_list = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True)
test_pairs, CV_test_list, rho_mean_test_list, rate_test_list, g_test_list, eta_test_list = brunel.stimulate_brunel(test_dataset, examples=examples_test, shuffle=False)

print("mean CV: ", np.mean(CV_list))
print("mean rho_mean: ", np.mean(rho_mean_list))
print("mean rate: ", np.mean(rate_list))
print("mean g: ", np.mean(g_list))
print("mean eta: ", np.mean(eta_list))

feature_dim = training_pairs[0][0].numel()
readout = Readout(input_size=feature_dim, num_classes=10)
readout.train_readout(training_pairs, n_epochs=n_epochs)
acc = readout.test_readout(test_pairs)
print(f"Accuracy: {acc:.2f}%")


# # =============================== Data ===============================
# data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)
# train_dataset, test_dataset = data.load_MNIST()

# results_final = []

# from tqdm import tqdm

# gs = [3.0, 4.0, 5.0, 6.0]  # [3.0, 4.0, 5.0, 6.0, 7.0]
# etas = [0.6, 0.8, 1.0, 1.3]  # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# for eta in tqdm(etas, desc="Sweeping eta"):
#     for g in tqdm(gs, desc="Sweeping g"):

#         # =============================== Brunel ===============================
#         brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt, mnist_input=True, self_tuning=False, eta=eta, g=g)
#         brunel.build_brunel()

#         training_pairs, CV_list, rho_mean_list, rate_list, g_list, eta_list = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True)
#         test_pairs, _, _, _, _, _= brunel.stimulate_brunel(test_dataset, examples=examples_test, shuffle=False)

#         results = []

#         CV_mean = np.mean(CV_list)
#         results.append(CV_mean)

#         CV_std = np.std(CV_list)
#         results.append(CV_std)

#         rho_mean_mean = np.mean(rho_mean_list)
#         results.append(rho_mean_mean)

#         rho_mean_std = np.std(rho_mean_list)
#         results.append(rho_mean_std)

#         rate_mean = np.mean(rate_list)
#         results.append(rate_mean)

#         rate_std = np.std(rate_list)
#         results.append(rate_std)

#         g_mean = np.mean(g_list)
#         results.append(g_mean)

#         g_std = np.std(g_list)
#         results.append(g_std)

#         eta_mean = np.mean(eta_list)
#         results.append(eta_mean)

#         eta_std = np.std(eta_list)
#         results.append(eta_std)

#         # =============================== Readout ===============================

#         feature_dim = training_pairs[0][0].numel()
#         readout = Readout(input_size=feature_dim, num_classes=10)
#         readout.train_readout(training_pairs, n_epochs=n_epochs)
#         acc = readout.test_readout(test_pairs)
#         print(f"Accuracy: {acc:.2f}%")

#         results.append(acc)
#         results_final.append(results)



# print(results_final)


# import os
# results_final_array = np.array(results_final)
# if not os.path.exists("results/results_final.npy"):
#     os.makedirs("results/results_final.npy")
# np.save("results/results_final.npy", results_final_array)







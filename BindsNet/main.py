from input_data import Data
from reservoir import Reservoir
from reservoir_brunel import Reservoir as Reservoir_brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch




seed = 0
n_neurons=1000
n_epochs=500
examples=500
time=250
dt=1.0
intensity=64
conn_prob=0.1
w_e=1.0
g=4.0
w_ext=3.0
rate_ext=150.0
w_mnist=0.2
readout_from="E"

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)
train_dataset, test_dataset = data.load_MNIST()

res = Reservoir_brunel(n_neurons=n_neurons, time=time, dt=dt)
res.build_reservoir_brunel(conn_prob=conn_prob, w_e = w_e, g = g, w_ext = w_ext, rate_ext = rate_ext, w_mnist = w_mnist, readout_from = readout_from)



training_pairs = res.train_reservoir(train_dataset, examples=examples, shuffle=True)
test_pairs = res.test_reservoir(test_dataset, examples=examples, shuffle=False)

feature_dim = training_pairs[0][0].numel()
readout = Readout(input_size=feature_dim, num_classes=10)
readout.train_readout(training_pairs, n_epochs=n_epochs)

acc = readout.test_readout(test_pairs)
print(f"Accuracy: {acc:.2f}%")




custom_text = "I now have included STDP"
summary =  ""
summary += f"==================================\n"
summary += f"{custom_text}\n"
summary += f"Number of neurons: {n_neurons}\n"
summary += f"Number of epochs: {n_epochs}\n"
summary += f"Number of examples: {examples}\n"
summary += f"Time: {time}\n"
summary += f"dt: {dt}\n"
summary += f"Intensity: {intensity}\n"
summary += f"conn_prob: {conn_prob}\n"
summary += f"w_e: {w_e}\n"
summary += f"g: {g}\n"
summary += f"w_ext: {w_ext}\n"
summary += f"rate_ext: {rate_ext}\n"
summary += f"w_mnist: {w_mnist}\n"
summary += f"readout_from: {readout_from}\n"
summary += f"Accuracy: {acc:.2f}%\n"
summary += f"==================================\n"

with open("BindsNet/results/result.txt", "a") as f:
    f.write(summary)
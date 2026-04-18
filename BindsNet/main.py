from input_data import Data
from reservoir import Reservoir
from brunel import Brunel
from readout import Readout
from tqdm import tqdm
import numpy as np
import torch

from visualizer import *



seed = 0
n_neurons=800
n_epochs=100
examples_train=500
examples_test=500
time=250
dt=1.0
intensity=64



np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

data = Data(time=time, dt=dt, shuffle=True, intensity=intensity)
train_dataset, test_dataset = data.load_MNIST()


reservoir = Reservoir(n_neurons=n_neurons, time=time, dt=dt)
reservoir.build_reservoir()

training_pairs = reservoir.train_reservoir(train_dataset, examples=examples_train, shuffle=True)
test_pairs = reservoir.test_reservoir(test_dataset, examples=examples_test, shuffle=False)
#brunel = Brunel(n_neurons=n_neurons, time=time, dt=dt)
#brunel.build_brunel()

#training_pairs = brunel.stimulate_brunel(train_dataset, examples=examples_train, shuffle=True)
#test_pairs = brunel.stimulate_brunel(test_dataset, examples=examples_test, shuffle=False)


feature_dim = training_pairs[0][0].numel()

readout = Readout(input_size=feature_dim, num_classes=10)
readout.train_readout(training_pairs, n_epochs=n_epochs)
acc = readout.test_readout(test_pairs)
print(f"Accuracy: {acc:.2f}%")

# visualizer = Visualizer(training_pairs)
# visualizer.plot_tsne(perplexity=30)
# visualizer.plot_confusion_proximity() 



# custom_text = "I now have included STDP"
# summary =  ""
# summary += f"==================================\n"
# summary += f"{custom_text}\n"
# summary += f"Number of neurons: {n_neurons}\n"
# summary += f"Number of epochs: {n_epochs}\n"
# summary += f"Number of examples: {examples}\n"
# summary += f"Time: {time}\n"
# summary += f"dt: {dt}\n"
# summary += f"Intensity: {intensity}\n"
# summary += f"conn_prob: {conn_prob}\n"
# summary += f"w_e: {w_e}\n"
# summary += f"g: {g}\n"
# summary += f"w_ext: {w_ext}\n"
# summary += f"rate_ext: {rate_ext}\n"
# summary += f"w_mnist: {w_mnist}\n"
# summary += f"readout_from: {readout_from}\n"
# summary += f"Accuracy: {acc:.2f}%\n"
# summary += f"==================================\n"

# with open("BindsNet/results/result.txt", "a") as f:
#     f.write(summary)
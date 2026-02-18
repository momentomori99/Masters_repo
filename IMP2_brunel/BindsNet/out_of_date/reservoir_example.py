import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights


#parameters
seed=0
n_neurons=800
n_epochs=100
examples=500
n_workers=-1
time=250
dt=1.0
intensity=64
progress_interval=10
update_interval=600
plot = True
gpu = False
train = True
device = "cpu"

# Set the seed
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Creating a simple torch NN
network = Network(dt=dt)
inpt = Input(n=784, shape=(1, 28, 28))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, thresh=-52 + np.random.randn(n_neurons).astype(float))
network.add_layer(output, name="O")

C1 = Connection(source=inpt, target=output, w= 0.5 * torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w= 0.5 * torch.randn(output.n, output.n))

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

# Monitors for visualizing activity
spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time, device = device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltage = {"O": Monitor(network.layers["O"], ["v"], time=time, device = device)}
network.add_monitor(voltage["O"], name="O_voltage")


# Get MNIST training images and labels
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)


# Now training the network
print("Training the network")
n_iters = examples
training_pairs = []
pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i >= n_iters:
        break
    datum = dataPoint["encoded_image"].view(int(time/dt), 1, 1, 28, 28).to(device)
    print(datum.shape)
    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
    network.run(inputs={"I": datum}, time=time)
    training_pairs.append([spikes["O"].get("s"), label])
    network.reset_state_variables()
    break
    
    



#define logistic regression model 

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.linear_1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        return out



# Create and train logistic regression model on reservoir training outputs
model = NN(n_neurons * time, 10).to(device)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# Now training the logistic regression model
print("\n Training the read out")

pbar = tqdm(enumerate(range(n_epochs)))
for epoch, _ in pbar:
    avg_loss = 0

    # Extract spike outputs from reservoir for a training sample
    #       i   -> Loop index
    #       s   -> Reservoir output spikes
    #       l   -> Image label
    for i, (s, l) in enumerate(training_pairs):
        # Reset gradients to 0
        optimizer.zero_grad()

        # Run spikes through logistic regression model
        outputs = model(s)

        # Calculate MSE
        label = torch.zeros(1, 1, 10).float().to(device)
        label[0, 0, l] = 1.0
        loss = criterion(outputs.view(1, 1, -1), label)
        avg_loss += loss.data

        # Optimize parameters
        loss.backward()
        optimizer.step()

    pbar.set_description_str(
        "Epoch: %d/%d, Loss: %.4f"
        % (epoch + 1, n_epochs, avg_loss / len(training_pairs))
    )




# Now we do this on the test set
n_iters = examples
test_pairs = []
pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))

    network.run(inputs={"I": datum}, time=time)
    test_pairs.append([spikes["O"].get("s"), label])
    network.reset_state_variables()


print("\n Testing the read out")
# Test model with previously trained logistic regression classifier
correct, total = 0, 0
for s, label in test_pairs:
    outputs = model(s)
    _, predicted = torch.max(outputs.data.unsqueeze(0), 1)
    total += 1
    correct += int(predicted == label.long().to(device))

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (n_iters, 100 * correct / total)
)
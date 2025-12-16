import time

import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import numpy as np

nest.ResetKernel()

startbuild = time.time() 

# ---- Simulation parameters ----
dt = 0.1 # the resolution in ms 
simtime = 500.0 # Simulation time in ms
delay = 1.5  # synaptic delay in ms

# ---- Parameters crucial for AI firing regime in the neurons ----
g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

# ---- Number of neurons in the netowrk and the number of neurons recorded from ----
order = 300
NE = 4 * order  # number of excitatory neurons
NI = 1 * order  # number of inhibitory neurons
N_neurons = NE + NI  # number of neurons in total
N_rec = 50  # record from 50 neurons

# ---- Connectivity parameters ----
CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE)  # total number of synapses per neuron

# ---- LIF neuron parameters ----
tauMem = 20.0  # time constant of membrane potential in ms
theta = 20.0  # membrane threshold potential in mV
neuron_params = {"C_m": 1.0, "tau_m": tauMem, "t_ref": 2.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": theta}
J = 0.1  # postsynaptic amplitude in mV
J_ex = J  # amplitude of excitatory postsynaptic potential
J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential

# ---- Threshold rate, external firing rate and converted spikes per second ----
nu_th = theta / (J * CE * tauMem)
nu_ex = eta * nu_th
p_rate = 10.0 * nu_ex * CE


print("p_rate", p_rate)



nest.resolution = dt
nest.print_time = True
nest.overwrite_files = True

print("Building network")

nodes_ex = nest.Create("iaf_psc_alpha", NE, params=neuron_params)
nodes_in = nest.Create("iaf_psc_alpha", NI, params=neuron_params)
noise = nest.Create("poisson_generator", params={"rate": p_rate})
espikes = nest.Create("spike_recorder")
ispikes = nest.Create("spike_recorder")

print(espikes)

#Flash light
stim_on   = 200.0   # ms – when the flash starts
stim_off  = 260.0   # ms – when the flash ends
stim_rate = 10 * p_rate  # 5x stronger than background, tweak as you like


espikes.set(label="brunel-py-ex", record_to="ascii")
ispikes.set(label="brunel-py-in", record_to="ascii")

"""
Creating two excitatory synapse models + inhibitory.
We want one stattic excitatory synapse for noise to neurons. And one recoding connections-
One STDP excitatory synapse for recurrent E -> (E + I) connections.
Inhibitory synapse is static.
"""

#Static excitatory synapse (for Poisson input and spike recorders)
nest.CopyModel("static_synapse", "excitatory_bg", {"weight": J_ex, "delay": delay})


stim_neurons = nodes_ex[:50]               # "visual" subpopulation

#Flash light
stimulus_pg = nest.Create(
    "poisson_generator",
    params={
        "rate": stim_rate,
        "start": stim_on,
        "stop": stim_off,
    }
)

# Inhibitor synapes (static)
nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})

# STDP synapse for recurrent excitatory connections
nest.CopyModel(
    "stdp_synapse",           # built-in STDP model in NEST
    "excitatory_stdp",
    {
        "weight": J_ex,
        "delay": delay,
        "lambda": 0.01,       # learning rate
        "alpha": 1.0,         # target ratio of depression/potentiation
        "Wmax": 1.0,          # max weight (This should be tuned)
    },
)

# Connect Poisson input with static synapses - to make sure that the external drive is not plastic
nest.Connect(noise, nodes_ex, syn_spec="excitatory_bg")
nest.Connect(noise, nodes_in, syn_spec="excitatory_bg")

#Flash light
nest.Connect(stimulus_pg, stim_neurons, syn_spec="excitatory_bg")

# Connect spike recorders with static synapses
nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory_bg")
nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory_bg")
# Because these connectoon are just for monitoring, they should not be plastic

#Now, make recurrent excitatory connections STDP
print("Connecting network")
print("Excitatory connections")

conn_params_ex = {"rule": "fixed_indegree", "indegree": CE}
nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory_stdp")

print("Inhibitory connections")

conn_params_in = {"rule": "fixed_indegree", "indegree": CI}
nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

"""
Now, 
- all recurrent E -> (E + I) connections are plastic.
- all I -> (E + I) connections are static
- all external Poisson input is static
"""


# We want to add weight recorder to watch weight evolution
wr = nest.Create("weight_recorder")
nest.SetDefaults("excitatory_stdp", {"weight_recorder": wr[0]})


# Simulation
nest.Simulate(simtime)

# --- weight evolution analysis ---

w_events = wr.get("events")
times   = w_events["times"]
senders = w_events["senders"]
targets = w_events["targets"]
weights = w_events["weights"]


# -------------------------------------------------------------------
# Identify which neurons received the stimulus
# -------------------------------------------------------------------

# In your code, ALL excitatory neurons get the flash currently.
# But you probably want ONLY A SUBSET to get the flash.
# So we define:
stim_ids = set(stim_neurons.tolist())
mask_stim = np.array([t in stim_ids for t in targets])
mask_non  = ~mask_stim


# -------------------------------------------------------------------
# Helper: choose a few synapses to visualize (to avoid thousands of curves)
# -------------------------------------------------------------------
def pick_pairs(mask, n=40):
    # Select only synapses under the mask
    send_sub = senders[mask]
    targ_sub = targets[mask]
    pairs = list({(s, t) for s, t in zip(send_sub, targ_sub)})
    return pairs[:n]    # first n pairs

pairs_stim = pick_pairs(mask_stim, n=40)
pairs_non  = pick_pairs(mask_non,  n=40)

# -------------------------------------------------------------------
# PLOTTING: two subplots (stimulated vs non-stimulated)
# -------------------------------------------------------------------

plt.figure(figsize=(12, 5))

# --- Weights onto stimulated neurons ---
plt.subplot(1, 2, 1)
for s, t in pairs_stim:
    m = (senders == s) & (targets == t)
    plt.plot(times[m], weights[m], lw=0.8)

plt.axvspan(stim_on, stim_off, alpha=0.2, color='yellow')
plt.title("Synaptic weights → stimulated neurons")
plt.xlabel("time (ms)")
plt.ylabel("weight")

# --- Weights onto non-stimulated neurons ---
plt.subplot(1, 2, 2)
for s, t in pairs_non:
    m = (senders == s) & (targets == t)
    plt.plot(times[m], weights[m], lw=0.8)

plt.axvspan(stim_on, stim_off, alpha=0.2, color='yellow')
plt.title("Synaptic weights → non-stimulated neurons")
plt.xlabel("time (ms)")

plt.tight_layout()
plt.show()

import nest
import matplotlib.pyplot as plt

nest.ResetKernel()

# ---- Parameters ------
neuron_parameters = {
    "C_m": 250.0,       # membrane capacitance
    "tau_m": 10.0,      # membrane time constant
    "t_ref": 2.0,       # refractory period
    "E_L": -70.0,       # resting potential
    "V_reset": -70.0,   # initial membrane potential
    "V_th": -55.0,      # spiking threshold
    "I_e": 0.0,         # external constant current
    "tau_syn_ex": 0.2,  # excitatory synaptic time constant
    "tau_syn_in": 2.0,  # inhibitory synaptic time cosntant
}

noise_parameters = {
    "rate": 700000.0,    # rate
    "start": 400.0,     # start of the noise
    "stop": 600.0       # End of the noise
}

stdp_synapse = {
    "synapse_model": "stdp_synapse",
    "weight": 50.0,
    "delay":1.5
}

# --- Neurons ---
neuron1 = nest.Create("iaf_psc_alpha", neuron_parameters)
neuron2 = nest.Create("iaf_psc_alpha", neuron_parameters)

# Make neuron2 a bit more excitable so it actually spikes
neuron2.set(I_e=370.0)  
neuron1.set(I_e=330.0)

# --- Noise to neuron1 ---
noise = nest.Create("poisson_generator", noise_parameters)

# --- Recorders (optional but useful) ---
spike_rec1 = nest.Create("spike_recorder")
spike_rec2 = nest.Create("spike_recorder")

multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"])

# --- Connect everything ---

# Noise drives neuron1
nest.Connect(noise, neuron1, syn_spec={"weight": 2.0})

# STDP synapse between neuron1 and neuron2
conn_dict = {"rule": "one_to_one"}
nest.Connect(neuron1, neuron2, conn_dict, stdp_synapse)

# Record spikes
nest.Connect(neuron1, spike_rec1)
nest.Connect(neuron2, spike_rec2)

# Record membrane potentials
nest.Connect(multimeter, neuron1)
nest.Connect(multimeter, neuron2)

# --- Get connection handle for weight tracking ---
conn = nest.GetConnections(neuron1, neuron2)
# This should be a ConnectionCollection of length 1

# --- Simulate in steps and record weight ---
T_total = 1000.0  # total time in ms
dt = 5.0          # step size in ms
n_steps = int(T_total / dt)

times = []
weights = []

for i in range(n_steps):
    nest.Simulate(dt)
    # conn.weight is a numpy array (here of length 1)
    weights.append(conn.weight)
    times.append((i + 1) * dt)

# --- Prepare data for plotting ---

# Define consistent colors for neuron1 and neuron2
color1 = "#1f77b4"  # blue
color2 = "#d62728"  # red

dmm = multimeter.get()
events = dmm["events"]
Vms = events["V_m"]
ts = events["times"]
senders = events["senders"]

V1 = Vms[senders == neuron1[0]]
t1 = ts[senders == neuron1[0]]
V2 = Vms[senders == neuron2[0]]
t2 = ts[senders == neuron2[0]]

spike_events_1 = spike_rec1.events
spike_events_2 = spike_rec2.events
spike_times_1 = spike_events_1["times"]
spike_times_2 = spike_events_2["times"]

fig, axs = plt.subplots(
    3, 1, 
    sharex=True, 
    figsize=(9, 8), 
    gridspec_kw={'height_ratios': [1, 0.3, 1]}  # Membrane potentials, more narrow spike raster, then weight
)

# --- Top plot: membrane potentials ---
axs[0].plot(t1, V1, label="neuron1 V_m", color=color1)
axs[0].plot(t2, V2, label="neuron2 V_m", color=color2)
axs[0].legend()
axs[0].set_ylabel("V_m (mV)")
axs[0].set_title("Membrane potentials")

# --- Middle plot: spike raster (narrower) ---
axs[1].plot(spike_times_1, [0] * len(spike_times_1), '|', color=color1, markersize=10, label="neuron1 spike")
axs[1].plot(spike_times_2, [1] * len(spike_times_2), '|', color=color2, markersize=10, label="neuron2 spike")
axs[1].set_yticks([0, 1])
axs[1].set_yticklabels(["neuron1", "neuron2"])
axs[1].set_ylabel("Spike")
axs[1].set_title("Spike times")

# --- Bottom plot: weight evolution ---
axs[2].plot(times, weights, color=color1)
# Add vertical dashed lines for spike times, using respective colors
for i, st in enumerate(spike_times_1):
    axs[2].axvline(st, color=color1, linestyle='--', linewidth=1, alpha=0.7, label='neuron1 spike' if i == 0 else None)
for i, st in enumerate(spike_times_2):
    axs[2].axvline(st, color=color2, linestyle='--', linewidth=1, alpha=0.7, label='neuron2 spike' if i == 0 else None)

# Only add legends for the first instance
handles, labels = axs[2].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
if by_label:
    axs[2].legend(by_label.values(), by_label.keys())

axs[2].set_xlabel("time (ms)")
axs[2].set_ylabel("synaptic weight")
axs[2].set_title("STDP weight evolution")

plt.tight_layout()
plt.show()

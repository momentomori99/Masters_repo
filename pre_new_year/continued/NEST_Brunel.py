import time

import matplotlib.pyplot as plt
import nest
import nest.raster_plot

nest.ResetKernel()

startbuild = time.time() 

# ---- Simulation parameters ----
dt = 0.1 # the resolution in ms 
simtime = 1000.0 # Simulation time in ms
delay = 1.5  # synaptic delay in ms

# ---- Parameters crucial for AI firing regime in the neurons ----
g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

# ---- Number of neurons in the netowrk and the number of neurons recorded from ----
order = 2500
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
p_rate = 1000.0 * nu_ex * CE



nest.resolution = dt
nest.print_time = True
nest.overwrite_files = True

print("Building network")

nodes_ex = nest.Create("iaf_psc_alpha", NE, params=neuron_params)
nodes_in = nest.Create("iaf_psc_alpha", NI, params=neuron_params)
noise = nest.Create("poisson_generator", params={"rate": p_rate})
espikes = nest.Create("spike_recorder")
ispikes = nest.Create("spike_recorder")

espikes.set(label="brunel-py-ex", record_to="ascii")
ispikes.set(label="brunel-py-in", record_to="ascii")

print("Connecting devices")

nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})

nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory")

nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory")

print("Connecting network")

print("Excitatory connections")

conn_params_ex = {"rule": "fixed_indegree", "indegree": CE}
nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")

print("Inhibitory connections")

conn_params_in = {"rule": "fixed_indegree", "indegree": CI}
nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

endbuild = time.time()


print("Simulating")

nest.Simulate(simtime)

endsimulate = time.time()

events_ex = espikes.n_events
events_in = ispikes.n_events

rate_ex = events_ex / simtime * 1000.0 / N_rec
rate_in = events_in / simtime * 1000.0 / N_rec

num_synapses_ex = nest.GetDefaults("excitatory")["num_connections"]
num_synapses_in = nest.GetDefaults("inhibitory")["num_connections"]
num_synapses = num_synapses_ex + num_synapses_in

build_time = endbuild - startbuild
sim_time = endsimulate - endbuild

print("Brunel network simulation (Python)")
print(f"Number of neurons : {N_neurons}")
print(f"Number of synapses: {num_synapses}")
print(f"       Excitatory : {num_synapses_ex}")
print(f"       Inhibitory : {num_synapses_in}")
print(f"Excitatory rate   : {rate_ex:.2f} spks/s")
print(f"Inhibitory rate   : {rate_in:.2f} spks/s")
print(f"Building time     : {build_time:.2f} s")
print(f"Simulation time   : {sim_time:.2f} s")

nest.raster_plot.from_device(espikes, hist=True)
plt.show()
import nest
import matplotlib.pyplot as plt
import numpy as np


nest.ResetKernel()
nest.set_verbosity("M_FATAL")

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
static_synapse = {
    "synapse_model": "static_synapse",
    "weight": 50.0,
    "delay": 1.5
}

# --- Neurons ---
neuron_i = nest.Create("iaf_psc_alpha", neuron_parameters)
neuron_1 = nest.Create("iaf_psc_alpha", neuron_parameters)
neuron_2 = nest.Create("iaf_psc_alpha", neuron_parameters)
neuron_3 = nest.Create("iaf_psc_alpha", neuron_parameters)


# Make neuron2 a bit more excitable so it actually spikes
neuron_i.set(I_e=380.0)  
neuron_1.set(I_e=380.0)
neuron_2.set(I_e=380.0)
neuron_3.set(I_e=380.0)


# --- Noise to neurons ---
noise = nest.Create("poisson_generator", noise_parameters)

# --- Recorders (optional but useful) ---
spike_rec_i = nest.Create("spike_recorder")
spike_rec_1 = nest.Create("spike_recorder")
spike_rec_2 = nest.Create("spike_recorder")
spike_rec_3 = nest.Create("spike_recorder")

multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"])




# --- Connect everything ---

# Noise drives neuron1
#           nest.Connect(noise, neuron_1, syn_spec={"weight": 2.0})

# STDP synapse between neuron1 and neuron2
conn_dict = {"rule": "one_to_one"}
nest.Connect(neuron_1, neuron_i, conn_dict, static_synapse)
nest.Connect(neuron_2, neuron_i, conn_dict, static_synapse)
nest.Connect(neuron_3, neuron_i, conn_dict, static_synapse)

# Record spikes
nest.Connect(neuron_i, spike_rec_i)
nest.Connect(neuron_1, spike_rec_1)
nest.Connect(neuron_2, spike_rec_2)
nest.Connect(neuron_3, spike_rec_3)

# Record membrane potentials
nest.Connect(multimeter, neuron_i)
nest.Connect(multimeter, neuron_1)
nest.Connect(multimeter, neuron_2)
nest.Connect(multimeter, neuron_3)

# Helper functions
def spikes(s, t_prev, t_now, S):
    sp_times = s["times"]
    spiked = int(np.any((sp_times > t_prev) & (sp_times <= t_now)))
    S.append(spiked)
    return spiked

def H_nmda(u, alpha=0.062, beta=3.57):
    return 1/ (1 + alpha * np.exp(-beta*u))



# --- Simulate in steps and record weight ---
T_total = 1000.0  # total time in ms
dt = 5          # step size in ms
n_steps = int(T_total / dt)


tau_nmda = 100
tau_E = 50
E_nmda = 0.0
w_e = 0.5

E1_tilde, E2_tilde, E3_tilde = 0.0, 0.0, 0.0
E1_list, E2_list, E3_list, Ei_list = [], [], [], []
g_nmda_i, g_nmda_1, g_nmda_2, g_nmda_3 = [], [], [], []
times = []
U_i, U_1, U_2, U_3 = [], [], [], []
S_i, S_1, S_2, S_3 = [], [], [], []
g_i, g_1, g_2, g_3 = 0.0, 0.0, 0.0, 0.0

for step in range(n_steps):
    nest.Simulate(dt)
    t_prev = step * dt 
    t_now = (step + 1) * dt

    mm = multimeter.events
    V = mm["V_m"]
    T = mm["times"]
    G = mm["senders"]

    # membrane potential recording
    u_i = V[G == neuron_i[0]][-1]
    U_i.append(u_i)

    # spike detection
    # neuron i
    s_i = spike_rec_i.events
    spiked_i = spikes(s_i, t_prev, t_now, S_i)
    
    # neuron 1
    s_1 = spike_rec_1.events
    spiked_1 = spikes(s_1, t_prev, t_now, S_1)

    # neuron 2
    s_2 = spike_rec_2.events
    spiked_2 = spikes(s_2, t_prev, t_now, S_2)

    # neuron 3
    s_3 = spike_rec_3.events
    spiked_3 = spikes(s_3, t_prev, t_now, S_3)

    # conductance
    #g_i += dt * (-g_i / tau_nmda + w_e * spiked_i)
    #g_nmda_i.append(g_i)
    g_1 += dt * (-g_1 / tau_nmda + w_e * spiked_1)
    g_nmda_1.append(g_1)
    g_2 += dt * (-g_2 / tau_nmda + w_e * spiked_2)
    g_nmda_2.append(g_2)
    g_3 += dt * (-g_3 / tau_nmda + w_e * spiked_3)
    g_nmda_3.append(g_3)

    H_val = H_nmda(u_i)

    E1_tilde += dt * ( -E1_tilde / tau_E - g_1 * H_val * (u_i - E_nmda) )
    E2_tilde += dt * ( -E2_tilde / tau_E - g_2 * H_val * (u_i - E_nmda) )
    E3_tilde += dt * ( -E3_tilde / tau_E - g_3 * H_val * (u_i - E_nmda) )

    E1_list.append(E1_tilde)
    E2_list.append(E2_tilde)
    E3_list.append(E3_tilde)

    E_i = E1_tilde + E2_tilde + E3_tilde
    Ei_list.append(E_i)



   

    

    

    

    
    

    
    times.append((step + 1) * dt)


#plt.plot(times, g_nmda_1)
# plt.plot(times, E1_list)
# plt.plot(times, E2_list)
# plt.plot(times, E3_list)
plt.plot(times, Ei_list)
plt.show()
# --- Prepare data for plotting ---

# # Define consistent colors for neuron1 and neuron2
# color1 = "#1f77b4"  # blue
# color2 = "#d62728"  # red

# dmm = multimeter.get()
# events = dmm["events"]
# Vms = events["V_m"]
# ts = events["times"]
# senders = events["senders"]

# V1 = Vms[senders == neuron1[0]]
# t1 = ts[senders == neuron1[0]]
# V2 = Vms[senders == neuron2[0]]
# t2 = ts[senders == neuron2[0]]

# spike_events_1 = spike_rec1.events
# spike_events_2 = spike_rec2.events
# spike_times_1 = spike_events_1["times"]
# spike_times_2 = spike_events_2["times"]



import nest
import matplotlib.pyplot as plt
import numpy as np

nest.ResetKernel()

# ---------- Parametere ----------
neuron_parameters = {
    "C_m": 250.0,
    "tau_m": 10.0,
    "t_ref": 2.0,
    "E_L": -70.0,
    "V_reset": -70.0,
    "V_m": -70.0,
    "V_th": -55.0,
    "I_e": 0.0,
    "tau_syn_ex": 0.2,
    "tau_syn_in": 2.0,
}

noise_parameters = {
    "rate": 700000.0,   # juster som du vil
    "start": 400.0,
    "stop": 600.0,
}

stdp_cluster = {
    "synapse_model": "stdp_synapse",
    "weight": 70.0,
    "delay": 1.5,
}

stdp_pair = {
    "synapse_model": "stdp_synapse",
    "weight": 60.0,
    "delay": 1.5,
}

# ---------- Populasjoner ----------
# 2 clusters with 10 neurons each
cluster1 = nest.Create("iaf_psc_alpha", 10, params={**neuron_parameters, "I_e": 0.0})
cluster2 = nest.Create("iaf_psc_alpha", 10, params={**neuron_parameters, "I_e": 0.0})

# Nevron I og II (litt mer excitable)
neuron_I = nest.Create("iaf_psc_alpha", params={**neuron_parameters, "I_e": 360.0})
neuron_II = nest.Create("iaf_psc_alpha", params={**neuron_parameters, "I_e": 360.0})

# Felles Poisson-støy
noise1 = nest.Create("poisson_generator", params=noise_parameters)
noise2 = nest.Create("poisson_generator", params=noise_parameters)

# ---------- Koblinger ----------
# noise -> cluster (samme støy til alle)
nest.Connect(noise1, cluster1, syn_spec={"weight": 1.0})
nest.Connect(noise2, cluster2, syn_spec={"weight": 1.0})

# cluster 1 -> neuron I (10→1, STDP)
conn_spec_cluster = {"rule": "all_to_all"}
nest.Connect(cluster1, neuron_I, conn_spec_cluster, stdp_cluster)

# cluster 2 -> neuron II (10→1, STDP)
nest.Connect(cluster2, neuron_II, conn_spec_cluster, stdp_cluster)

# neuron I <-> neuron II (1↔1, STDP)
nest.Connect(neuron_I, neuron_II, syn_spec=stdp_pair)
nest.Connect(neuron_II, neuron_I, syn_spec=stdp_pair)

# ---------- Multimeter: alle potensialer ----------
multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"], interval=0.1)

# Alle nevroner vi vil måle fra
all_neurons = cluster1 + cluster2 + neuron_I + neuron_II
nest.Connect(multimeter, all_neurons)

# ---------- Hent synapser for vektlogging ----------
conns_1I = nest.GetConnections(cluster1, neuron_I)     # 10 stk
conns_2II = nest.GetConnections(cluster2, neuron_II)   # 10 stk
conns_I_II = nest.GetConnections(neuron_I, neuron_II)  # 1 stk
conns_II_I = nest.GetConnections(neuron_II, neuron_I)  # 1 stk

# ---------- Simuler i steg og logg vekter ----------
T_total = 1000.0
dt = 5.0
n_steps = int(T_total / dt)

times = []
w_1I = []
w_2II = []
w_I_II = []
w_II_I = []

for i in range(n_steps):
    nest.Simulate(dt)
    times.append((i + 1) * dt)

    w_1I.append(conns_1I.weight.copy())
    w_2II.append(conns_2II.weight.copy())
    w_I_II.append(conns_I_II.weight)
    w_II_I.append(conns_II_I.weight)

times = np.array(times)
w_1I = np.vstack(w_1I)    # (n_steps, 10)
w_2II = np.vstack(w_2II)  # (n_steps, 10)
w_I_II = np.array(w_I_II)
w_II_I = np.array(w_II_I)

# ---------- Hent multimeter-data ----------
dmm = multimeter.get()
events = dmm["events"]
Vms = events["V_m"]
ts = events["times"]
senders = events["senders"]

def extract_trace(gid):
    mask = (senders == gid)
    return ts[mask], Vms[mask]

# ---------- FIGUR 1: potensialer i 3 subplots ----------
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# ØVERST: alle nevroner fra cluster1
for i, gid in enumerate(cluster1):
    t, V = extract_trace(gid)
    axes[0].plot(t, V, label=f"C1_{i}")
axes[0].set_title("Cluster 1 membrane potentials")
axes[0].set_ylabel("V_m (mV)")
axes[0].legend(loc="upper right", ncol=2, fontsize=8)

# MIDTEN: alle nevroner fra cluster2
for i, gid in enumerate(cluster2):
    t, V = extract_trace(gid)
    axes[1].plot(t, V, label=f"C2_{i}")
axes[1].set_title("Cluster 2 membrane potentials")
axes[1].set_ylabel("V_m (mV)")
axes[1].legend(loc="upper right", ncol=2, fontsize=8)

# NEDERST: neuron I og neuron II
for label, gid in zip(["I", "II"], [neuron_I[0], neuron_II[0]]):
    t, V = extract_trace(gid)
    axes[2].plot(t, V, label=f"Neuron {label}")
axes[2].set_title("Neuron I and II membrane potentials")
axes[2].set_xlabel("time (ms)")
axes[2].set_ylabel("V_m (mV)")
axes[2].legend(loc="upper right")

plt.tight_layout()
plt.show()

# ---------- FIGUR 2: vektutvikling ----------
plt.figure(figsize=(10, 5))
plt.plot(times, w_1I.mean(axis=1), label="mean C1 → I")
plt.plot(times, w_2II.mean(axis=1), label="mean C2 → II")
plt.plot(times, w_I_II, label="I → II")
plt.plot(times, w_II_I, label="II → I")
plt.xlabel("time (ms)")
plt.ylabel("synaptic weight")
plt.title("STDP weight evolution")
plt.legend()
plt.tight_layout()
plt.show()

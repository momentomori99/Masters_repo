import time

import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import numpy as np
import scipy.special as sp


# -----------------------------
# Helpers
# -----------------------------
def _to_id_set(neuron_ids):
    """Convert NEST NodeCollection / iterable into a set of integer GIDs."""
    if neuron_ids is None:
        return None

    # NEST NodeCollection often supports .tolist()
    if hasattr(neuron_ids, "tolist"):
        return set(int(x) for x in neuron_ids.tolist())

    # Some installs allow list(NodeCollection) -> list of gids
    try:
        return set(int(x) for x in list(neuron_ids))
    except TypeError:
        pass

    # If it's a single object, last resort
    return {int(neuron_ids)}


def compute_cv_from_spike_recorder(spike_recorder, min_isis=2, neuron_ids=None):
    """
    Compute CV of ISIs per neuron from a NEST spike_recorder.
    """
    ev = nest.GetStatus(spike_recorder, "events")[0]
    times = np.asarray(ev.get("times", []), dtype=float)   # ms
    senders = np.asarray(ev.get("senders", []), dtype=int)

    if times.size == 0:
        return {}, np.nan

    id_set = _to_id_set(neuron_ids)
    if id_set is not None:
        mask = np.isin(senders, list(id_set))
        times = times[mask]
        senders = senders[mask]
        if times.size == 0:
            return {}, np.nan

    cv_per_neuron = {}
    for nid in np.unique(senders):
        t = times[senders == nid]
        t.sort()
        if t.size < 3:
            continue

        isis = np.diff(t)
        if isis.size < min_isis:
            continue

        mu = isis.mean()
        if mu <= 0:
            continue

        cv_per_neuron[int(nid)] = float(isis.std(ddof=0) / mu)

    if not cv_per_neuron:
        return {}, np.nan

    return cv_per_neuron, float(np.mean(list(cv_per_neuron.values())))

def LambertWm1(x):
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = tauMem / tauSyn
    b = 1.0 / tauSyn - 1.0 / tauMem

    # time of maximum
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)

    # maximum of PSP for current of unit amplitude
    return (
        np.exp(1.0)
        / (tauSyn * CMem * b)
        * ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b - t_max * np.exp(-t_max / tauSyn))
    )


# -----------------------------
# Main
# -----------------------------
nest.ResetKernel()

startbuild = time.time()

# ---- Simulation parameters ----
dt = 0.1          # ms
simtime = 1000.0  # ms
delay = 5.0       # ms

# ---- Network parameters ----
g = 15.0        # inhibitory/excitatory weight ratio
eta = 2.0      # external rate relative to threshold rate
epsilon = 0.1  # connection probability

order = 2500
NE = 4 * order
NI = 1 * order
N_neurons = NE + NI
N_rec = 50

CE = int(epsilon * NE)
CI = int(epsilon * NI)

# ---- Neuron/synapse parameters ----
tauSyn = 0.5
tauMem = 20.0
CMem = 250.0
theta = 20.0

neuron_params = {
    "C_m": CMem,
    "tau_m": tauMem,
    "tau_syn_ex": tauSyn,
    "tau_syn_in": tauSyn,
    "t_ref": 2.0,
    "E_L": 0.0,
    "V_reset": 0.0,
    "V_m": 0.0,
    "V_th": theta,
}

J = 0.1  # mV PSP amplitude target
J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J_ex = J / J_unit
J_in = -g * J_ex

# External drive
nu_th = (theta * CMem) / (J_ex * CE * np.exp(1.0) * tauMem * tauSyn)
nu_ex = eta * nu_th
p_rate = 1000.0 * nu_ex * CE  # Hz

# ---- Kernel settings ----
nest.resolution = dt
nest.print_time = True
nest.overwrite_files = True

print("Building network")

# ---- Create nodes ----
nodes_ex = nest.Create("iaf_psc_alpha", NE, params=neuron_params)
nodes_in = nest.Create("iaf_psc_alpha", NI, params=neuron_params)

noise = nest.Create("poisson_generator", params={"rate": float(p_rate)})

# IMPORTANT: use memory so CV + raster_plot can read events
espikes = nest.Create("spike_recorder")
ispikes = nest.Create("spike_recorder")
espikes.set(label="brunel-py-ex", record_to="memory")
ispikes.set(label="brunel-py-in", record_to="memory")

print("Connecting devices")

# ---- Synapse models ----
nest.CopyModel("static_synapse", "excitatory", {"weight": float(J_ex), "delay": float(delay)})
nest.CopyModel("static_synapse", "inhibitory", {"weight": float(J_in), "delay": float(delay)})

# ---- External input ----
nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory")

# ---- Record from a subset ----
nest.Connect(nodes_ex[:N_rec], espikes)
nest.Connect(nodes_in[:N_rec], ispikes)

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

# ---- CV calculation ----
cv_ex_dict, cv_ex_mean = compute_cv_from_spike_recorder(espikes, neuron_ids=nodes_ex[:N_rec])
cv_in_dict, cv_in_mean = compute_cv_from_spike_recorder(ispikes, neuron_ids=nodes_in[:N_rec])

print(f"Mean CV (E, {len(cv_ex_dict)} neurons): {cv_ex_mean}")
print(f"Mean CV (I, {len(cv_in_dict)} neurons): {cv_in_mean}")

# ---- Rates ----
events_ex = int(espikes.n_events)
events_in = int(ispikes.n_events)
rate_ex = events_ex / simtime * 1000.0 / N_rec
rate_in = events_in / simtime * 1000.0 / N_rec

# ---- Synapse counts ----
num_synapses_ex = nest.GetDefaults("excitatory")["num_connections"]
num_synapses_in = nest.GetDefaults("inhibitory")["num_connections"]
num_synapses = num_synapses_ex + num_synapses_in

# ---- Timing ----
build_time = endbuild - startbuild
sim_time = endsimulate - endbuild

print("\nBrunel network simulation (Python)")
print(f"Number of neurons : {N_neurons}")
print(f"Number of synapses: {num_synapses}")
print(f"       Excitatory : {num_synapses_ex}")
print(f"       Inhibitory : {num_synapses_in}")
print(f"Excitatory rate   : {rate_ex:.2f} Hz")
print(f"Inhibitory rate   : {rate_in:.2f} Hz")
print(f"Building time     : {build_time:.2f} s")
print(f"Simulation time   : {sim_time:.2f} s")

# ---- Raster plot ----
if events_ex == 0 and events_in == 0:
    print("WARNING: No spikes recorded. Try increasing eta, simtime, or check parameters.")
else:
    nest.raster_plot.from_device(espikes, hist=True)
    plt.show()

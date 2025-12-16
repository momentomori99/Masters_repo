import nest
import numpy as np
import matplotlib.pyplot as plt

dt = 0.1 
simtime = 1000.0 

n_exc = 2 
n_inh = 1

rate_exc_input = 10*8000.0
rate_inh_input = 7000.0

w_exc = 1.0
w_inh = -2.0

nest.ResetKernel()
nest.SetKernelStatus({"resolution": dt})

exc_neurons = nest.Create("iaf_psc_alpha", n_exc)
inh_neurons = nest.Create("iaf_psc_alpha", n_inh)

all_neurons = exc_neurons + inh_neurons

pg_exc = nest.Create("poisson_generator", params={"rate": rate_exc_input})
pg_inh = nest.Create("poisson_generator", params={"rate": rate_inh_input})


# Connecting the nodes
nest.Connect(pg_exc, all_neurons, syn_spec={"weight": w_exc})
#nest.Connect(pg_inh, all_neurons, syn_spec={"weight": w_inh})

# E → all
nest.Connect(exc_neurons, all_neurons, syn_spec={"weight": w_exc})
# I → all
nest.Connect(inh_neurons, all_neurons, syn_spec={"weight": w_inh})

# Recorders, spikes and V_m
# One spike recorder for all neurons
spike_rec = nest.Create("spike_recorder")
nest.Connect(all_neurons, spike_rec)

# Multimeter to record membrane potential (V_m)
# For now we record from ALL neurons 
multimeter = nest.Create("multimeter", params={
    "interval": dt,
    "record_from": ["V_m"]
})

nest.Connect(multimeter, all_neurons)

nest.Simulate(simtime)


spike_events = spike_rec.events
spike_times = np.array(spike_events["times"])
spike_senders = np.array(spike_events["senders"])

#print("Total spikes recorded:", len(spike_times))
#print("Neurons that spiked (GIDs):", np.unique(spike_senders))

exc_gids = list(exc_neurons)
inh_gids = list(inh_neurons)

spike_times_exc = []
for gid in exc_gids:
    times_gid = spike_times[spike_senders == gid]
    spike_times_exc.append(times_gid)
print("----")
print(spike_times_exc)
print("----")

spike_times_inh = []
for gid in inh_gids:
    times_gid = spike_times[spike_senders == gid]
    spike_times_inh.append(times_gid)

print("Exc presynaptic neurons (GIDs):", exc_gids)
for i, times in enumerate(spike_times_exc):
    print(f"  Exc neuron {exc_gids[i]} fired {len(times)} spikes")

print("Inh presynaptic neurons (GIDs):", inh_gids)
for i, times in enumerate(spike_times_inh):
    print(f"  Inh neuron {inh_gids[i]} fired {len(times)} spikes")
# -----------------------------
# Read out membrane potential
# -----------------------------
mm_events = multimeter.events
Vm_all = np.array(mm_events["V_m"])
t_all = np.array(mm_events["times"])
senders_vm = np.array(mm_events["senders"])

# We choose one postsynaptic neuron to study, e.g. the first excitatory neuron
target_gid = exc_neurons[0]
print("Target postsynaptic neuron GID:", target_gid)

# Mask out V_m for this neuron only
mask_vm = (senders_vm == target_gid)
t_vm = t_all[mask_vm]
Vm = Vm_all[mask_vm]

# t_vm is our time grid for this neuron.
# It should be regularly spaced with step dt.
print("Length of V_m trace:", len(Vm))
print("First few time points:", t_vm[:5])



plt.figure(figsize=(8, 4))
plt.plot(t_vm, Vm)
plt.xlabel("Time (ms)")
plt.ylabel("V_m (mV)")
plt.title(f"Membrane potential of neuron gid={target_gid}")
plt.tight_layout()
#plt.show()

# Optional: simple raster plot of spikes
# plt.figure(figsize=(8, 3))
# plt.scatter(spike_times, spike_senders, s=2)
# plt.xlabel("Time (ms)")
# plt.ylabel("Neuron GID")
# plt.title("Spike raster")
# plt.tight_layout()
# plt.show()



def H_nmda(u, alpha=0.062, beta=3.57):
    return 1.0 / (1.0 + alpha*np.exp(-beta*u))

# -----------------------------
# Helper: bin spike times into counts per time bin
# -----------------------------
def bin_spikes(spike_times_list, t_grid, dt):
    """
    spike_times_list: list of 1D arrays, spike times for each synapse/neuron
    t_grid: 1D array of time points (we use t_vm from the multimeter)
    dt: bin width

    Returns:
      spikes[T, n_syn]: spikes[i, j] = number of spikes of synapse j in bin i
    """
    T = len(t_grid)
    n_syn = len(spike_times_list)
    spikes = np.zeros((T, n_syn), dtype=float)

    # Bin edges: [t0, t1, ..., t_last, t_last + dt]
    bin_edges = np.concatenate([t_grid, [t_grid[-1] + dt]])

    for j, times in enumerate(spike_times_list):
        if len(times) == 0:
            continue
        counts, _ = np.histogram(times, bins=bin_edges)
        spikes[:, j] = counts

    return spikes

# Use t_vm as our time axis for reconstruction
t = t_vm.copy()
u = Vm.copy()

# Bin the spikes of presynaptic excitatory and inhibitory neurons
spikes_exc_binned = bin_spikes(spike_times_exc, t, dt)  # shape: (T, n_exc)
spikes_inh_binned = bin_spikes(spike_times_inh, t, dt)  # shape: (T, n_inh)

print("Binned spike array shapes:")
print("  Exc:", spikes_exc_binned.shape)
print("  Inh:", spikes_inh_binned.shape)


# -----------------------------
# Reconstruct NMDA / GABAA and E(t), I(t)
# -----------------------------

T = len(t)                   # number of time points
n_exc_pre = len(spike_times_exc)
n_inh_pre = len(spike_times_inh)

# Synaptic parameters (you can tune these)
tau_nmda = 100.0   # ms
tau_E = 50.0       # ms
tau_gabaa = 10.0   # ms
tau_I = 20.0       # ms

E_NMDA = 0.0       # mV (example)
E_GABAA = -70.0    # mV (example)

# For now, set all recurrent weights to same values as NEST connections
w_e = np.ones(n_exc_pre) * w_exc
w_i = np.ones(n_inh_pre) * abs(w_inh)  # magnitude for GABA (we'll keep sign in current)

# State variables over time
g_nmda = np.zeros((T, n_exc_pre))    # g_NMDA_j(t)
E_tilde = np.zeros((T, n_exc_pre))   # tilde E_j(t)
E_global = np.zeros(T)               # E(t) = sum_j tilde E_j(t)

g_gabaa = np.zeros((T, n_inh_pre))   # g_GABAA_k(t)
I_trace = np.zeros(T)                # I(t)

# Initial conditions at time index 0
g_nmda_t = np.zeros(n_exc_pre)
E_tilde_t = np.zeros(n_exc_pre)

g_gabaa_t = np.zeros(n_inh_pre)
I_t = 0.0

# Euler integration loop
for i in range(1, T):
    u_prev = u[i-1]  # membrane potential at previous time step

    # ---- Excitatory side (NMDA) ----
    # Update NMDA conductances for each excitatory presynaptic neuron
    # dg/dt = -g/tau_nmda + w_e * S_exc
    g_nmda_t += dt * (
        -g_nmda_t / tau_nmda
        + w_e * spikes_exc_binned[i-1]
    )

    # Update tilde E_j
    # tau_E dE_tilde/dt = -E_tilde - g_NMDA * H(u) * (u - E_NMDA)
    H_val = H_nmda(u_prev)  # same for all j (depends only on u)
    dE_tilde = dt * (
        (-E_tilde_t / tau_E)
        - g_nmda_t * H_val * (u_prev - E_NMDA)
    )
    E_tilde_t += dE_tilde

    # Store current values
    g_nmda[i] = g_nmda_t
    E_tilde[i] = E_tilde_t
    E_global[i] = np.sum(E_tilde_t)

    # ---- Inhibitory side (GABAA) ----
    # dg_GABAA/dt = -g_GABAA/tau_gabaa + w_i * S_inh
    g_gabaa_t += dt * (
        -g_gabaa_t / tau_gabaa
        + w_i * spikes_inh_binned[i-1]
    )

    # Inhibitory current sum_k g_GABAA_k(t) * (u - E_GABAA)
    total_inh_current = np.sum(g_gabaa_t * (u_prev - E_GABAA))

    # tau_I dI/dt = -I + total_inh_current
    dI = dt * (
        -I_t / tau_I
        + total_inh_current
    )
    I_t += dI

    # Store
    g_gabaa[i] = g_gabaa_t
    I_trace[i] = I_t


# -----------------------------
# Plot E(t) and I(t) for the target neuron
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(t, E_global, label="E(t) (global excitatory trace)")
plt.plot(t, I_trace, label="I(t) (global inhibitory trace)")
plt.xlabel("Time (ms)")
plt.ylabel("Trace value (arbitrary units)")
plt.title(f"E(t) and I(t) for postsynaptic neuron gid={target_gid}")
plt.legend()
plt.tight_layout()
plt.show()


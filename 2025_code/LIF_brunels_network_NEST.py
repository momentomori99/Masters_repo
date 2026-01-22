import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from numpy import exp 

# ============================================================================
# Iris dataset 
USE_IRIS_INPUT = True   

# Load Iris
iris = load_iris()
X = iris.data
y = iris.target

# Normalize features
scaler = MinMaxScaler()
Xn = scaler.fit_transform(X)

# Map to firing rates
r_min, r_max = 7.0, 50.0
rates = r_min + Xn * (r_max - r_min)   # shape (150, 4)

# Choosing ONE class 
class_id = 0
idx = np.where(y == class_id)[0]
train_rates = rates[idx]               # (50, 4)

W_FEAT_MULT = 55.0
# ============================================================================

# ============================================================================
# Brunel-style helper (from NEST brunel_alpha_nest example)
# - We need this to convert a desired PSP amplitude J (mV) into a PSC amplitude
#   for iaf_psc_alpha, so that J really means "PSP peak in mV".
def LambertWm1(x):
    nest.ll_api.sli_push(x)
    nest.ll_api.sli_run("LambertWm1")
    return nest.ll_api.sli_pop()

def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = tauMem / tauSyn
    b = 1.0 / tauSyn - 1.0 / tauMem
    t_max = 1.0 / b * (-LambertWm1(-exp(-1.0 / a) / a) - 1.0 / a)
    return (exp(1.0) / (tauSyn * CMem * b) *
            ((exp(-t_max / tauMem) - exp(-t_max / tauSyn)) / b -
             t_max * exp(-t_max / tauSyn)))


# ============================================================================
# Main parameters
g = 4.5       # inhibitory/excitatory strength ratio
eta = 0.9     # external rate relative to threshold rate

# External drive target selection:
# - "E"  -> bombard only excitatory population
# - "EI" -> bombard both excitatory and inhibitory populations
noise_targets = "EI"


# ============================================================================
# Network size and connectivity 
seed = 100699
NE = 800
NI = 200
N = NE + NI

epsilon = 0.1                 # connection probability 
CE = int(epsilon * NE)        # indegree from excitatory population
CI = int(epsilon * NI)        # indegree from inhibitory population

delay = 1.5                   # ms
dt = 0.1                      # ms simulation resolution
simtime = 3000.0              # ms

# ============================================================================
# Neuron and synapse constants (Brunel alpha-synapses)
tauSyn = 0.5                  # ms (synaptic time constant)
tauMem = 20.0                 # ms
CMem = 250.0                  # pF
theta = 20.0                  # mV threshold
t_ref = 2.0                   # ms
E_L = 0.0                     # mV
V_reset = 0.0                 # mV

# Desired PSP peak for excitatory connections (Brunel typically uses J=0.1 mV)
J = 0.1                       # mV

# Convert PSP peak (mV) -> PSC amplitude for iaf_psc_alpha
J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J_ex = J / J_unit             # excitatory PSC amplitude (pA-ish in NEST units)
J_in = -g * J_ex              # inhibitory PSC amplitude

# Threshold rate nu_th (1/ms), then nu_ext = eta * nu_th
# p_rate is the poisson_generator rate in Hz that yields nu_ext per synapse.
nu_th = (theta * CMem) / (J_ex * CE * exp(1.0) * tauMem * tauSyn)  # 1/ms
nu_ext = eta * nu_th                                                # 1/ms
p_rate = 1000.0 * nu_ext * CE                                       # Hz


# ============================================================================
# NEST setup
nest.ResetKernel()
nest.SetKernelStatus({
    "resolution": dt,
    "rng_seed": seed,
    "print_time": True,
    "overwrite_files": True,
})
np.random.seed(seed)

neuron_params = {
    "C_m": CMem,
    "tau_m": tauMem,
    "tau_syn_ex": tauSyn,
    "tau_syn_in": tauSyn,
    "t_ref": t_ref,
    "E_L": E_L,
    "V_reset": V_reset,
    "V_m": 0.0,
    "V_th": theta,
}
nest.SetDefaults("iaf_psc_alpha", neuron_params)

# One poisson generator (classic Brunel) with calibrated p_rate
nest.SetDefaults("poisson_generator", {"rate": p_rate})


# ============================================================================
# Create populations
E = nest.Create("iaf_psc_alpha", NE)
I = nest.Create("iaf_psc_alpha", NI)

# Random initial membrane potentials (optional)
V_init = np.random.uniform(V_reset, theta, size=N)
nest.SetStatus(E + I, [{"V_m": float(v)} for v in V_init])


# ============================================================================
# Features groups in the excitatory population

# Feature-to-neuron assignment
group_size = 50
n_features = 4
n_feat_neurons = group_size * n_features  # 200

# First 200 excitatory neurons are feature-driven
E_feat = E[:n_feat_neurons]
E_groups = [
    E_feat[i*group_size:(i+1)*group_size]
    for i in range(n_features)
]

# Remaining excitatory neurons (pure reservoir)
E_rest = E[n_feat_neurons:]
# ============================================================================


# ============================================================================
# Synapse models
# Keep plasticity only for E->E, like your earlier approach.
stdp_params_EE = {
    "synapse_model": "stdp_synapse",
    "weight": J_ex,
    "delay": delay,
    "Wmax": 3.0 * J_ex,
    "lambda": 0.01,
    "alpha": 1.0,
    "tau_plus": 20.0,
}

static_ex = {"synapse_model": "static_synapse", "weight": J_ex, "delay": delay}
static_in = {"synapse_model": "static_synapse", "weight": J_in, "delay": delay}

# Connectivity rule: Brunel is typically fixed indegree
conn_ex = {"rule": "fixed_indegree", "indegree": CE}
conn_in = {"rule": "fixed_indegree", "indegree": CI}


# ============================================================================
# Recurrent connectivity (Brunel)
# E->E plastic
nest.Connect(E, E, conn_ex, stdp_params_EE)

# E->I static
nest.Connect(E, I, conn_ex, static_ex)

# I->(E+I) static
nest.Connect(I, E + I, conn_in, static_in)


# ============================================================================
# External Poisson bombardment (configurable targets)
pg_feat = nest.Create("poisson_generator", n_features) # features
noise = nest.Create("poisson_generator") # Noise

if noise_targets.upper() == "E":
    nest.Connect(noise, E, syn_spec=static_ex)
elif noise_targets.upper() == "EI":
    nest.Connect(noise, E, syn_spec=static_ex)
    nest.Connect(noise, I, syn_spec=static_ex)
else:
    raise ValueError("noise_targets must be 'E' or 'EI'.")


# Connect each generator to its feature group
for k in range(n_features):
    nest.Connect(
        pg_feat[k],
        E_groups[k],
        conn_spec={"rule": "all_to_all"},
        syn_spec={"weight": W_FEAT_MULT * J_ex, "delay": delay}
    )


# ============================================================================
# Record spikes (optional)
spikes_E = nest.Create("spike_recorder")
spikes_I = nest.Create("spike_recorder")
nest.Connect(E, spikes_E)
nest.Connect(I, spikes_I)


# ============================================================================
# Weight tracking: track ONLY E->E (those are plastic)
dt_sample = 50.0
T_total = simtime

weight_snapshots = []
time_points = []

conns_EE = nest.GetConnections(source=E, target=E)  # handle list is stable
t = 0.0
while t < T_total:

    if USE_IRIS_INPUT:
        # Pick a random row from train_rates
        random_row = train_rates[np.random.choice(train_rates.shape[0])]
        # Set feature-specific rates
        for k in range(n_features):
            nest.SetStatus(pg_feat[k], {"rate": float(random_row[k])})
            
    else:
        for k in range(n_features):
            nest.SetStatus(pg_feat[k],{"rate": 0.0})

    nest.Simulate(dt_sample)
    w = np.array(nest.GetStatus(conns_EE, "weight"), dtype=float)
    weight_snapshots.append(w)
    time_points.append(t + dt_sample)
    t += dt_sample

print(f"Done. simtime={simtime} ms, g={g}, eta={eta}, p_rate={p_rate:.2f} Hz, noise_targets={noise_targets}")


# ============================================================================
# Visualization: animated histogram of E->E weights
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(10, 6))

hist_bins = 60
all_weights = np.concatenate(weight_snapshots)
bins = np.linspace(np.min(all_weights), np.max(all_weights), hist_bins)

print(3.0 * J_ex)

def animate(i):
    ax.clear()
    w = weight_snapshots[i]
    tt = time_points[i]
    # Plot histogram
    ax.hist(w, bins=bins, alpha=0.75, density=True, color='C0', label='Histogram')
    # Overlay density plot using gaussian_kde
    try:
        kde = gaussian_kde(w)
        xx = np.linspace(bins[0], bins[-1], 500)
        ax.plot(xx, kde(xx), color='C3', linewidth=2, label='Density')
    except Exception as e:
        pass  # KDE may fail for empty/small data, just skip density
    ax.set_xlabel("E→E synaptic weight (PSC amplitude units)")
    ax.set_ylabel("Density")
    ax.set_title(f"E→E STDP weight distribution   t={tt:.0f} ms   (g={g}, eta={eta}, noise={noise_targets})")
    ax.set_xlim([bins[0], bins[-1]])
    ax.legend()

frame_inds = np.arange(0, len(weight_snapshots), 2)  # speed-up
ani = animation.FuncAnimation(fig, animate, frames=frame_inds, interval=250, repeat=False)

# Save the animation to file (prefer MP4 with ffmpeg; fallback to GIF with pillow)
save_fps = max(1, int(1000.0 / 250.0))  # match the visual playback speed (~4 FPS)
if animation.writers.is_available("ffmpeg"):
    ani.save("EE_weights_evolution.mp4", writer="ffmpeg", dpi=180, fps=save_fps)
elif animation.writers.is_available("pillow"):
    ani.save("EE_weights_evolution.gif", writer="pillow", dpi=180, fps=save_fps)
else:
    print("No animation writer found (ffmpeg/pillow). Install one to save the animation.")

plt.show()


# ============================================================================
# Visualization: early/mid/late KDE-ish via hist overlays (no seaborn)
def hist_overlay(data, bins, label):
    counts, edges = np.histogram(data, bins=bins, density=False)
    centers = 0.5 * (edges[1:] + edges[:-1])
    plt.plot(centers, counts, label=label)




print("W_Max!:",3.0 * J_ex)
print("p_rate!:",p_rate)

# ============================================================================
# Simple raster plot using spike recorder events (no ID remapping)
events_E = nest.GetStatus(spikes_E, "events")[0]
events_I = nest.GetStatus(spikes_I, "events")[0]

plt.figure(figsize=(10, 6))
if len(events_E["times"]):
    plt.scatter(events_E["times"], events_E["senders"], s=1, color='C0', alpha=0.6, label='E')
if len(events_I["times"]):
    plt.scatter(events_I["times"], events_I["senders"], s=1, color='C1', alpha=0.6, label='I')
plt.xlabel("Time (ms)")
plt.ylabel("Neuron GID")
plt.title("Raster plot of all neurons (E blue, I orange)")
plt.xlim([0, simtime])
plt.legend(markerscale=6, loc='upper right')
plt.tight_layout()
plt.savefig("raster_plot.png", dpi=180)
plt.show()
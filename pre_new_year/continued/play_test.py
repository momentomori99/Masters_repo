"""
Simple reservoir computing example with NEST + Brunel network + STDP + Iris dataset.

- Input: Iris features encoded as Poisson rates.
- Reservoir: Brunel-style random network with STDP on excitatory synapses.
- Readout: Logistic Regression trained on spike counts of reservoir neurons.

This code is written for clarity, not for efficiency.
"""

import nest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Load and preprocess IRIS
# -----------------------------

iris = load_iris()
X = iris.data           # shape (150, 4)
y = iris.target         # labels 0, 1, 2

# Scale features to [0, 1] so we can map them to firing rates
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# 2. Build Brunel-like reservoir
# -----------------------------

def build_brunel_reservoir():
    """
    Build a small Brunel-style reservoir network with STDP on excitatory synapses.
    Returns:
        exc_neurons, inh_neurons, all_neurons, input_generators, readout_neurons, neuron_index
    """

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})  # ms

    # --- Network size ---
    N_exc = 200
    N_inh = 50
    N_total = N_exc + N_inh

    # --- Brunel-like parameters (simplified) ---
    delay = 1.5         # ms
    epsilon = 0.1       # connection probability
    w_exc = 1.0         # excitatory weight
    g = 5.0             # inhibitory strength factor, so w_inh = -g * w_exc
    w_inh = -g * w_exc

    # --- Create neurons ---
    exc_neurons = nest.Create("iaf_psc_alpha", N_exc)
    inh_neurons = nest.Create("iaf_psc_alpha", N_inh)
    all_neurons = exc_neurons + inh_neurons

    # --- Background noise to keep network active ---
    noise = nest.Create("poisson_generator", 1)
    nest.SetStatus(noise, {"rate": 8000.0})  # Hz
    noise_syn = {"weight": w_exc, "delay": delay}
    nest.Connect(noise, all_neurons, syn_spec=noise_syn)

    # --- Internal recurrent connectivity ---
    conn_dict = {"rule": "pairwise_bernoulli", "p": epsilon}

    # Excitatory connections: use STDP synapses (NEST 3: use 'synapse_model', not 'model')
    exc_syn = {
        "synapse_model": "stdp_synapse",
        "weight": w_exc,
        "delay": delay
    }

    # Inhibitory connections: static synapses
    inh_syn = {
        "synapse_model": "static_synapse",
        "weight": w_inh,
        "delay": delay
    }

    # E -> all
    nest.Connect(exc_neurons, all_neurons,
                 conn_spec=conn_dict,
                 syn_spec=exc_syn)

    # I -> all
    nest.Connect(inh_neurons, all_neurons,
                 conn_spec=conn_dict,
                 syn_spec=inh_syn)

    # --- Input Poisson generators for 4 features ---
    # --- Input Poisson generators for 4 features ---
    input_generators = nest.Create("poisson_generator", 4)

    # Connect each input generator to all excitatory neurons (for simplicity)
    # NEST will use all_to_all by default, so each of the 4 generators
    # connects to all excitatory neurons.
    input_weight = 2.0
    input_syn = {"weight": input_weight, "delay": delay}
    nest.Connect(input_generators, exc_neurons, syn_spec=input_syn)


    # --- Choose a subset of excitatory neurons for readout ---
    N_readout = 50
    readout_neurons = exc_neurons[:N_readout]

    # Map neuron IDs -> indices in feature vector
    neuron_index = {}
    for i, nid in enumerate(readout_neurons):
        neuron_index[nid] = i

    return exc_neurons, inh_neurons, all_neurons, input_generators, readout_neurons, neuron_index



# -----------------------------
# 3. Encode input and run reservoir
# -----------------------------

def encode_features_to_rates(features):
    """
    Map 4 scaled features (in [0,1]) to 4 Poisson firing rates.
    """
    base_rate = 5.0       # Hz
    rate_scale = 80.0     # Hz
    rates = []
    for val in features:
        rate = base_rate + val * rate_scale
        rates.append(rate)
    return rates


def run_reservoir_for_sample(features,
                             input_generators,
                             readout_neurons,
                             neuron_index,
                             sim_time=150.0):
    """
    Run the reservoir for one Iris sample.
    - Set input generator rates according to features.
    - Simulate.
    - Count spikes of readout neurons.

    Returns:
        spike_counts: numpy array of shape (N_readout,)
    """

    # Reset dynamic state and time, but keep connectivity and synaptic weights (including STDP changes)
    nest.ResetNetwork()

    # Set Poisson rates from features
    rates = encode_features_to_rates(features)
    for i in range(4):
        nest.SetStatus([input_generators[i]], {"rate": rates[i]})

    # Create a spike recorder for this sample
    spike_recorder = nest.Create("spike_recorder")
    nest.Connect(readout_neurons, spike_recorder)

    # Simulate
    nest.Simulate(sim_time)

    # Get spike events
    events = nest.GetStatus(spike_recorder, "events")[0]
    senders = events["senders"]

    # Count spikes per readout neuron
    N_readout = len(readout_neurons)
    spike_counts = np.zeros(N_readout, dtype=float)

    for s in senders:
        # s is a neuron ID
        idx = neuron_index.get(s, None)
        if idx is not None:
            spike_counts[idx] += 1.0

    return spike_counts


# -----------------------------
# 4. Main: build reservoir, collect states, train logistic regression
# -----------------------------

def main():
    # Build Brunel reservoir with STDP
    (exc_neurons,
     inh_neurons,
     all_neurons,
     input_generators,
     readout_neurons,
     neuron_index) = build_brunel_reservoir()

    # Prepare matrix for reservoir states
    n_samples = X_scaled.shape[0]
    n_readout = len(readout_neurons)
    reservoir_states = np.zeros((n_samples, n_readout))

    # Run reservoir for each Iris sample
    print("Running reservoir for each sample...")
    for i in range(n_samples):
        features = X_scaled[i]
        spike_counts = run_reservoir_for_sample(features,
                                                input_generators,
                                                readout_neurons,
                                                neuron_index,
                                                sim_time=150.0)
        reservoir_states[i, :] = spike_counts

        if (i + 1) % 25 == 0:
            print("Processed sample", i + 1, "/", n_samples)

    # Train/test split on reservoir states
    X_train, X_test, y_train, y_test = train_test_split(
        reservoir_states, y, test_size=0.3, random_state=0, stratify=y
    )

    # Train logistic regression readout
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy (reservoir + logistic regression):", acc)


if __name__ == "__main__":
    main()

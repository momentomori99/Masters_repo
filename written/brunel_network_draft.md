# Brunel's Network — Writing Drafts

---

## Related Work (short paragraph)

Brunel's network was introduced by Nicolas Brunel in his 2000 paper *"Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons"* [Brunel, 2000]. In this work, Brunel analytically and numerically studied how large recurrent networks of leaky integrate-and-fire neurons, composed of excitatory and inhibitory subpopulations with sparse random connectivity, can exhibit qualitatively distinct collective activity states depending on two key control parameters: the relative inhibitory synaptic strength g, and the ratio of external input rate to threshold rate η. The model has since become a standard reference model for studying excitation-inhibition balance in recurrent spiking networks, and serves as the architectural basis for the reservoir used in this thesis. A detailed description of the network structure and its dynamical regimes is given in the Theory section.

---

## Theory — Brunel's Network (full subsection)

Brunel's network consists of N sparsely connected leaky integrate-and-fire (LIF) neurons, divided into an excitatory population of size N_E = 0.8N and an inhibitory population of size N_I = 0.2N. Connectivity is random and sparse: each neuron receives synaptic input from a fixed number C of randomly selected presynaptic neurons, where C ≪ N, corresponding to a connection probability of approximately 10%. In addition to recurrent input, each neuron receives external Poissonian spike trains representing background input from outside the network.

The dynamics of the network are governed by two dimensionless parameters. The first is **g**, which represents the ratio of inhibitory to excitatory synaptic weight. A value of g = 1 means inhibitory synapses are equally as strong as excitatory ones; values g > 1 mean inhibition is stronger. The second parameter is **η**, defined as the ratio of the external input firing rate to the threshold rate — the minimum firing rate needed to bring a neuron to threshold in the absence of recurrent input. Together, g and η define a two-dimensional parameter space in which distinct dynamical regimes can be identified.

Brunel identified four principal activity states in this parameter space:

- **Synchronous Regular (SR)**: The network fires in a highly synchronized, periodic manner. Individual neurons fire regularly and at high rates.
- **Asynchronous Regular (AR)**: Neurons fire at a stationary rate with little synchrony, and individual spike trains are regular.
- **Synchronous Irregular (SI)**: Global network activity oscillates, but individual neurons fire irregularly and at low rates. This state emerges when inhibition is strong (g > 4) and the external drive is moderate.
- **Asynchronous Irregular (AI)**: Individual neurons fire in an irregular, Poisson-like manner, with no coherent global oscillation. Population-level activity is approximately stationary.

Of these four regimes, the **AI state** is considered most consistent with the activity observed in the mammalian cortex in vivo. Recordings from cortical neurons during waking states show highly irregular, low-rate spike trains with near-Poisson statistics [Softky & Koch, 1993; Shadlen & Newsome, 1994], which are characteristic of the AI regime. Brunel showed that this state is achieved when inhibition is sufficiently dominant (g > 4) and external drive is suprathreshold (η > 1), placing the network in a regime where recurrent excitation and inhibition are approximately balanced. This excitation-inhibition balance, sometimes referred to as the *balanced state*, is thought to be a fundamental organizational principle of cortical circuits [van Vreeswijk & Sompolinsky, 1996].

In this thesis, Brunel's network serves as the recurrent core of the spiking reservoir. The parameters g and η are used to steer the reservoir toward the AI regime, with the goal of maintaining the network in a dynamical state that supports rich, temporally varied activity — a property considered beneficial for the computational capacity of a reservoir.

---

## References to track down

- Brunel, N. (2000). Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons. *Journal of Computational Neuroscience*, 8, 183–208.
- Softky, W. R., & Koch, C. (1993). The highly irregular firing of cortical cells is inconsistent with temporal integration of random EPSPs. *Journal of Neuroscience*, 13(1), 334–350.
- Shadlen, M. N., & Newsome, W. T. (1994). Noise, neural codes and cortical organization. *Current Opinion in Neurobiology*, 4(4), 569–579.
- van Vreeswijk, C., & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitatory and inhibitory activity. *Science*, 274(5293), 1724–1726.

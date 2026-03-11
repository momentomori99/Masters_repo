
# Theory

## Brunel's network 

Brunel's network consists of N sparsely connected leaky integrate-and-fire (LIF) neurons, divided into an excitatory population of size N\_E = 0.8N and an inhibitory population of size N\_I = 0.2N. Connectivity is random and sparse: each neuron receives synaptic input from a fixed number C of randomly selected presynaptic neurons, where C ≪ N, corresponding to a connection probability of approximately 10%. In addition to recurrent input, each neuron receives external Poissonian spike trains representing background input from outside the network.

The dynamics of the network are governed by two dimensionless parameters. The first is **g**, which represents the ratio of inhibitory to excitatory synaptic weight. A value of g = 1 means inhibitory synapses are equally as strong as excitatory ones; values g > 1 mean inhibition is stronger. The second parameter is **η**, defined as the ratio of the external input firing rate to the threshold rate — the minimum firing rate needed to bring a neuron to threshold in the absence of recurrent input. Together, g and η define a two-dimensional parameter space in which distinct dynamical regimes can be identified.

Brunel identified four principal activity states in this parameter space:

- **Synchronous Regular (SR)**: The network fires in a highly synchronized, periodic manner. Individual neurons fire regularly and at high rates.
- **Asynchronous Regular (AR)**: Neurons fire at a stationary rate with little synchrony, and individual spike trains are regular.
- **Synchronous Irregular (SI)**: Global network activity oscillates, but individual neurons fire irregularly and at low rates. This state emerges when inhibition is strong (g > 4) and the external drive is moderate.
- **Asynchronous Irregular (AI)**: Individual neurons fire in an irregular, Poisson-like manner, with no coherent global oscillation. Population-level activity is approximately stationary.

Of these four regimes, the **AI state** is considered most consistent with the activity observed in the mammalian cortex in vivo. Recordings from cortical neurons during waking states show highly irregular, low-rate spike trains with near-Poisson statistics [Softky & Koch, 1993; Shadlen & Newsome, 1994], which are characteristic of the AI regime. Brunel showed that this state is achieved when inhibition is sufficiently dominant (g > 4) and external drive is suprathreshold (η > 1), placing the network in a regime where recurrent excitation and inhibition are approximately balanced. This excitation-inhibition balance, sometimes referred to as the *balanced state*, is thought to be a fundamental organizational principle of cortical circuits [van Vreeswijk & Sompolinsky, 1996].

In this thesis, Brunel's network serves as the recurrent core of the spiking reservoir. The parameters g and η are used to steer the reservoir toward the AI regime, with the goal of maintaining the network in a dynamical state that supports rich, temporally varied activity — a property considered beneficial for the computational capacity of a reservoir.

# Related work

## BindsNet approach to reservoir computing

BindsNet python library package was made for simulating of spiking neural networks, specifically geared towards machine learning and reinforcement learning. In their paper [SOURCE] they claim that existing software frameworks support a wide range of neural functionality, but yet are typically not suitable for rapid prototyping  or application to problems in the domain of machine learning. 

BinsNet is the main library that is used in the work of this thesis. They have an open repository on the github with some of the project examples which use their library. One specific example that this thesis takes and inspiration from, is an example of reservoir computing framework. In this example they use MNIST dataset to encode the pixels into the spiking rates. The way BindsNet does it is that they look at the values of the pixels, and calculate the firing rate from those values, and produce N matrices for each pixel where each element in the matrix is either 1 and 0, indicating if that pixel has spiked or not. The number N corresponds the total number of timesteps of the simulation. 

During the simulation itself, each pixel neuron is connected to the reservoir network, that consists of LIF neurons (ref. to theory section about LIF). In every timestep, if the pixels value in the matrix is set to 1, then it spikes and gives a value to the membrane potential of the LIF neuron in the network that it is connected to. During this simulation, in every timestep, the model takes note of which LIF neurons has spiked in everytimestep, and from each timestep it creates a new array where each element is its own neuron from the network. In each timestep, it gives a value of either 1 or 0 in every element neuron in the array indicating whether the neuron has spike or not. In the next timestep, the same process is repeated and at the end of the simulation resulting N arrays are produced from the simulation. Each indicating which neurons has spiked and not spiked in every time step. Here once again, N is the number of total time-steps during the whole simulation. After collecting N arrays, these arrays are then flattened to one long array, and this long array is then given to a logistic regression model for classification. 

## Brunel's network

Brunel's network was introduced by Nicolas Brunel in his 2000 paper *"Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons"* [Brunel, 2000]. In this work, Brunel analytically and numerically studied how large recurrent networks of leaky integrate-and-fire neurons, composed of excitatory and inhibitory subpopulations with sparse random connectivity, can exhibit qualitatively distinct collective activity states depending on two key control parameters: the relative inhibitory synaptic strength g, and the ratio of external input rate to threshold rate η. The model has since become a standard reference model for studying excitation-inhibition balance in recurrent spiking networks, and serves as the architectural basis for the reservoir used in this thesis. A detailed description of the network structure and its dynamical regimes is given in the Theory section.

# Method

## First modification - decoding the activity of the network

As explained in the Related work section, in the BindsNet example for reservoir, they create a new array for each timestep of the networks activity throughout the whole simulation, creating N arrays where N is the total numbers of the timesteps. By knowing the basic principles of the "the curse of the dimensionality" in the machine learning context, we know that this might be way to many features for the logistic regression to handle (explain more of this). Therefore, the first instinct was to reduce the number of features the logistic regression gets. To do this, the change was to instead of looking at which neurons fire and not in every timestep, the spikes gets added to a count in every timestep. This way, instead of getting N x M features, where again N is the total number of timestep, and M is the total number of LIF neurons, we get a only M features. This way, every feature value is the total number of spikes of a specific neurons during the whole simulation. However, this way it is easy to loose an important feature of the activity - temporal information. To tackles this problem, the model divides the simulation into K bins, where each bin is a specific time bin. For example if the simulation is 100ms length, and the model has 2 bins, then each bin would have 50ms. This way, each bin has M number of elements, each consisting the count of the spikes of each neuron during this bin time. This way, the model will produce less feature, but still maintain the temporal information of the activity. And the number of the bins would be a hyperparameter of the model. 

## Second modification - Introduce Brunel's network

Recordings from cortical neurons in awake, behaving animals have consistently shown that individual neurons fire in a highly irregular, low-rate manner — with interspike interval statistics close to those of a Poisson process [Softky & Koch, 1993; Shadlen & Newsome, 1994]. This irregularity is not noise to be suppressed; rather, it appears to be a fundamental feature of the operating regime of cortical circuits, arising from a tight balance between excitatory and inhibitory synaptic currents [van Vreeswijk & Sompolinsky, 1996]. The implication is that the cortex does not function as a neatly synchronized machine, but as a network perpetually held near the edge of cancellation between opposing drives.

This observation motivates the second modification. The baseline reservoir inherited from BindsNet uses a generic recurrent network with no principled structure governing its excitation-inhibition balance. If the goal is to build a reservoir that operates in a biologically realistic regime — one where activity is rich, temporally varied, and not dominated by either runaway excitation or silent suppression — then the network dynamics need to be explicitly designed around this balance. Brunel's network [Brunel, 2000] provides exactly this. By parameterizing the inhibitory-to-excitatory weight ratio g and the ratio of external drive to threshold η, the network can be steered into the Asynchronous Irregular (AI) regime, where individual neurons fire irregularly and at low rates while global activity remains stationary. As described in the Theory section, this is the regime most consistent with in vivo cortical recordings.

From a reservoir computing perspective, the AI regime is also appealing for a more functional reason: the irregular, decorrelated activity of individual neurons means the reservoir produces a high-dimensional, diverse representation of each input. A reservoir in which all neurons fire synchronously, or in which most are silent, compresses information. A reservoir in the AI state spreads it. This makes the downstream readout's classification task easier. The second modification therefore replaces the generic reservoir with a Brunel-structured network, introducing explicit excitatory and inhibitory subpopulations, sparse random connectivity governed by a fixed connection probability of 10%, and weight parameters tuned to place the network in the AI regime.


### Network structure

The reservoir consists of N LIF neurons divided into an excitatory population of size N\_E = 0.8N and an inhibitory population of size N\_I = 0.2N, following the ratio used in Brunel's original model. In addition to the recurrent connections within the network and the input connections from the stimulus, each neuron receives an independent Poissonian noise input. This noise represents the background synaptic bombardment that a cortical neuron receives from the rest of the brain, outside of the circuit being modelled. Crucially, it is this noise drive — not the stimulus alone — that keeps the network active between stimulus-evoked spikes and prevents the network from falling silent.

The recurrent weight matrices are initialized to target the AI regime. Excitatory-to-excitatory and excitatory-to-inhibitory connections are drawn from a normal distribution with a positive mean, scaled by the excitatory synaptic weight w\_E. Inhibitory-to-excitatory and inhibitory-to-inhibitory connections are drawn with a negative mean, scaled by -g \* w\_E, where g is the inhibitory strength parameter. Setting g > 4 ensures that inhibitory synapses are strong enough to suppress runaway excitation and push the network into the AI regime. The parameter η controls the external noise rate relative to the threshold rate, and setting η slightly above 1 ensures neurons receive just enough drive to fire at low rates without saturating. Together, the choice of g and η places the network in the target dynamical state, which can be verified by inspecting the raster plot of the network activity: in the AI regime, spikes appear scattered and irregular with no visible oscillatory structure. [TODO: include raster plot figure here.]

## Connectivity in the network

### Non Spatial connectivity


For the case where we have a simple Brunel's network, we look at each neuron in the network, and iterate through each other neuron in the same network. When we iterate, we let the connection to be determined by probability. And this probability is set to be 10%, same as the Brunel set in his paper [SOURCE]. This means that each neuron has 10% probability to be connected to other neuron. Initiating for which neurons are connected to which, we call this a mask matrix. This mask matrix simply indicates which neurons are connected, by 1, and which are not, by 0.  As for the weight itself, at the beginning of the build of the model, we set a variable for the mean weight, and standard deviation of the weight. Then in the build the mask matrix is multiplied by a value drawn from a normal distribution, which has the mean that is defined in the build. And the same with the standard deviation, which is also defined in the build. 

### Connecting the input spikes to the network in non spatial part

Since the connections are assigned randomly. Each input spiking neuron is connected to a random excitatory neuron in the network. In order to not overwhelm some excitatory neurons with inputs, we create a rule that does not allow multiple input to be connected to one single E neuron. That is each E neuron does not get multiple connections from different input neurons, except multiple noise neurons. However, each input neuron can be connected to multiple excitatory neurons. The reason for this design, is that each excitatory neuron will take responsability to spike as the result of only one input neuron. Meaning each E neuron is dedicated to react to only one specific stimuli, and not multiple. 

### Spatial connectivity

We introduce spatial connectivity in the spiking reservoir by arranging each neuron on a two-dimensional lattice. Each neuron is assigned on a unique spatial coordinate (i, j) on this lattice, where i is the neurons row element, and j is the neurons column element. The neuron population consist of excitatory and inhibitory sub-populations, which are randomly distributed across the lattice randomly during initialization.  [Here you should have the image of the lattice that indicates which neurons are E and which are I.]

Inside the lattice, each neuron has 8 neighbors shown in a figure [include figure here]. However, the neurons on edges has only 3 neighbors, and neurons on the corners has only 2 neighbors. [again show a figure illustrating this]. For this simulation, we let all neurons has equal amount of neighbors. This is both simplification and a assumption [Give some more context for why we wish to use this assumption]. In order to fulfill this assumption, we define a toroidal geometry of the connections in this lattice. As the result of this geometry, neurons at the edge have neighboring positions with the neurons that are located on the adjacent edge. The same applies to the corner neurons too. [Again, have a figure showing this.]

Regarding the connectivity, the aim here is for the neurons to be connected to its nearest neighbors. This is achieved by defining the probability for the connectivity by the Gaussian kernel. This calculates the distance between the source neuron and a target neuron, and assigns a probability that is dependent on the distance [Here very important to show the mathematical expression for this].

An important parameter of this kernel is sigma. This parameter determines the size of the connectivity radius. The higher value of the sigma parameter, the more outgoing connections does each neuron have. 

This is done for every source neuron to every target neuron, and by doing this a connectivity probability matrix is created. Where each row of this matrix indicates the source neuron index, and each column indicates the target neuron index.  

However, for more flexible testing of the network, we introduce a new parameter epsilon. Which has a purpose of defining a connectivity density for each neuron. To achieve this, we take the defined connection probability matrix and scale it with [show the equation]. This is a global rescaling and probability normalization which in result can be used as density calibrator. [you should definitely talk more about how it works and how it makes sense mathematically].

Since the goal of this matrix is to determine which neurons are connected to which, the ideal way of showing this is by indicating which connections exists by defining it as a value 1, and which connection do not exist, by defining it as a value 0. 

Since we now have a connection probability matrix, we use a simple clamp so that elements that have values under 0.5 gets assigned as 0.0, and values that are higher or equal than 0.5 gets assigned as 1.0. This way we have created a mask matrix, which shows which neurons are connected, and which are not. [here show two plots, one in 2D of an example neuron with outgoing connections, also one showing in 1D. Also multiple version with different values of epsilon and sigma].

However, neurons in a neural network are connected by different values of weights. We therefore want to scale each connection with a specific value of the weight. In order to achieve this, we perform the same in the non-spatial network principle;  we simply multiple this mask matrix with values taken from normal distribution, with a mean value for the weight, and a standard deviation of this weight as the parameters of this normal distribution. 

### Connecting input to spatial network

After introducing the spatial structure into the reservoir by arranging excitatory and inhibitory neurons on a two-dimensional lattice as explained in the previous paragraph, the input connectivity must be reconsidered. In a non-spatial reservoir, input pixels may be connected to randomly selected excitatory neurons. However, such random connectivity ignores a fundamental organization principle of biological sensory systems, the topographic mapping. 

In the mammalian visual system, neighboring photoreceptors in the retina project to neighboring neurons in the laterla geniculate nucleus (LGN), and this spatial organization is preserved in primary visual cortex (V1). This retinotopic organization ensures that spatial proximity in visual space is maintained in cortical representations [source! Hube & Wiesel]

Since the goal of this study is to preserve biologival plausability, we therefore reconsider the input connectivity. Pixels that are spatially close in the MNIST image are preferentially connected to excitatory neurons that are also spatially close on the two dimensional lattice where our excitatory and inhibitory neurons lie. It is important to note that all the input neurons are connected to only excitatory neurons, and not inhibitory neurons. 

[I think it would be important to write an appendix of how I define this in very details, all the math and all.]

Each pixel primarily excites neurons near its assigned location, and this locality is controlled by a parameter sigma. This means that the spiking input neuron is connected to all the excitatory neuorons, but a specific input spiking pixel is strongest connected to one assigned location. The influence of this spiking input pixel decreases smoothly with distance, thus creating localized receptive fields. The key idea is that for a pixel to drive a neighborhood of excitatory neurons rather than a single unit. This assigned locality and the influence decrease smoothly with distance is modeled with Gaussian. Which has one important parameter, as mentioned, sigma. This sigma determines how wide the neighborhood is, meaning a small value of sigma gives a sharp, localized drive, while large value of sigma gives broad, distributed drive. 

However, this can lead to some problems, if the network has more neurons than pixels, then we solve it it by letting multiple neurons represent each pixel. If the network has fewer neurons, multiple pixels overlap in representation. This type of mapping is continuous and resolution dependent. 

## Third modification - Self-tuning mechanism

Placing the network in the AI regime by manually setting g and η is straightforward when the input statistics are fixed and well-understood. However, the dynamical state of the network is not only determined by g and η — it also depends on the activity level of the incoming stimulus. A strong input drives more excitatory activity, potentially pushing the network away from the AI regime toward a more synchronous or saturated state. A very weak input may let the network drift toward silence. In either case, the fixed parameter settings chosen during initialization may no longer be appropriate.

To address this, the model includes a self-tuning mechanism that monitors two statistical properties of the network activity after each sample and adjusts g and η accordingly. The two quantities measured are the **mean firing rate** of the excitatory population and the **coefficient of variation (CV)** of the interspike intervals (ISIs).

The mean firing rate is simply the average number of spikes per second across all excitatory neurons. In the AI regime, this rate is expected to be low — typically a few Hz — consistent with the sparse firing observed in cortical neurons in vivo. A high mean rate signals that the network is too excitable; a very low rate signals that activity is being suppressed.

The coefficient of variation is defined for each neuron as the standard deviation of its ISIs divided by the mean of its ISIs:

CV = std(ISI) / mean(ISI)

A CV close to 1 is characteristic of a Poisson process — the statistical signature of the AI regime, where spikes arrive irregularly. A CV significantly below 1 indicates regular, clock-like firing (SR or AR state). A CV above 1 indicates bursty firing. By averaging CV across all neurons that fired at least three spikes during the simulation, a single scalar summary of the irregularity of network activity is obtained.

The self-tuning loop uses these two measurements to update g and η after each stimulus presentation. If the mean rate is too high or the CV is too low — indicating the network is drifting toward synchrony or saturation — g is increased to strengthen inhibition. If the rate is too low, η is increased to provide more external drive. The adjustments are small and incremental, functioning as a feedback controller that keeps the network near the target operating point.

It is worth noting that for the MNIST classification task specifically, this mechanism is not strictly necessary. Because all MNIST digits are drawn from the same image distribution with similar pixel statistics, the aggregate drive to the network remains roughly constant across samples, and the network naturally stays in the same dynamical regime throughout the run. However, the self-tuning mechanism becomes important in settings where input statistics vary substantially across samples — for example, if the reservoir were applied to time series data where some samples are highly active and others nearly quiescent. In such cases, a fixed g and η would cause the network state to drift, and the self-tuning mechanism provides the means to correct it.

# Discussion

## Goals of STDP and how this model does not require it

One interpretation of why stdp did not work as intended, or why we chose not to use it in this model, is that most of the time stdp is used to achieve the excitation-inhibition balance in neural networks. [Here you have to have more than that.] However, since we use Brunel's network, and we focus on achieving asynchroneous and irregular state of the network, which in result makes the network excibit excitation-inhibition balance. Meaning that we already have this balance, and there is no need for further use other learning methods to achieve this. By this reason, we show that Brunel's network is also extremely useful tool, and it can be interpreted as some sort of alternative to some basic learning algorithms as stdp. 






import nest
import matplotlib.pyplot as plt

print("--------------------------------")

# Creating nodes
neuron1 = nest.Create("iaf_psc_alpha")
neuron2 = nest.Create("iaf_psc_alpha")
noise_ex = nest.Create("poisson_generator")
noise_in = nest.Create("poisson_generator")

# Defining parameters to neurons
neuron1.set(I_e=376.0)
neuron2.set(I_e=0.0)

noise_ex.set(rate=80000.0)
noise_in.set(rate=15000.0)

# Measurements
multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"])

spikerecorder = nest.Create("spike_recorder")


# Connecting nodes
nest.Connect(multimeter, neuron1)
nest.Connect(multimeter, neuron2)
nest.Connect(neuron1, spikerecorder)
nest.Connect(neuron2, spikerecorder)

nest.Connect(neuron1, neuron2, syn_spec = {"weight":20.0, "delay":1.0})

#syn_dict_ex = {"weight": 1.2}
#syn_dict_in = {"weight": -2.0}
#nest.Connect(noise_ex, neuron1, syn_spec=syn_dict_ex)
#nest.Connect(noise_in, neuron1, syn_spec=syn_dict_in)

#Simualating
nest.Simulate(1000.0)




# Looking at the data

# SPIKES
events = spikerecorder.get("events")
senders = events["senders"]
ts = events["times"]
plt.plot(ts, senders, ".")
plt.show()


#MEMBRARE POTENTIAL
dmm = multimeter.get()
events = dmm["events"]
Vms = events["V_m"]
ts = events["times"]
sender = events["senders"]

Vms1 = Vms[sender==neuron1[0]]
Vms2 = Vms[sender==neuron2[0]]

ts1 = ts[sender==neuron1[0]]
ts2 = ts[sender==neuron2[0]]

plt.figure()
plt.plot(ts1, Vms1)
plt.plot(ts2, Vms2)
plt.show()


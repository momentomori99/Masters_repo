import nest
import matplotlib.pyplot as plt

ndict = {"I_e": 200.0, "tau_m": 20.0}
nest.SetDefaults("iaf_psc_alpha", ndict)

neuronpop1 = nest.Create("iaf_psc_alpha", 100)
neuronpop2 = nest.Create("iaf_psc_alpha", 100)
neuronpop3 = nest.Create("iaf_psc_alpha", 100)

edict = {"I_e": 200.0, "tau_m": 20.0}
idict = {"I_e": 300.0}
nest.CopyModel("iaf_psc_alpha", "exc_iaf_psc_alpha", params=edict)
nest.CopyModel("iaf_psc_alpha", "inh_iaf_psc_alpha", params=idict)

epop1 = nest.Create("exc_iaf_psc_alpha", 100)
epop2 = nest.Create("exc_iaf_psc_alpha", 100)
ipop1 = nest.Create("inh_iaf_psc_alpha", 30)
ipop2 = nest.Create("inh_iaf_psc_alpha", 30)

# ------------------------------------------
# Connecting two populations of neurons with deterministic connections

pop1 = nest.Create("iaf_psc_alpha", 10)
pop2 = nest.Create("iaf_psc_alpha", 10)
pop1.set({"I_e": 376.0})

multimeters = nest.Create("multimeter", 10)
multimeters.set({"record_from":["V_m"]})

nest.Connect(pop1, pop2, syn_spec={"weight":20.0}) # By default, all connected to all
nest.Connect(pop1, pop2, "one_to_one", syn_spec={"weight":20.0}) # 1st neuron in pop1 connected only to 1st neuron in pop2, and so on
nest.Connect(multimeters, pop2, "one_to_one")


# Connecting two populations of neurons with random connections

d = 1.0
Je = 2.0
Ke = 20
Ji = -4.0
Ki = 12

conn_dict_ex = {"rule": "fixed_indegree", "indegree": Ke}
conn_dict_in = {"rule": "fixed_indegree", "indegree": Ki}
syn_dict_ex = {"delay": d, "weight": Je}
syn_dict_in = {"delay": d, "weight": Ji}

nest.Connect(epop1, ipop1, conn_dict_ex, syn_dict_ex)
nest.Connect(ipop1, epop1, conn_dict_in, syn_dict_in)

# We can also use 
# - fixed_outdegree
# - fixed_total_number
# - pairwise_bernoulli (looks like what was used in Brunels network)

allow_autapses = True # allowing self connections
allow_multapses = True # allowing multiple connections between two neurons. 

# ----------------------------------------------------------------------------
# Specifiying the behaviour of devices
pg = nest.Create("poisson_generator")
pg.set({"start": 100.0, "stop": 150.0}) #active only between 100 and 150 ms
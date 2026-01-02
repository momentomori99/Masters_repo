import nest

#Setting defaults to synapses
nest.SetDefaults("stdp_synapse", {"tau_plus": 15.0})

# Create customisec variants of synapse models with freely chosen name
nest.CopyModel("stdp_synapse", "layer1_stdp_synapse", {"Wmax": 90.0})

print(nest.GetDefaults("stdp_synapse")) # Get the defaults of specific synapse model

# Connecting two populations with synapse models:

epop1 = nest.Create("exc_iaf_psc_alpha", 10)
epop2 = nest.Create("exc_iaf_psc_alpha", 10)

conn_dict = {"rule": "pairwise_bernoulli", "p": 0.1}
syn_dict = {"synapse_model": "stdp_synapse", "alpha": 1.0}
nest.Connect(epop1, epop2, conn_dict, syn_dict)

# We can also have different choise for syn_dict
syn_dict = {"synapse_model": "stdp_synapse", 
            "alpha": nest.random.uniform(min=1, max=2),
            "weight": nest.random.uniform(min=1, max=2),
            "delay": 1.0}
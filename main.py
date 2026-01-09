
from preprocessing import Preprocessing
from brunel import Brunel
from readout import Readout
import numpy as np
from tqdm import tqdm
import time
import nest
import os
from datetime import datetime


# Prepare the filename and string to save
now = datetime.now()
filename = f"data/info_{now.strftime('%Y%m%d_%H%M%S')}.txt"

def freeze_stdp(brunel):
    all_neurons = brunel.nodes_ex + brunel.nodes_in
    conns = nest.GetConnections(source=brunel.nodes_ex, target=all_neurons, synapse_model="excitatory_stdp")
    conns.set({"lambda": 0.0})

def reset_state(brunel):
    nest.SetStatus(brunel.nodes_ex + brunel.nodes_in, {"V_m": 0.0})

def train_stdp(brunel, y_train, simtime = 200.0, n_epochs = 3):
    brunel.simtime = simtime
    rng = np.random.RandomState(0)

    # Turning off the noise:
    print("Turning the noise off for stdp training")
    nest.SetStatus(brunel.noise, {"rate": brunel.p_rate * 0.79})


    for ep in range(n_epochs):
        print(f"Epoch {ep+1} of {n_epochs}")
        #brunel.get_stdp_weights(bins=100, show_top_bottom=False, plot=True, return_weights=False, folder_name="")
        input_vector = []
        non_input_vector = []
        for j in tqdm(range(len(X)), desc="Samples"):
            
            brunel.give_input(X[j].reshape(1, -1))

            t0 = nest.GetKernelStatus("biological_time")
            brunel.simulate()
            t1 = nest.GetKernelStatus("biological_time")
            dt = (t1 - t0) / 1000.0

            spike_vector = brunel.get_spike_vector_window(t0, t1)
            inp = spike_vector[:brunel.feature_size].sum() / (brunel.feature_size * dt)
            non = spike_vector[brunel.feature_size:].sum() / ((brunel.NE - brunel.feature_size) * dt)
            input_vector.append(inp)
            non_input_vector.append(non)



            

        brunel.get_stdp_weights(bins=100, show_top_bottom=False, plot=True, return_weights=False, folder_name="")
        brunel.plot_raster()
       
        
        dt = (t1 - t0) / 1000.0
        print(f" FR input-group (Hz) mean: {np.mean(np.array(input_vector))}")
        print(f"FR non-input (Hz) mean: {np.mean(np.array(non_input_vector))}")
        print(f"FR input-group (Hz) std: {np.std(np.array(input_vector))}")
        print(f"FR non-input (Hz) std: {np.std(np.array(non_input_vector))}")
        #inp = spike_vector[:brunel.feature_size].sum() / (brunel.feature_size * dt)
        #non = spike_vector[brunel.feature_size:].sum() / ((brunel.NE - brunel.feature_size) * dt)
        print(f"Spike vector: {spike_vector}")
        print("Noise rate actual:", nest.GetStatus(brunel.noise, "rate")[0])
        print(f"X[j]: {X[j]}")
        #print("FR input-group (Hz):", inp)
        #print("FR non-input (Hz):", non)
        print("NE:", brunel.NE)
        print("feature_size:", brunel.feature_size)
        print("vector len:", len(spike_vector))
        print("input block sum:", spike_vector[:brunel.feature_size].sum())
        print("non block sum:", spike_vector[brunel.feature_size:].sum())
        print("dt ms:", t1 - t0)
        

        


    # #brunel.get_stdp_weights(bins=100, show_top_bottom=False, plot=True, return_weights=False, folder_name="")
    summary = ""
    summary += f"Simulation time: {simtime} ms ---"
    summary += f"Number of epochs: {n_epochs} --- "
    summary += f"Number of runs (samples * epochs): {len(X) * n_epochs}\n"
    
    return summary



preprocessing = Preprocessing()
#dataset_info, X, y = preprocessing.import_iris_dataset()
dataset_info, X, y = preprocessing.import_moon_dataset(plot=False)
#dataset_info, X, y = preprocessing.import_circles_dataset(plot=True)

##########################===================##########################

brunel = Brunel(input=X[0].reshape(1, -1), stdp=True, reset=True, N_neurons=1000)
summary = brunel.print_summary()
brunel.build_network()

#summary_train_stdp = "NO TRAINING WITH STDP"
summary_train_stdp = train_stdp(brunel, y, simtime=1000.0, n_epochs=1)

# Turning the noise back on
#print("Turning the noise back on")
#brunel.p_rate = (1000.0 * brunel.nu_ex * brunel.CE) / brunel.p_rate_scaler
#nest.SetStatus(brunel.noise, {"rate": brunel.p_rate})
#print(f"Noise rate: {brunel.p_rate:.2f} Hz")


#freeze_stdp(brunel)
#brunel.simtime = 1000.0


# spike_matrix = []
# for sample in tqdm(X, desc="Processing samples"):
#     print(f"Processing sample {sample}")
#     reset_state(brunel)
#     brunel.give_input(sample.reshape(1, -1))
#     t0 = nest.GetKernelStatus("biological_time")
#     brunel.simulate()
#     t1 = nest.GetKernelStatus("biological_time")

#     spike_vector = brunel.get_spike_vector_window(t0, t1)
#     spike_matrix.append(spike_vector)

#     firing_rate_ex_list = []
#     firing_rate_in_list = []
#     print(brunel.get_firing_rates_window(t0, t1))
#     firing_rate_ex_list.append(brunel.get_firing_rates_window(t0, t1)[0])
#     firing_rate_in_list.append(brunel.get_firing_rates_window(t0, t1)[1])

#     print(np.mean(firing_rate_ex_list))

#     avg_firing_rate_ex = np.mean(firing_rate_ex_list)
#     avg_firing_rate_in = np.mean(firing_rate_in_list)
#     min_firing_rate_ex = np.min(firing_rate_ex_list)
#     max_firing_rate_ex = np.max(firing_rate_ex_list)
#     min_firing_rate_in = np.min(firing_rate_in_list)
#     max_firing_rate_in = np.max(firing_rate_in_list)
 


# spike_matrix = np.asarray(spike_matrix, dtype=float)
# np.save("data/spike_matrix.npy", spike_matrix)
# #spike_matrix = np.load("data/spike_matrix.npy")

# readout = Readout(spike_matrix, y)
# summary_readout = readout.cross_validation_pca()





# # Save to file
# with open(filename, "w") as f:
#     f.write("======== Dataset information: ========")
#     f.write("\n")
#     f.write(f"{dataset_info}")
#     f.write("\n")
#     f.write("\n")
#     f.write("======== Brunel summary: ========")
#     f.write("\n")
#     f.write(summary)
#     f.write("\n")
#     f.write("\n")
#     f.write(f"{summary_train_stdp}")
#     f.write("\n")
#     f.write("======== Readout summary: ========")
#     f.write("\n")
#     f.write(summary_readout)
#     f.write("\n")
#     f.write("======== Spike matrix summary: ========")
#     f.write("\n")
#     f.write(f"Average firing rate (excitatory): {avg_firing_rate_ex}")
#     f.write("\n")
#     f.write(f"Average firing rate (inhibitory): {avg_firing_rate_in}")
#     f.write("\n")
#     f.write(f"Minimum firing rate (excitatory): {min_firing_rate_ex}")
#     f.write("\n")
#     f.write(f"Maximum firing rate (excitatory): {max_firing_rate_ex}")



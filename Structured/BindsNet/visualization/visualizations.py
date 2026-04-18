import torch
import matplotlib.pyplot as plt

def plot_raster(E_spikes, I_spikes, title_excitatory, title_inhibitory):
    

        sE = E_spikes.squeeze(1)
        sI = I_spikes.squeeze(1)

        tE, nE = torch.where(sE > 0)
        tI, nI = torch.where(sI > 0)

        fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        ax[0].scatter(tE.cpu(), nE.cpu(), color="slategrey", marker=".", s=10, edgecolors="none", alpha=0.8)
        ax[0].set_title(title_excitatory)
        ax[0].set_ylabel("Neuron idx")
        ax[1].scatter(tI.cpu(), nI.cpu(), color="slategrey", marker=".", s=10, edgecolors="none", alpha=0.8)
        ax[1].set_title(title_inhibitory)
        ax[1].set_xlabel("Time (timestep) ms")
        ax[1].set_ylabel("Neuron idx")
        ax[0].grid(True, linestyle="--", alpha=0.6)
        ax[1].grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show(block=True)
        #plt.savefig(f"BindsNet/results/self_tuning/test1/raster_plot_{time.time()}.png")
        plt.close()

def plot_rate_distribution(time, E_spike_counts, I_spike_counts, title_excitatory, title_inhibitory):
        neuron_rates_E = E_spike_counts / (time / 1000.0) #Hz to spikes/sec
        avr_rate_E = neuron_rates_E.mean()
        neuron_rates_I = I_spike_counts / (time / 1000.0) #Hz to spikes/sec
        avr_rate_I = neuron_rates_I.mean()

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].hist(neuron_rates_E.cpu().numpy(), bins=100, color='skyblue', alpha=0.7)
        ax[0].set_title(title_excitatory)
        ax[0].axvline(avr_rate_E, color='salmon', linestyle='--', linewidth=2, label=f"Average rate: {avr_rate_E:.2f} Hz")
        ax[0].legend()
        ax[1].hist(neuron_rates_I.cpu().numpy(), bins=100, color='salmon', alpha=0.7)
        ax[1].set_title(title_inhibitory)
        ax[1].axvline(avr_rate_I, color='skyblue', linestyle='--', linewidth=2, label=f"Average rate: {avr_rate_I:.2f} Hz")
        ax[1].legend()
        plt.xlabel('Firing rate (spikes/s)')
        plt.ylabel('Number of neurons')
        plt.tight_layout()
        plt.show(block=True)
        plt.close()

def plot_spike_distribution(E_spike_counts, I_spike_counts, title):
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)


        E_flat = E_spike_counts.flatten()
        I_flat = I_spike_counts.flatten()

        for i, val in enumerate(E_flat):
            num_spikes = int(val)
            ax.vlines(i, 0, num_spikes, color='skyblue', alpha= 0.8)
        for i, val in enumerate(I_flat):
            i = len(E_flat) + i
            num_spikes = int(val)
            ax.vlines(i, 0, num_spikes, color='salmon', alpha= 0.8)

        ax.set_title(title)
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Number of spikes")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        plt.show(block=True)
        plt.close()
import sys
import os
import importlib.util

_BINDSNET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'BindsNet')
sys.path.insert(0, _BINDSNET_DIR)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

_spec = importlib.util.spec_from_file_location(
    "bindsnet_framework", os.path.join(_BINDSNET_DIR, "framework.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Framework = _mod.Framework

from tools.feature_encoding import encode_feature_map, spikes_to_binned_counts

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)


class TrajectoryFramework:
    def __init__(self, bin_ms=10, **framework_kwargs):
        self.bin_ms = bin_ms
        self.fw = Framework(**framework_kwargs)
        self.fw.build_network()

    def extract_trajectory(self, dataset, index):
        """Run one sample, return cumulative spike counts per bin: (N_bins, N_E)."""
        sample = dataset[index]
        label = sample["label"]
        feat_spikes = encode_feature_map(
            sample["feature_map"], self.fw.time, self.fw.dt, self.fw.intensity,
        )
        _, _, E_spikes, _ = self.fw.run(feat_spikes)
        binned = spikes_to_binned_counts(
            E_spikes, bin_ms=self.bin_ms, dt=self.fw.dt, time=self.fw.time,
        ).numpy()
        trajectory = np.cumsum(binned, axis=0)
        return trajectory, label

    def plot_trajectory_3d(self, trajectory, label=None, save_name="trajectory_3d.png"):
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(trajectory)

        n = reduced.shape[0]
        t_norm = np.linspace(0, 1, n)
        cmap = plt.cm.viridis

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        segments = [[reduced[i], reduced[i + 1]] for i in range(n - 1)]
        seg_colors = cmap((t_norm[:-1] + t_norm[1:]) / 2)
        lc = Line3DCollection(segments, colors=seg_colors, linewidths=2)
        ax.add_collection3d(lc)

        sc = ax.scatter(
            reduced[:, 0], reduced[:, 1], reduced[:, 2],
            c=t_norm, cmap="viridis", s=40, edgecolors="k", linewidths=0.4,
            zorder=5,
        )
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="Time bin")

        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC 1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC 2 ({ev[1]:.1%})")
        ax.set_zlabel(f"PC 3 ({ev[2]:.1%})")
        ax.set_title(f"3D Trajectory (label={label})")
        plt.tight_layout()
        plt.savefig(os.path.join(_RESULTS_DIR, save_name), dpi=150, bbox_inches="tight")
        plt.show()

    def plot_trajectory_2d(self, trajectory, label=None, save_name="trajectory_2d.png"):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(trajectory)

        n = reduced.shape[0]
        t_norm = np.linspace(0, 1, n)
        cmap = plt.cm.viridis

        fig, ax = plt.subplots(figsize=(8, 6))

        points = reduced.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = cmap((t_norm[:-1] + t_norm[1:]) / 2)
        lc = LineCollection(segments, colors=seg_colors, linewidths=2)
        ax.add_collection(lc)

        sc = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=t_norm, cmap="viridis", s=40, edgecolors="k", linewidths=0.4,
            zorder=5,
        )
        fig.colorbar(sc, ax=ax, shrink=0.8, label="Time bin")

        ax.autoscale()
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC 1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC 2 ({ev[1]:.1%})")
        ax.set_title(f"2D Trajectory (label={label})")
        plt.tight_layout()
        plt.savefig(os.path.join(_RESULTS_DIR, save_name), dpi=150, bbox_inches="tight")
        plt.show()

    def plot_multi_trajectory_3d(self, trajectories, labels,
                                  save_name="multi_trajectory_3d.png"):
        concat = np.concatenate(trajectories, axis=0)
        pca = PCA(n_components=3)
        pca.fit(concat)

        unique_labels = sorted(set(labels))
        cmap = plt.cm.tab10
        label_to_color = {l: cmap(i / max(len(unique_labels) - 1, 1))
                          for i, l in enumerate(unique_labels)}

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for traj, label in zip(trajectories, labels):
            reduced = pca.transform(traj)
            color = label_to_color[label]
            n = reduced.shape[0]
            segments = [[reduced[i], reduced[i + 1]] for i in range(n - 1)]
            lc = Line3DCollection(segments, colors=[color] * (n - 1),
                                  linewidths=1.5, alpha=0.7)
            ax.add_collection3d(lc)
            ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                       color=color, s=25, edgecolors="k", linewidths=0.3,
                       zorder=5)

        handles = [plt.Line2D([0], [0], color=label_to_color[l], lw=2,
                              label=str(l)) for l in unique_labels]
        ax.legend(handles=handles, title="Class")

        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC 1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC 2 ({ev[1]:.1%})")
        ax.set_zlabel(f"PC 3 ({ev[2]:.1%})")
        ax.set_title("3D Trajectories by Class")
        plt.tight_layout()
        plt.savefig(os.path.join(_RESULTS_DIR, save_name), dpi=150, bbox_inches="tight")
        plt.show(block=True)

    def save_trajectory_3d_frames(self, trajectory, label=None,
                                     frames_dir="frames_3d"):
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(trajectory)
        ev = pca.explained_variance_ratio_

        n = reduced.shape[0]
        t_norm = np.linspace(0, 1, n)
        cmap = plt.cm.viridis

        out_dir = os.path.join(_RESULTS_DIR, frames_dir)
        os.makedirs(out_dir, exist_ok=True)

        pad = np.array([reduced.min(axis=0), reduced.max(axis=0)])
        margin = (pad[1] - pad[0]) * 0.1

        for k in range(1, n + 1):
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")

            if k > 1:
                segs = [[reduced[i], reduced[i + 1]] for i in range(k - 1)]
                seg_colors = cmap((t_norm[:k - 1] + t_norm[1:k]) / 2)
                lc = Line3DCollection(segs, colors=seg_colors, linewidths=2)
                ax.add_collection3d(lc)

            ax.scatter(
                reduced[:k, 0], reduced[:k, 1], reduced[:k, 2],
                c=t_norm[:k], cmap="viridis", vmin=0, vmax=1,
                s=40, edgecolors="k", linewidths=0.4, zorder=5,
            )

            ax.set_xlim(pad[0][0] - margin[0], pad[1][0] + margin[0])
            ax.set_ylim(pad[0][1] - margin[1], pad[1][1] + margin[1])
            ax.set_zlim(pad[0][2] - margin[2], pad[1][2] + margin[2])
            ax.set_xlabel(f"PC 1 ({ev[0]:.1%})")
            ax.set_ylabel(f"PC 2 ({ev[1]:.1%})")
            ax.set_zlabel(f"PC 3 ({ev[2]:.1%})")
            ax.set_title(f"3D Trajectory (label={label})  bin {k}/{n}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"frame_{k:04d}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {n} frames to {out_dir}/")

    def save_multi_trajectory_3d_frames(self, trajectories, labels,
                                         frames_dir="frames_multi_3d"):
        concat = np.concatenate(trajectories, axis=0)
        pca = PCA(n_components=3)
        pca.fit(concat)
        ev = pca.explained_variance_ratio_

        reduced_list = [pca.transform(t) for t in trajectories]
        n_bins = trajectories[0].shape[0]

        unique_labels = sorted(set(labels))
        cmap = plt.cm.tab10
        label_to_color = {l: cmap(i / max(len(unique_labels) - 1, 1))
                          for i, l in enumerate(unique_labels)}

        all_reduced = np.concatenate(reduced_list, axis=0)
        pad = np.array([all_reduced.min(axis=0), all_reduced.max(axis=0)])
        margin = (pad[1] - pad[0]) * 0.1

        out_dir = os.path.join(_RESULTS_DIR, frames_dir)
        os.makedirs(out_dir, exist_ok=True)

        handles = [plt.Line2D([0], [0], color=label_to_color[l], lw=2,
                              label=str(l)) for l in unique_labels]

        for k in range(1, n_bins + 1):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            for red, label in zip(reduced_list, labels):
                color = label_to_color[label]
                if k > 1:
                    segs = [[red[i], red[i + 1]] for i in range(k - 1)]
                    lc = Line3DCollection(segs, colors=[color] * (k - 1),
                                          linewidths=1.5, alpha=0.7)
                    ax.add_collection3d(lc)
                ax.scatter(red[:k, 0], red[:k, 1], red[:k, 2],
                           color=color, s=25, edgecolors="k", linewidths=0.3,
                           zorder=5)

            ax.legend(handles=handles, title="Class")
            ax.set_xlim(pad[0][0] - margin[0], pad[1][0] + margin[0])
            ax.set_ylim(pad[0][1] - margin[1], pad[1][1] + margin[1])
            ax.set_zlim(pad[0][2] - margin[2], pad[1][2] + margin[2])
            ax.set_xlabel(f"PC 1 ({ev[0]:.1%})")
            ax.set_ylabel(f"PC 2 ({ev[1]:.1%})")
            ax.set_zlabel(f"PC 3 ({ev[2]:.1%})")
            ax.set_title(f"3D Trajectories by Class  bin {k}/{n_bins}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"frame_{k:04d}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {n_bins} frames to {out_dir}/")

    def plot_multi_trajectory_2d(self, trajectories, labels,
                                  save_name="multi_trajectory_2d.png"):
        concat = np.concatenate(trajectories, axis=0)
        pca = PCA(n_components=2)
        pca.fit(concat)

        unique_labels = sorted(set(labels))
        cmap = plt.cm.tab10
        label_to_color = {l: cmap(i / max(len(unique_labels) - 1, 1))
                          for i, l in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=(9, 7))

        for traj, label in zip(trajectories, labels):
            reduced = pca.transform(traj)
            color = label_to_color[label]
            n = reduced.shape[0]
            points = reduced.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=[color] * (n - 1),
                                linewidths=1.5, alpha=0.7)
            ax.add_collection(lc)
            ax.scatter(reduced[:, 0], reduced[:, 1], color=color, s=25,
                       edgecolors="k", linewidths=0.3, zorder=5)

        handles = [plt.Line2D([0], [0], color=label_to_color[l], lw=2,
                              label=str(l)) for l in unique_labels]
        ax.legend(handles=handles, title="Class")

        ax.autoscale()
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC 1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC 2 ({ev[1]:.1%})")
        ax.set_title("2D Trajectories by Class")
        plt.tight_layout()
        plt.savefig(os.path.join(_RESULTS_DIR, save_name), dpi=150, bbox_inches="tight")
        plt.show()

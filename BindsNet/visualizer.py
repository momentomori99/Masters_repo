import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    """Visualize high-dimensional reservoir states and their clustering properties."""
    
    def __init__(self, pairs):
        """
        Initialize visualizer with training/test pairs.
        
        Args:
            pairs: List of (features, label) tuples
        """
        self.pairs = pairs
        self.features = torch.stack([p[0] for p in pairs]).numpy()
        self.labels = np.array([int(p[1]) for p in pairs])
        self.n_classes = len(np.unique(self.labels))
        
    def plot_tsne(self, perplexity=30, n_iter=1000, random_state=42, figsize=(12, 10)):
        """
        Visualize using t-SNE (t-distributed Stochastic Neighbor Embedding).
        Best for preserving local structure and revealing clusters.
        
        Args:
            perplexity: Controls local vs global structure (5-50 typical)
            n_iter: Number of iterations
            random_state: Random seed for reproducibility
        """
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, 
                    random_state=random_state, verbose=1)
        embedded = tsne.fit_transform(self.features)
        
        # Create color map
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes))
        
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(self.n_classes):
            mask = self.labels == i
            ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                      c=[colors[i]], label=f'Digit {i}', 
                      alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE Visualization of Reservoir States', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return embedded
    
    def plot_pca(self, n_components=2, figsize=(12, 10)):
        """
        Visualize using PCA (Principal Component Analysis).
        Best for understanding variance and linear separability.
        
        Args:
            n_components: 2 or 3 for visualization
        """
        print("Computing PCA...")
        pca = PCA(n_components=n_components)
        embedded = pca.fit_transform(self.features)
        
        # Print explained variance
        var_explained = pca.explained_variance_ratio_
        print(f"Explained variance by PC1: {var_explained[0]*100:.2f}%")
        print(f"Explained variance by PC2: {var_explained[1]*100:.2f}%")
        if n_components == 3:
            print(f"Explained variance by PC3: {var_explained[2]*100:.2f}%")
        print(f"Total variance explained: {sum(var_explained)*100:.2f}%")
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes))
        
        if n_components == 2:
            fig, ax = plt.subplots(figsize=figsize)
            for i in range(self.n_classes):
                mask = self.labels == i
                ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                          c=[colors[i]], label=f'Digit {i}', 
                          alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', fontsize=12)
            ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', fontsize=12)
            ax.set_title('PCA Visualization of Reservoir States', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            plt.show(block=True)
            
        elif n_components == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            for i in range(self.n_classes):
                mask = self.labels == i
                ax.scatter(embedded[mask, 0], embedded[mask, 1], embedded[mask, 2],
                          c=[colors[i]], label=f'Digit {i}', 
                          alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', fontsize=10)
            ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', fontsize=10)
            ax.set_zlabel(f'PC3 ({var_explained[2]*100:.1f}% var)', fontsize=10)
            ax.set_title('PCA 3D Visualization of Reservoir States', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
        
        return embedded, pca
    
    def plot_distance_matrix(self, metric='euclidean', figsize=(10, 8)):
        """
        Visualize pairwise distances between samples as a heatmap.
        Reveals overall structure and clustering.
        
        Args:
            metric: 'euclidean' or 'cosine'
        """
        from scipy.spatial.distance import pdist, squareform
        
        print(f"Computing {metric} distance matrix...")
        if metric == 'cosine':
            # Normalize features for cosine similarity
            features_norm = self.features / (np.linalg.norm(self.features, axis=1, keepdims=True) + 1e-8)
            distances = pdist(features_norm, metric='cosine')
        else:
            distances = pdist(self.features, metric=metric)
        
        dist_matrix = squareform(distances)
        
        # Sort by label for better visualization
        sorted_idx = np.argsort(self.labels)
        dist_matrix_sorted = dist_matrix[sorted_idx][:, sorted_idx]
        labels_sorted = self.labels[sorted_idx]
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(dist_matrix_sorted, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{metric.capitalize()} Distance', fontsize=12)
        
        # Add label boundaries
        boundaries = np.where(np.diff(labels_sorted) != 0)[0] + 0.5
        for b in boundaries:
            ax.axhline(b, color='red', linewidth=1, alpha=0.5)
            ax.axvline(b, color='red', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Sample Index (sorted by label)', fontsize=12)
        ax.set_ylabel('Sample Index (sorted by label)', fontsize=12)
        ax.set_title(f'Pairwise {metric.capitalize()} Distance Matrix\n(Sorted by Label)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return dist_matrix
    
    def plot_class_separation(self, figsize=(14, 5)):
        """
        Analyze within-class vs between-class distances.
        Good metric for cluster quality.
        """
        from scipy.spatial.distance import pdist, squareform
        
        distances = squareform(pdist(self.features, metric='euclidean'))
        
        within_class_dists = []
        between_class_dists = []
        
        for i in range(len(self.labels)):
            for j in range(i+1, len(self.labels)):
                if self.labels[i] == self.labels[j]:
                    within_class_dists.append(distances[i, j])
                else:
                    between_class_dists.append(distances[i, j])
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram comparison
        axes[0].hist(within_class_dists, bins=50, alpha=0.6, label='Within-class', color='blue')
        axes[0].hist(between_class_dists, bins=50, alpha=0.6, label='Between-class', color='red')
        axes[0].set_xlabel('Euclidean Distance', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distance Distributions', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[1].boxplot([within_class_dists, between_class_dists], 
                       labels=['Within-class', 'Between-class'])
        axes[1].set_ylabel('Euclidean Distance', fontsize=12)
        axes[1].set_title('Distance Statistics', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show(block=True)
        
        # Print statistics
        print("\n=== Cluster Separation Statistics ===")
        print(f"Within-class distance: {np.mean(within_class_dists):.4f} ± {np.std(within_class_dists):.4f}")
        print(f"Between-class distance: {np.mean(between_class_dists):.4f} ± {np.std(between_class_dists):.4f}")
        separation_ratio = np.mean(between_class_dists) / (np.mean(within_class_dists) + 1e-8)
        print(f"Separation ratio (higher is better): {separation_ratio:.4f}")
        
        return within_class_dists, between_class_dists
    
    def plot_per_class_separation(self, figsize=(14, 8)):
        """
        Analyze separation for each digit class individually.
        Identifies which digits are well-separated vs confused.
        """
        from scipy.spatial.distance import cdist
        
        # Compute mean feature vector for each class
        class_means = []
        for i in range(self.n_classes):
            mask = self.labels == i
            class_means.append(self.features[mask].mean(axis=0))
        class_means = np.array(class_means)
        
        # Compute pairwise distances between class centroids
        centroid_distances = cdist(class_means, class_means, metric='euclidean')
        
        # Compute within-class variance for each class
        within_class_var = []
        for i in range(self.n_classes):
            mask = self.labels == i
            class_features = self.features[mask]
            var = np.mean(np.var(class_features, axis=0))
            within_class_var.append(var)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Heatmap of centroid distances
        im = axes[0].imshow(centroid_distances, cmap='viridis', aspect='auto')
        axes[0].set_xticks(range(self.n_classes))
        axes[0].set_yticks(range(self.n_classes))
        axes[0].set_xlabel('Digit Class', fontsize=12)
        axes[0].set_ylabel('Digit Class', fontsize=12)
        axes[0].set_title('Inter-Class Centroid Distances', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[0])
        
        # Add text annotations
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                text = axes[0].text(j, i, f'{centroid_distances[i, j]:.1f}',
                                   ha="center", va="center", color="white", fontsize=8)
        
        # Within-class variance bar plot
        axes[1].bar(range(self.n_classes), within_class_var, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Digit Class', fontsize=12)
        axes[1].set_ylabel('Within-Class Variance', fontsize=12)
        axes[1].set_title('Within-Class Variance by Digit', fontsize=13, fontweight='bold')
        axes[1].set_xticks(range(self.n_classes))
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return centroid_distances, within_class_var
    
    def plot_confusion_proximity(self, figsize=(10, 8)):
        """
        Create a confusion-style matrix based on feature similarity.
        Shows which digit pairs have similar reservoir representations.
        """
        from scipy.spatial.distance import cdist
        
        # Compute class centroids
        class_means = []
        for i in range(self.n_classes):
            mask = self.labels == i
            class_means.append(self.features[mask].mean(axis=0))
        class_means = np.array(class_means)
        
        # Compute distances and convert to similarity
        distances = cdist(class_means, class_means, metric='euclidean')
        # Convert to similarity (higher = more similar)
        max_dist = distances.max()
        similarity = max_dist - distances
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(similarity, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xlabel('Digit Class', fontsize=12)
        ax.set_ylabel('Digit Class', fontsize=12)
        ax.set_title('Feature Similarity Matrix\n(Higher = More Confusable)', 
                     fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity', fontsize=12)
        
        # Add text annotations
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i != j:  # Don't show diagonal (self-similarity)
                    text = ax.text(j, i, f'{similarity[i, j]:.1f}',
                                  ha="center", va="center", 
                                  color="black" if similarity[i, j] > similarity.mean() else "white",
                                  fontsize=9)
        
        plt.tight_layout()
        plt.show(block=True)
        
        return similarity


# Example usage:
"""
# After training your reservoir:
visualizer = ReservoirVisualizer(training_pairs)

# 1. t-SNE for cluster visualization (best for seeing clusters)
visualizer.plot_tsne(perplexity=30)

# 2. PCA for variance analysis
visualizer.plot_pca(n_components=2)
visualizer.plot_pca(n_components=3)  # 3D version

# 3. Distance matrix heatmap
visualizer.plot_distance_matrix(metric='euclidean')

# 4. Analyze cluster quality
visualizer.plot_class_separation()

# 5. Per-class analysis
visualizer.plot_per_class_separation()

# 6. Confusion/similarity matrix
visualizer.plot_confusion_proximity()
"""
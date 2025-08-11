import Project

def plot_cumulative_variance(pca, label=""):
    """Plot cumulative explained variance with optional condition label."""
    explained = Project.np.cumsum(pca.explained_variance_ratio_)
    Project.plt.plot(explained, marker='o')
    Project.plt.axhline(0.9, color='r', linestyle='--')
    Project.plt.xlabel('Number of Principal Components')
    Project.plt.ylabel('Cumulative Explained Variance')
    title = 'Cumulative Explained Variance by PCA Components'
    if label:
        title += f' ({label})'
    Project.plt.title(title)
    Project.plt.grid(True)
    Project.plt.tight_layout()
    Project.plt.show()

def print_top_positive_negative_PC_weights(pca, labels_data, condition, top_k=5):
    """Print top-k positively and negatively covarying channel weights for first 2 PCs."""
    for i in range(2):  # First 2 PCs
        pc_weights = pca.components_[i]
        sorted_indices_pos = Project.np.argsort(pc_weights)[-top_k:][::-1]  # top positive
        sorted_indices_neg = Project.np.argsort(pc_weights)[:top_k]         # top negative

        print(f"\n--- Top {top_k} Positively Covarying Channels for PC{i+1} ({condition}) ---")
        for idx in sorted_indices_pos:
            print(f"{labels_data[idx]}: {pc_weights[idx]:.4f}")

        print(f"\n--- Top {top_k} Negatively Covarying Channels for PC{i+1} ({condition}) ---")
        for idx in sorted_indices_neg:
            print(f"{labels_data[idx]}: {pc_weights[idx]:.4f}")


def plot_PCVarianceContribution_BarCharts(pca, label=""):
    """Plot a bar chart showing each PC's variance contribution."""
    Project.plt.figure(figsize=(10, 5))
    Project.plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100)
    Project.plt.xlabel('Principal Component')
    Project.plt.ylabel('Variance Explained (%)')
    title = f'Variance Contribution of Each Principal Component'
    if label:
        title += f' ({label})'
    Project.plt.title(title)
    Project.plt.grid(True)
    Project.plt.tight_layout()
    Project.plt.show()


def plot_BarChartSlicesWithPCWeights(pca, label=""):
    """Plot top 5 PC bar charts sliced into channel contributions, with optional condition label."""
    components = pca.components_
    expl_var = pca.explained_variance_ratio_ * 100
    top_n_pcs = 5
    top_k_channels = 5
    fig, ax = Project.plt.subplots(figsize=(10, 6))
    bottom = Project.np.zeros(top_n_pcs)
    colors = Project.plt.cm.tab20.colors

    for i in range(top_k_channels):
        values = []
        labels_new = []
        for pc in range(top_n_pcs):
            weights = components[pc]
            abs_weights = Project.np.abs(weights)
            top_idx = Project.np.argsort(abs_weights)[-top_k_channels:]  # top-k indices
            channel_idx = top_idx[-(i + 1)]
            channel_contrib = abs_weights[channel_idx] / abs_weights[top_idx].sum() * expl_var[pc]
            values.append(channel_contrib)
            labels_new.append(labels_data[channel_idx])

        ax.bar(Project.np.arange(1, top_n_pcs + 1), values, bottom=bottom,
               label=labels_new[0], color=colors[i % len(colors)])
        bottom += values

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Contribution (%)')
    title = f'Top {top_k_channels} Channel Contributions to Top {top_n_pcs} PCs'
    if label:
        title += f' ({label})'
    ax.set_title(title)
    ax.legend(title='Channels', bbox_to_anchor=(1.05, 1), loc='upper left')
    Project.plt.tight_layout()
    Project.plt.show()

if __name__ == "__main__":
    EEG_data, labels_data, info = Project.loadEEG_and_extract_labels('eeg_subj9.mat')
    behavioral_data = Project.loadbehavioraldata('subj9_behavioural.xlsx')
    eegs_by_condition = Project.group_EEG_by_sensory_condition(EEG_data, behavioral_data)
    for condition, eeg in eegs_by_condition.items():
        pca = Project.perform_spatial_PCA(eeg)

        # Plotting functions
        plot_cumulative_variance(pca, condition)
        plot_PCVarianceContribution_BarCharts(pca, condition)
        plot_BarChartSlicesWithPCWeights(pca, condition)

        # Print top 5 pos/neg covarying channels for PC1 and PC2
        print_top_positive_negative_PC_weights(pca, labels_data, condition)

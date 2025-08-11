import Project


def plot_tucker_core_slice_energy(core, mode_names, label=""):
    """
    Visualize the energy contribution of Tucker core tensor slices along each mode.
    This gives intuition similar to PCA variance explained.
    """

    fig, axs = Project.plt.subplots(1, 3, figsize=(18, 5))
    modes = ['Spatial', 'Temporal', 'Trial']

    for i, mode in enumerate(modes):
        # Sum over other two modes
        summed = Project.np.sum(core**2, axis=tuple(j for j in range(3) if j != i))
        axs[i].bar(Project.np.arange(len(summed)), summed)
        axs[i].set_title(f'{mode} Mode Core Slice Energy' + (f' ({label})' if label else ''))
        axs[i].set_xlabel(f'{mode} Component Index')
        axs[i].set_ylabel('Energy')

    Project.plt.tight_layout()
    Project.plt.show()


def plot_top_channels_from_tucker(factor_matrix, labels_data, label="", top_n=5):
    """
    Plot top contributing channels to spatial mode factors.
    Similar to PCA channel contribution plots.
    """
    import matplotlib.pyplot as plt

    spatial_factors = factor_matrix  # shape: (n_channels, n_components)
    abs_weights = Project.np.abs(spatial_factors)
    top_channels = Project.np.argsort(abs_weights, axis=0)[-top_n:, :]  # shape: (top_n, n_components)

    fig, axs = plt.subplots(1, top_n, figsize=(20, 5), sharey=True)
    for i in range(top_n):
        idxs = top_channels[i]
        labels = [labels_data[j] for j in idxs]
        weights = abs_weights[idxs, i]
        axs[i].bar(labels, weights)
        axs[i].set_title(f'Spatial Factor {i+1}')
        axs[i].tick_params(axis='x', rotation=45)

    plt.suptitle(f"Top Channel Contributions to Tucker Spatial Factors" + (f' ({label})' if label else ''))
    plt.tight_layout()
    plt.show()

def compute_energy_distribution(core, threshold=0.90):
    total_energy = Project.np.sum(core**2)
    mode_names = ['Spatial', 'Temporal', 'Trial']
    n_components_needed = {}

    for i, mode in enumerate(mode_names):
        summed = Project.np.sum(core**2, axis=tuple(j for j in range(3) if j != i))
        sorted_energy = Project.np.sort(summed)[::-1]  # descending
        cumulative_energy = Project.np.cumsum(sorted_energy)
        num_needed = Project.np.searchsorted(cumulative_energy, threshold * total_energy) + 1
        n_components_needed[mode] = num_needed

    return n_components_needed


def plot_cumulative_energy(core, condition_label="", threshold=0.90):
    total_energy = Project.np.sum(core**2)
    mode_names = ['Spatial', 'Temporal', 'Trial']

    Project.plt.figure(figsize=(18, 5))
    Project.plt.suptitle(f'Cumulative Energy per Mode – {condition_label}', fontsize=16)

    for i, mode in enumerate(mode_names):
        summed = Project.np.sum(core**2, axis=tuple(j for j in range(3) if j != i))
        sorted_energy = Project.np.sort(summed)[::-1]  # descending
        cumulative_energy = Project.np.cumsum(sorted_energy) / total_energy

        Project.plt.subplot(1, 3, i + 1)
        Project.plt.plot(Project.np.arange(1, len(cumulative_energy) + 1), cumulative_energy, marker='o')
        Project.plt.axhline(y=threshold, color='r', linestyle='--')
        Project.plt.title(f'{mode} Mode')
        Project.plt.xlabel('Number of Components')
        Project.plt.ylabel('Cumulative Energy')
        Project.plt.grid(True)

    Project.plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
    Project.plt.show()

if __name__ == "__main__":
    EEG_data, labels_data, info = Project.loadEEG_and_extract_labels('eeg_subj9.mat')
    behavioral_data = Project.loadbehavioraldata('subj9_behavioural.xlsx')
    eegs_by_condition = Project.group_EEG_by_sensory_condition(EEG_data, behavioral_data)

    # Run Tucker decomposition and visualize
    tucker_results = Project.tucker_decompose_conditions(eegs_by_condition, ranks=((10, 6, 7)))
    for condition, (core, factors) in tucker_results.items():
        plot_tucker_core_slice_energy(core, ['Spatial', 'Temporal', 'Trial'], condition)
        plot_top_channels_from_tucker(factors[0], labels_data, condition, top_n=5)
        n_components = compute_energy_distribution(core)
        print(f"Components needed to explain 90% energy for {condition}:", n_components)
        plot_cumulative_energy(core, condition_label=condition)






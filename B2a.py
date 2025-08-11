import Project

def plot_TopographicMapofTuckerComponents(spatial_factors, labels_data, title_prefix="Tucker", top_n=5):
    """
    Plot topographic maps of top-n Tucker spatial components (one per subplot).

    Args:
        spatial_factors: np.ndarray of shape (n_channels, n_components)
        labels_data: list of EEG channel names
        title_prefix: prefix for figure title (e.g., condition name)
        top_n: number of components to plot
    """
    import mne
    import matplotlib.pyplot as plt

    info = mne.create_info(ch_names=labels_data, sfreq=1000, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    fig, axes = plt.subplots(1, top_n, figsize=(20, 4))
    for i in range(top_n):
        component = spatial_factors[:, i]  # (64,)
        mne.viz.plot_topomap(component, info, axes=axes[i], show=False)
        axes[i].set_title(f"{title_prefix} Comp {i + 1}")

    plt.suptitle(f"{title_prefix} Spatial Patterns (Top {top_n})", fontsize=14)
    plt.tight_layout()
    plt.show()

# Load EEG and behavior
EEG_data, labels_data, info = Project.loadEEG_and_extract_labels('eeg_subj9.mat')
behavioral_data = Project.loadbehavioraldata('subj9_behavioural.xlsx')
eeg_by_condition = Project.group_EEG_by_sensory_condition(EEG_data, behavioral_data)

# Run Tucker decomposition
tucker_results = Project.tucker_decompose_conditions(eeg_by_condition, ranks=(10, 6, 7))

# Plot Tucker spatial topomaps
for condition, (core, factors) in tucker_results.items():
    spatial_factors = factors[0]  # shape: (64, 5) if ranks=(5, 30, 10)
    plot_TopographicMapofTuckerComponents(spatial_factors, labels_data, title_prefix=condition, top_n=5)

import Project

def plot_TopographicMapofPCComponents(pca):
    """Plot a tomographic brain map of PC components for top 5 PCs"""
    info = Project.mne.create_info(ch_names=labels_data, sfreq=1000, ch_types='eeg')
    montage = Project.mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    fig, axes = Project.plt.subplots(1, 5, figsize=(18, 4))
    for i in range(5):
        pc_weights = pca.components_[i]  # shape (64,)
        Project.mne.viz.plot_topomap(pc_weights, info, axes=axes[i], show=False)
        axes[i].set_title(f'PC {i + 1}')
    Project.plt.suptitle('PCA Spatial Patterns (Electrode Loadings)', fontsize=14)
    Project.plt.tight_layout()
    Project.plt.show()

if __name__ == "__main__":
    EEG_data, labels_data, info = Project.loadEEG_and_extract_labels('eeg_subj9.mat')
    behavioral_data = Project.loadbehavioraldata('subj9_behavioural.xlsx')
    #spatial_pca = perform_spatial_PCA(EEG_data)
    eeg_by_condition = Project.group_EEG_by_sensory_condition(EEG_data, behavioral_data)


    # Loop over conditions, run PCA and plot top 2 PCs
    for condition, eeg_data in eeg_by_condition.items():
        pca = Project.perform_spatial_PCA(eeg_data)

        for i in range(2):  # PC1 and PC2
            fig, ax = Project.plt.subplots()
            Project.mne.viz.plot_topomap(pca.components_[i], info, axes=ax, show=False)
            ax.set_title(f"{condition} PC{i + 1} Plotted Coefficients")
            Project.plt.show()
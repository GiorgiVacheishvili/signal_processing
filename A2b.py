import Project

def compute_trial_averaged_scores(eeg_by_condition, spatial_pcas):
    """
    Project trials onto spatial PC1 and PC2 and compute trial-averaged score time series.

    Args:
        eeg_by_condition: dict of EEG arrays (64, time, trials) per condition
        spatial_pcas: dict of PCA objects per condition

    Returns:
        Dictionary mapping condition → (avg_score_pc1, avg_score_pc2), each shape (time,)
    """
    scores_by_condition = {}

    for condition in eeg_by_condition:
        eeg = eeg_by_condition[condition]  # shape: 64×500×n_trials
        n_trials = eeg.shape[2]
        pca = spatial_pcas[condition]
        pc1 = pca.components_[0]  # shape: (64,)
        pc2 = pca.components_[1]  # shape: (64,)

        # Initialize score matrix
        scores_pc1 = Project.np.zeros((n_trials, eeg.shape[1]))  # (n_trials, 500)
        scores_pc2 = Project.np.zeros((n_trials, eeg.shape[1]))

        # Project each trial
        for i in range(n_trials):
            trial = eeg[:, :, i]  # shape: (64, 500)
            scores_pc1[i, :] = pc1 @ trial
            scores_pc2[i, :] = pc2 @ trial

        # Average across trials
        avg_pc1 = Project.np.mean(scores_pc1, axis=0)
        avg_pc2 = Project.np.mean(scores_pc2, axis=0)
        scores_by_condition[condition] = (avg_pc1, avg_pc2)

    return scores_by_condition


def plot_avg_score_timeseries(scores_by_condition):
    """Plot trial-averaged score time series for PC1 and PC2"""
    time = Project.np.arange(100, 600)  # in ms

    fig, axs = Project.plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for condition, (avg_pc1, avg_pc2) in scores_by_condition.items():
        axs[0].plot(time, avg_pc1, label=condition)
        axs[1].plot(time, avg_pc2, label=condition)

    axs[0].set_title("PC1 Trial-Averaged Score Time Series")
    axs[1].set_title("PC2 Trial-Averaged Score Time Series")
    axs[1].set_xlabel("Time (ms)")
    for ax in axs:
        ax.set_ylabel("Score")
        ax.legend()
    Project.plt.tight_layout()
    Project.plt.show()

# Load EEG and behavior
eeg_data, labels, info = Project.loadEEG_and_extract_labels("eeg_subj9.mat")
behavioral_df = Project.loadbehavioraldata("subj9_behavioural.xlsx")

# Group EEG by sensory condition
eeg_by_condition = Project.group_EEG_by_sensory_condition(eeg_data, behavioral_df)

# Compute spatial PCAs per condition
spatial_pcas = {cond: Project.perform_spatial_PCA(eeg_by_condition[cond]) for cond in eeg_by_condition}

# Project trials onto PCs and get trial-averaged scores
scores_by_condition = compute_trial_averaged_scores(eeg_by_condition, spatial_pcas)

# Plot the results
plot_avg_score_timeseries(scores_by_condition)

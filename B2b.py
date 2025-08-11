import Project

def compute_trial_averaged_tucker_scores(eeg_by_condition, tucker_results):
    """
    Project trials onto Tucker spatial components and compute trial-averaged score time series.

    Args:
        eeg_by_condition: dict of EEG arrays (64, time, trials)
        tucker_results: dict mapping condition → (core, [spatial, temporal, trial] factors)

    Returns:
        dict mapping condition → (avg_score_comp1, avg_score_comp2), each shape (time,)
    """
    scores_by_condition = {}

    for condition in eeg_by_condition:
        eeg = eeg_by_condition[condition]  # shape: (64, time, trials)
        spatial_factors = tucker_results[condition][1][0]  # shape: (64, n_components)
        comp1 = spatial_factors[:, 0]
        comp2 = spatial_factors[:, 1]
        n_trials = eeg.shape[2]
        timepoints = eeg.shape[1]

        # Initialize score arrays
        scores_1 = Project.np.zeros((n_trials, timepoints))
        scores_2 = Project.np.zeros((n_trials, timepoints))

        # Project each trial onto spatial components
        for i in range(n_trials):
            trial = eeg[:, :, i]  # shape: (64, time)
            scores_1[i, :] = comp1 @ trial
            scores_2[i, :] = comp2 @ trial

        avg_score1 = Project.np.mean(scores_1, axis=0)
        avg_score2 = Project.np.mean(scores_2, axis=0)
        scores_by_condition[condition] = (avg_score1, avg_score2)

    return scores_by_condition

def plot_avg_tucker_score_timeseries(scores_by_condition):
    """Plot trial-averaged score time series for Tucker spatial components 1 and 2"""
    import numpy as np
    import matplotlib.pyplot as plt

    time = np.arange(100, 600)  # ms

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for condition, (avg1, avg2) in scores_by_condition.items():
        axs[0].plot(time, avg1, label=condition)
        axs[1].plot(time, avg2, label=condition)

    axs[0].set_title("Tucker Spatial Component 1 – Trial-Averaged Score")
    axs[1].set_title("Tucker Spatial Component 2 – Trial-Averaged Score")
    axs[1].set_xlabel("Time (ms)")
    for ax in axs:
        ax.set_ylabel("Score")
        ax.legend()
    plt.tight_layout()
    plt.show()

# Load EEG and behavior
eeg_data, labels, info = Project.loadEEG_and_extract_labels("eeg_subj9.mat")
behavioral_df = Project.loadbehavioraldata("subj9_behavioural.xlsx")

# Group EEG by condition
eeg_by_condition = Project.group_EEG_by_sensory_condition(eeg_data, behavioral_df)

# Run Tucker decomposition
tucker_results = Project.tucker_decompose_conditions(eeg_by_condition, ranks=(10, 6, 7))

# Compute trial-averaged scores from Tucker spatial components
tucker_scores = compute_trial_averaged_tucker_scores(eeg_by_condition, tucker_results)

# Plot the results
plot_avg_tucker_score_timeseries(tucker_scores)

import Project  # your custom project module (containing your loaders)

def prepare_data_for_classification(eeg_data, behavioral_df):
    n_channels, n_timepoints, n_trials = eeg_data.shape
    X = Project.np.transpose(eeg_data, (2, 0, 1)).reshape(n_trials, n_channels * n_timepoints)  # (216, 32000)
    y = behavioral_df.sort_values(by='Trial_Number').reset_index(drop=True)['SensoryCondition'].values
    return X, y

def run_logistic_regression(X, y):
    scaler = Project.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = Project.LogisticRegression(solver='lbfgs', max_iter=1000)
    scores = Project.cross_val_score(clf, X_scaled, y, cv=5)
    return scores, Project.np.mean(scores), clf, X_scaled  # return clf and X_scaled for later use

def plot_channel_importance(clf, ch_names):
    coef_all = Project.np.abs(clf.coef_)  # shape: (n_classes, 32000)
    mean_abs_weights = Project.np.mean(coef_all, axis=0)  # (32000,)

    coef_map = mean_abs_weights.reshape((64, 500))  # back to (channels, timepoints)
    spatial_importance = coef_map.sum(axis=1)  # importance per channel
    spatial_importance /= Project.np.max(spatial_importance)  # normalize

    # Top 10 channels by contribution
    top_indices = Project.np.argsort(spatial_importance)[-10:][::-1]  # descending order
    top_channels = [(ch_names[i], spatial_importance[i]) for i in top_indices]
    print("\nTop 10 contributing EEG channels for classification:")
    for rank, (ch, weight) in enumerate(top_channels, 1):
        print(f"{rank}. {ch}: {weight:.4f}")

    # Plot topomap
    info = Project.mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
    montage = Project.mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    fig, ax = Project.plt.subplots()
    Project.mne.viz.plot_topomap(spatial_importance, info, axes=ax, show=True)
    ax.set_title('Channel Discrimination Importance (Logistic Regression)')

def plot_temporal_importance(clf):
    """
    Plot and print sorted peaks in the temporal importance curve for logistic regression weights.

    Args:
        clf: trained LogisticRegression classifier
    """
    coef_all = Project.np.abs(clf.coef_)  # shape: (n_classes, 32000)
    mean_abs_weights = Project.np.mean(coef_all, axis=0)  # shape: (32000,)

    # Reshape to (64 channels, 500 timepoints)
    coef_map = mean_abs_weights.reshape((64, 500))

    # Sum across channels to get temporal importance per timepoint
    temporal_importance = coef_map.sum(axis=0)
    temporal_importance /= Project.np.max(temporal_importance)  # normalize

    time = Project.np.arange(100, 600)  # ms

    # === Find Peaks ===
    peaks, _ = Project.find_peaks(temporal_importance, prominence=0.01)

    # Extract and sort peak info
    peak_info = [(time[i], temporal_importance[i]) for i in peaks]
    peak_info_sorted = sorted(peak_info, key=lambda x: x[1], reverse=True)

    print("\nTemporal peaks sorted by importance (normalized):")
    for rank, (t, imp) in enumerate(peak_info_sorted, 1):
        print(f"{rank}. {t} ms: {imp:.4f}")

    # === Plot ===
    Project.plt.figure(figsize=(10, 4))
    Project.plt.plot(time, temporal_importance, label='Temporal Discriminative Power')
    Project.plt.plot([t for t, _ in peak_info], [imp for _, imp in peak_info], 'ro', label='Peaks')
    Project.plt.xlabel('Time (ms)')
    Project.plt.ylabel('Normalized Importance')
    Project.plt.title('Temporal Importance of EEG for Classification with Detected Peaks')
    Project.plt.legend()
    Project.plt.grid(True)
    Project.plt.tight_layout()
    Project.plt.show()
# === MAIN EXECUTION ===

# Load EEG and behavior
eeg_data, labels, info = Project.loadEEG_and_extract_labels('eeg_subj9.mat')
behavioral_df = Project.loadbehavioraldata('subj9_behavioural.xlsx')

# Prepare dataset
X, y = prepare_data_for_classification(eeg_data, behavioral_df)

# Run classification and get standardized data + trained model
scores, mean_accuracy, clf, X_scaled = run_logistic_regression(X, y)

# Report performance
print("Cross-validation scores:", scores)
print("Mean classification accuracy: {:.2f}%".format(mean_accuracy * 100))

# Train final model on full data (optional but consistent)
clf_final = Project.LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_scaled, y)

# Plot channel importance
plot_channel_importance(clf_final, labels)

# Plot temporl importance
plot_temporal_importance(clf_final)

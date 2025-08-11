"""PROJECT LIBRARY"""

# MODULES

from scipy.signal import find_peaks
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import mne
import pandas as pd
import scipy.io
import tensorly as tl
from tensorly.decomposition import tucker
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# DATA LOADING

def loadEEG_and_extract_labels(matfile, time1 = 100, time2 = 600):
    """Load EEG data with timepoints of 100-600ms and extract labels of each channel"""
    mat = scipy.io.loadmat(matfile)
    eeg_data = mat['DataX']['data'][0, 0][:, time1:time2, :]  # shape: (64, 3500, 216)
    chanlocs = mat['DataX']['chanlocs'][0, 0]
    chan_array = chanlocs['labels'][0]  # 1x64 array of label structs
    labels = [str(label[0]) for label in chan_array]
    # labels_data should already be your list of 64 channel names
    info = mne.create_info(ch_names=labels, sfreq=1000, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    return [eeg_data, labels, info]

def loadbehavioraldata(xlsxfile):
    """Load behavioral .xlsx data as a pandas DataFrame."""
    return pd.read_excel(xlsxfile)

def group_EEG_by_sensory_condition(eeg_data, behavioral_df):
    """
    Splits EEG data into three sets based on SensoryCondition.

    Args:
        eeg_data: numpy array of shape (channels, time, trials)
        behavioral_df: DataFrame with 'Trial_Number' and 'SensoryCondition' columns

    Returns:
        Dictionary with keys: 'visual', 'auditory', 'audiovisual'
        Each value is a subset of eeg_data for that condition
    """
    # Ensure the trials are ordered properly
    behavioral_df = behavioral_df.sort_values(by='Trial_Number').reset_index(drop=True)

    # Map condition values to trial indices
    visual_idx = behavioral_df[behavioral_df['SensoryCondition'] == 1].index.to_numpy()
    auditory_idx = behavioral_df[behavioral_df['SensoryCondition'] == 2].index.to_numpy()
    audiovisual_idx = behavioral_df[behavioral_df['SensoryCondition'] == 3].index.to_numpy()

    # Subset EEG trials
    eegs_by_condition = {
        'visual': eeg_data[:, :, visual_idx],
        'auditory': eeg_data[:, :, auditory_idx],
        'audiovisual': eeg_data[:, :, audiovisual_idx]
    }

    return eegs_by_condition

# PERFORM PCA

def perform_spatial_PCA(eeg):
    """Perform spatial PCA on EEG with a 2D matrix: 64 channels × (216 trials * 500 timepoints)"""
    reshaped = eeg.reshape(64, -1)
    pca = PCA()
    pca.fit(reshaped.T)  # Each row is one "observation", so we transpose to (108000, 64)
    return pca

def perform_temporal_PCA(eeg):
    """
    Perform PCA on EEG data reshaped to time × (channels × trials).
    Input shape: (channels, time, trials)
    Output: sklearn PCA object fitted on transposed data.
    """
    ch, t, tr = eeg.shape
    reshaped = eeg.transpose(1, 0, 2).reshape(t, -1)  # (time, channels × trials)
    pca = PCA()
    pca.fit(reshaped.T)  # Now rows = observations, cols = timepoints
    return pca

def perform_trial_PCA_and_get_scores(eeg, n_components):
    # Reshape for PCA: (n_trials, n_channels × timepoints)
    reshaped = eeg.transpose(2, 0, 1).reshape(eeg.shape[2], -1)  # shape: (n_trials, 64*500)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(reshaped)  # fit and transform in one step
    return scores

# PERFORM TUCKER DECOMPOSITION

def perform_tucker_decomposition(eeg_tensor, ranks=(5, 30, 10)):
    """
    Apply Tucker decomposition to EEG tensor: (channels × time × trials)

    Args:
        eeg_tensor: numpy array of shape (64, timepoints, trials)
        ranks: tuple of ranks for (spatial, temporal, trial) modes

    Returns:
        core: Core tensor
        factors: List of factor matrices for each mode (channels, time, trials)
    """
    core, factors = tucker(eeg_tensor, rank=ranks)
    return core, factors

def tucker_decompose_conditions(eegs_by_condition, ranks=(5, 30, 10)):
    """
    Perform Tucker decomposition for each sensory condition's EEG tensor.

    Args:
        eegs_by_condition: dict with keys ['visual', 'auditory', 'audiovisual']
        ranks: (channels, time, trials) desired core tensor rank

    Returns:
        Dictionary mapping condition to (core, factors)
    """
    results = {}
    for condition, eeg_tensor in eegs_by_condition.items():
        core, factors = perform_tucker_decomposition(eeg_tensor, ranks)
        results[condition] = (core, factors)
    return results

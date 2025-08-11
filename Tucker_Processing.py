# === MODULES ===
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne


# === DATA LOADING ===

def loadEEG_and_extract_labels(matfile, time1=100, time2=600):
    """Load EEG data with custom time window and extract channel labels."""
    mat = scipy.io.loadmat(matfile)
    eeg_data = mat['DataX']['data'][0, 0][:, time1:time2, :]  # shape: (64, timepoints, trials)
    chanlocs = mat['DataX']['chanlocs'][0, 0]
    chan_array = chanlocs['labels'][0]
    labels = [str(label[0]) for label in chan_array]
    info = mne.create_info(ch_names=labels, sfreq=1000, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    return [eeg_data, labels, info]

def loadbehavioraldata(xlsxfile):
    """Load behavioral .xlsx data as a pandas DataFrame."""
    return pd.read_excel(xlsxfile)

def group_EEG_by_sensory_condition(eeg_data, behavioral_df):
    """Split EEG into subsets based on sensory condition labels."""
    behavioral_df = behavioral_df.sort_values(by='Trial_Number').reset_index(drop=True)
    visual_idx = behavioral_df[behavioral_df['SensoryCondition'] == 1].index.to_numpy()
    auditory_idx = behavioral_df[behavioral_df['SensoryCondition'] == 2].index.to_numpy()
    audiovisual_idx = behavioral_df[behavioral_df['SensoryCondition'] == 3].index.to_numpy()

    return {
        'visual': eeg_data[:, :, visual_idx],
        'auditory': eeg_data[:, :, auditory_idx],
        'audiovisual': eeg_data[:, :, audiovisual_idx]
    }



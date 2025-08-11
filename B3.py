import Project

def build_tucker_feature_matrix(eeg_by_condition, tucker_results, max_spatial=3):
    """
    Build a feature matrix using Tucker spatial components.

    Args:
        eeg_by_condition: dict with condition → EEG array (64, time, trials)
        tucker_results: dict with condition → (core, [spatial, temporal, trial])
        max_components: how many spatial components to use

    Returns:
        X: feature matrix (n_trials, time × components)
        y: label array (n_trials,)
    """

    features = []
    labels = []
    label_map = {'visual': 1, 'auditory': 2, 'audiovisual': 3}

    for cond in eeg_by_condition:
        eeg = eeg_by_condition[cond]  # (64, time, trials)
        spatial_factors = tucker_results[cond][1][0]  # (64, n_spatial_components)
        pcs = spatial_factors[:, :max_spatial]  # shape: (64, k)
        n_trials = eeg.shape[2]

        for i in range(n_trials):
            trial = eeg[:, :, i]  # (64, time)
            proj = pcs.T @ trial  # shape: (k, time)
            features.append(proj.T.flatten())  # shape: (time × k,)
            labels.append(label_map[cond])

    X = Project.np.array(features)
    y = Project.np.array(labels)
    return X, y


def build_tucker_feature_matrix_spatiotemporal(eeg_by_condition, tucker_results, max_spatial=3, max_temporal=5):
    """
    Build a feature matrix using both spatial and temporal Tucker components.

    Args:
        eeg_by_condition: dict with condition → EEG array (64, time, trials)
        tucker_results: dict with condition → (core, [spatial, temporal, trial])
        max_spatial: number of spatial components to keep
        max_temporal: number of temporal components to keep

    Returns:
        X: feature matrix (n_trials, max_spatial × max_temporal)
        y: label array (n_trials,)
    """

    features = []
    labels = []
    label_map = {'visual': 1, 'auditory': 2, 'audiovisual': 3}

    for cond in eeg_by_condition:
        eeg = eeg_by_condition[cond]  # shape: (64, time, trials)
        spatial_factors = tucker_results[cond][1][0][:, :max_spatial]  # (64, max_spatial)
        temporal_factors = tucker_results[cond][1][1][:, :max_temporal]  # (time, max_temporal)
        n_trials = eeg.shape[2]

        for i in range(n_trials):
            trial = eeg[:, :, i]  # shape: (64, time)

            # Project onto spatial and then temporal components
            proj_spatial = spatial_factors.T @ trial  # shape: (max_spatial, time)
            proj_joint = proj_spatial @ temporal_factors  # shape: (max_spatial, max_temporal)

            features.append(proj_joint.flatten())  # shape: (max_spatial × max_temporal,)
            labels.append(label_map[cond])

    X = Project.np.array(features)
    y = Project.np.array(labels)
    return X, y


def show_confusion_matrix(y_true, y_pred, title):
    cm = Project.confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    disp = Project.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Visual', 'Auditory', 'Audiovisual'])
    disp.plot(cmap='Blues')
    Project.plt.title(title)
    Project.plt.show()

# Linear Regression
def classify_tucker_projections_LR(X, y):
    pipeline = Project.Pipeline([
        ('scaler', Project.StandardScaler()),
        ('clf', Project.LogisticRegression(max_iter=1000))
    ])
    cv = Project.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = Project.cross_val_score(pipeline, X, y, cv=cv)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    show_confusion_matrix(y_true_all, y_pred_all, "Tucker - Logistic Regression")
    print(f"Tucker LR Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

# SVM
def classify_tucker_projections_SVM(X, y):
    pipeline = Project.Pipeline([
        ('scaler', Project.StandardScaler()),
        ('clf', Project.SVC(kernel='linear', C=1.0))
    ])
    cv = Project.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = Project.cross_val_score(pipeline, X, y, cv=cv)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    show_confusion_matrix(y_true_all, y_pred_all, "Tucker - SVM")
    print(f"Tucker SVM Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

# RF
def classify_tucker_projections_RF(X, y):
    pipeline = Project.Pipeline([
        ('scaler', Project.StandardScaler()),
        ('clf', Project.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    cv = Project.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = Project.cross_val_score(pipeline, X, y, cv=cv)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    show_confusion_matrix(y_true_all, y_pred_all, "Tucker - Random Forest")
    print(f"Tucker RF Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

#K-NEAREST NEIGHBORS
def classify_tucker_projections_KNN(X, y):
    pipeline = Project.Pipeline([
        ('scaler', Project.StandardScaler()),
        ('clf', Project.KNeighborsClassifier(n_neighbors=5))
    ])
    cv = Project.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = Project.cross_val_score(pipeline, X, y, cv=cv)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    show_confusion_matrix(y_true_all, y_pred_all, "Tucker - KNN")
    print(f"Tucker KNN Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

#MLP NEURAL NET
def classify_tucker_projections_MLP(X, y):
    pipeline = Project.Pipeline([
        ('scaler', Project.StandardScaler()),
        ('clf', Project.MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
    ])
    cv = Project.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = Project.cross_val_score(pipeline, X, y, cv=cv)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    show_confusion_matrix(y_true_all, y_pred_all, "Tucker - MLP")
    print(f"Tucker MLP Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")




# Load EEG and behavior
eeg_data, labels, info = Project.loadEEG_and_extract_labels("eeg_subj9.mat", time1=100, time2=350)
behavioral_df = Project.loadbehavioraldata("subj9_behavioural.xlsx")
eeg_by_condition = Project.group_EEG_by_sensory_condition(eeg_data, behavioral_df)

# Run Tucker decomposition
tucker_results = Project.tucker_decompose_conditions(eeg_by_condition, ranks=(8, 30, 6) )  # e.g., use 8 spatial comps

# Build feature matrix and classify
X_tucker_spatial, y_tucker_spatial = build_tucker_feature_matrix(eeg_by_condition, tucker_results, max_spatial=8)

X_tucker_spatiotemporal, y_tucker_X_tucker_spatiotemporal = build_tucker_feature_matrix_spatiotemporal(
    eeg_by_condition,
    tucker_results,
    max_spatial=8,
    max_temporal=30
)

classify_tucker_projections_LR(X_tucker_spatial, y_tucker_spatial)
classify_tucker_projections_SVM(X_tucker_spatial, y_tucker_spatial)
#classify_tucker_projections_RF(X_tucker_spatial, y_tucker_spatial)
#classify_tucker_projections_KNN(X_tucker_spatial, y_tucker_spatial)
#classify_tucker_projections_MLP(X_tucker_spatial, y_tucker_spatial)

classify_tucker_projections_LR(X_tucker_spatiotemporal, y_tucker_X_tucker_spatiotemporal)
classify_tucker_projections_SVM(X_tucker_spatiotemporal, y_tucker_X_tucker_spatiotemporal)
#classify_tucker_projections_RF(X_tucker_spatiotemporal, y_tucker_X_tucker_spatiotemporal)
#classify_tucker_projections_KNN(X_tucker_spatiotemporal, y_tucker_X_tucker_spatiotemporal)
#classify_tucker_projections_MLP(X_tucker_spatiotemporal, y_tucker_X_tucker_spatiotemporal)

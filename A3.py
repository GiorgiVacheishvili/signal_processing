import Project

def build_feature_matrix(eeg_by_condition, spatial_pcas, max_pcs=3):
    """
    Build a feature matrix: trials × (timepoints × PCs), and corresponding labels.

    Returns:
        X: (n_trials, time × PCs)
        y: (n_trials,) sensory condition labels (1, 2, 3)
    """
    features = []
    labels = []

    label_map = {'visual': 1, 'auditory': 2, 'audiovisual': 3}

    for cond in eeg_by_condition:
        eeg = eeg_by_condition[cond]  # shape: (64, 500, n_trials)
        n_trials = eeg.shape[2]
        pca = spatial_pcas[cond]

        # Get the first k spatial PCs
        pcs = pca.components_[:max_pcs]  # shape: (k, 64)

        # Project each trial and flatten time × PCs into feature vector
        for i in range(n_trials):
            trial = eeg[:, :, i]  # (64, 500)
            proj = pcs @ trial  # (k, 500)
            features.append(proj.T.flatten())  # shape: (500 × k,)
            labels.append(label_map[cond])

    X = Project.np.array(features)  # (n_trials_total, 500 × k)
    y = Project.np.array(labels)
    return X, y
def show_confusion_matrix(y_true, y_pred, title):
    cm = Project.confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    disp = Project.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Visual', 'Auditory', 'Audiovisual'])
    disp.plot(cmap='Blues')
    Project.plt.title(title)
    Project.plt.show()

# LOGISTIC REGRESSION
def classify_PC_projections_LR(X, y):
    """Train classifier on PC projections and report accuracy and confusion matrix"""
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

    show_confusion_matrix(y_true_all, y_pred_all, "Logistic Regression")
    print(f"LR Classification Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")



# SUPPORT VECTOR MACHINE
def classify_PC_projections_SVM(X, y):
    """SVM classifier on PC projections and confusion matrix"""
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

    show_confusion_matrix(y_true_all, y_pred_all, "SUPPORT VECTOR MACHINE")
    print(f"SVM Classification Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")



# RANDOM FOREST
def classify_PC_projections_RF(X, y):
    """Random Forest classifier on PC projections"""
    pipeline = Project.Pipeline([
        ('scaler', Project.StandardScaler()),  # optional, RF doesn't need it
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

    show_confusion_matrix(y_true_all, y_pred_all, "RANDOM FOREST")
    print(f"Random Forest Classification Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")


# K-NEAREST NEIGHBORS
def classify_PC_projections_KNN(X, y):
    """KNN classifier on PC projections"""
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

    show_confusion_matrix(y_true_all, y_pred_all, "K-NEAREST NEIGHBORS")
    print(f"KNN Classification Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")


# MULTILAYER PERCEPTRON
def classify_PC_projections_MLP(X, y):
    """MLP classifier on PC projections"""
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

    show_confusion_matrix(y_true_all, y_pred_all, "MULTILAYER PERCEPTRON")
    print(f"MLP Classification Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")






# Load EEG and behavior
eeg_data, labels, info = Project.loadEEG_and_extract_labels("eeg_subj9.mat", time1 = 350, time2 = 600)
behavioral_df = Project.loadbehavioraldata("subj9_behavioural.xlsx")

# Group EEG by sensory condition
eeg_by_condition = Project.group_EEG_by_sensory_condition(eeg_data, behavioral_df)

# Compute spatial PCAs per condition
spatial_pcas = {cond: Project.perform_spatial_PCA(eeg_by_condition[cond]) for cond in eeg_by_condition}

# === Build feature matrix and classify
X, y = build_feature_matrix(eeg_by_condition, spatial_pcas, max_pcs=6)
classify_PC_projections_LR(X, y)
classify_PC_projections_SVM(X, y)
classify_PC_projections_RF(X, y)
classify_PC_projections_KNN(X, y)
classify_PC_projections_MLP(X, y)

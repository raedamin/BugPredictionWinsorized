# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 20:47:51 2025

@author: JUST
"""

import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm',
            'lcom3', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']
target = 'buggy'
data_folder = 'datasets/'  # Update this to your local folder path

# Feature selection
def select_features(X, y, estimator, n_features=10):
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    return selector.get_support(indices=True)

# Deep learning evaluation
def evaluate_deep_model(X, y, label, dataset_name):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    selected_idx = select_features(X, y, RandomForestClassifier(random_state=42))
    X = X[:, selected_idx]

    acc, prec, rec, f1, roc = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_balanced.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(patience=5, restore_best_weights=True)

        model.fit(X_train_balanced, y_train_balanced, epochs=50, batch_size=32,
                  validation_split=0.2, verbose=0, callbacks=[early_stop])

        y_pred_prob = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred, average='weighted'))
        rec.append(recall_score(y_test, y_pred, average='weighted'))
        f1.append(f1_score(y_test, y_pred, average='weighted'))
        roc.append(roc_auc_score(y_test, y_pred_prob))

    results.append({
        'Dataset': dataset_name,
        'Model': 'Deep Learning',
        'Preprocessing': label,
        'Accuracy': np.mean(acc),
        'Precision': np.mean(prec),
        'Recall': np.mean(rec),
        'F1 Score': np.mean(f1),
        'ROC AUC': np.mean(roc)
    })

    return pd.DataFrame(results)

# Process CSV files
all_results = []

for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        path = os.path.join(data_folder, file)
        df = pd.read_csv(path)
        dataset_name = os.path.splitext(file)[0]

        df['buggy'] = df['bug'].apply(lambda x: 1 if x > 0 else 0)
        y = df[target].values

        # Winsorized
        winsor_df = df.copy()
        for col in features:
            winsor_df[col] = winsorize(winsor_df[col], limits=(0, 0.05))
        X_winsor = winsor_df[features].values
        results_winsor = evaluate_deep_model(X_winsor, y, 'Winsorized', dataset_name)

        # Robust Scaled
        robust_df = df.copy()
        robust_df[features] = RobustScaler().fit_transform(robust_df[features])
        X_robust = robust_df[features].values
        results_robust = evaluate_deep_model(X_robust, y, 'Robust Scaled', dataset_name)

        all_results.append(results_winsor)
        all_results.append(results_robust)

final_df = pd.concat(all_results)

# Export to Excel
final_df.to_excel('deep_learning_results.xlsx', index=False)

with pd.ExcelWriter('deep_learning_results_by_sheet.xlsx', engine='xlsxwriter') as writer:
    for dataset in final_df['Dataset'].unique():
        sheet_df = final_df[final_df['Dataset'] == dataset]
        sheet_df.to_excel(writer, sheet_name=dataset[:31], index=False)

print("\n✅ Results exported to 'deep_learning_results_by_sheet.xlsx'")
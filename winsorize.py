import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# 1. Configuration
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm',
            'lcom3', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']
target = 'buggy'
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}
data_folder = 'datasets/'  # Folder containing CSV files

# 2. Feature selection function
def select_features(X, y, estimator, n_features=10):
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    return selector.get_support(indices=True)

# 3. Cross-validation function
def cross_validate(dataframe, label, dataset_name):
    X = dataframe[features].values
    y = dataframe[target].values
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    selected_idx = select_features(X, y, RandomForestClassifier(random_state=42))
    selected_features = [features[i] for i in selected_idx]
    X = dataframe[selected_features].values

    for name, clf in classifiers.items():
        acc, prec, rec, f1, roc = [], [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

            clf.fit(X_train_balanced, y_train_balanced)
            y_pred = clf.predict(X_test_scaled)
            y_pred_prob = clf.predict_proba(X_test_scaled)

            acc.append(accuracy_score(y_test, y_pred))
            prec.append(precision_score(y_test, y_pred, average='weighted'))
            rec.append(recall_score(y_test, y_pred, average='weighted'))
            f1.append(f1_score(y_test, y_pred, average='weighted'))

            if y_pred_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
            else:
                y_test_bin = label_binarize(y_test, classes=np.unique(y))
                roc_auc = roc_auc_score(y_test_bin, y_pred_prob, average='weighted', multi_class='ovr')
            roc.append(roc_auc)

        results.append({
            'Dataset': dataset_name,
            'Classifier': name,
            'Preprocessing': label,
            'Accuracy': np.mean(acc),
            'Precision': np.mean(prec),
            'Recall': np.mean(rec),
            'F1 Score': np.mean(f1),
            'ROC AUC': np.mean(roc)
        })

    return pd.DataFrame(results)

# 4. Process all CSV files
all_results = []

for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        path = os.path.join(data_folder, file)
        df = pd.read_csv(path)
        dataset_name = os.path.splitext(file)[0]

        # Encode 'bug' column into binary 'buggy'
        df['buggy'] = df['bug'].apply(lambda x: 1 if x > 0 else 0)

        # Winsorized version
        winsorized_df = df.copy()
        for col in features:
            winsorized_df[col] = winsorize(winsorized_df[col], limits=(0, 0.05))
        results_winsor = cross_validate(winsorized_df, 'Winsorized', dataset_name)

        # Robust-scaled version
        robust_df = df.copy()
        robust_df[features] = RobustScaler().fit_transform(robust_df[features])
        results_robust = cross_validate(robust_df, 'Robust Scaled', dataset_name)

        all_results.append(results_winsor)
        all_results.append(results_robust)

# 5. Combine and export results
final_df = pd.concat(all_results)
#final_df.to_excel('model_comparison_results.xlsx', index=False)
#print("\n✅ Results exported to 'model_comparison_results.xlsx'")
# 5. Combine and export results to multiple sheets
with pd.ExcelWriter('model_comparison_results.xlsx', engine='xlsxwriter') as writer:
    for label in final_df['Preprocessing'].unique():
        sheet_df = final_df[final_df['Preprocessing'] == label]
        sheet_df.to_excel(writer, sheet_name=label[:31], index=False)  # Excel sheet names max out at 31 chars
print("\n✅ Results exported to 'model_comparison_results.xlsx' with one sheet per dataset")
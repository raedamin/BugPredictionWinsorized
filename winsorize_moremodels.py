import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import ttest_rel, wilcoxon

# 1. Configuration
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm',
            'lcom3', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']
target = 'buggy'
data_folder = 'datasets/'  # Folder containing CSV files

# 2. Base classifiers
base_classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier()
}

# 3. Ensemble classifiers
def get_ensembles():
    estimators = [(name, clf) for name, clf in base_classifiers.items()]
    voting = VotingClassifier(estimators=estimators, voting='soft')
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    return {'Voting Ensemble': voting, 'Stacking Ensemble': stacking}

# 4. Feature selection
def select_features(X, y, estimator, n_features=10):
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    return selector.get_support(indices=True)

# 5. Cross-validation
def cross_validate(dataframe, label, dataset_name):
    X = dataframe[features].values
    y = dataframe[target].values
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    selected_idx = select_features(X, y, RandomForestClassifier(random_state=42))
    selected_features = [features[i] for i in selected_idx]
    X = dataframe[selected_features].values

    all_models = {**base_classifiers, **get_ensembles()}

    for name, clf in all_models.items():
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

# 6. Process all CSV files
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
            winsorized_df[col] = winsorize(winsorized_df[col], limits=(0, 0.1))
        results_winsor = cross_validate(winsorized_df, 'Winsorized', dataset_name)

        # Robust-scaled version
        robust_df = df.copy()
        robust_df[features] = RobustScaler().fit_transform(robust_df[features])
        results_robust = cross_validate(robust_df, 'Robust Scaled', dataset_name)

        all_results.append(results_winsor)
        all_results.append(results_robust)

("\n✅ Results exported to 'model_comparison_results 1oth.xlsx' with one sheet per dataset")
# 5. Combine and export results
final_df = pd.concat(all_results)
#final_df.to_excel('model_comparison_results.xlsx', index=False)
#print("\n✅ Results exported to 'model_comparison_results.xlsx'")
# 5. Combine and export results to multiple sheets
with pd.ExcelWriter('model_comparison_results 10th.xlsx', engine='xlsxwriter') as writer:
    for label in final_df['Preprocessing'].unique():
        sheet_df = final_df[final_df['Preprocessing'] == label]
        sheet_df.to_excel(writer, sheet_name=label[:31], index=False)  # Excel sheet names max out at 31 chars
print("\n✅ Results exported to 'model_comparison_results.xlsx' with one sheet per dataset")

def compare_preprocessing(final_df, metric='Accuracy'):
    print(f"\n📊 Statistical Comparison for {metric} (Winsorized vs Robust Scaled):")
    for dataset in final_df['Dataset'].unique():
        df = final_df[final_df['Dataset'] == dataset]
        winsor = df[df['Preprocessing'] == 'Winsorized'].sort_values('Classifier')[metric].values
        robust = df[df['Preprocessing'] == 'Robust Scaled'].sort_values('Classifier')[metric].values

        # Paired t-test
        t_stat, p_val = ttest_rel(winsor, robust)
        print(f"Dataset: {dataset} | t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

        # Wilcoxon test (non-parametric)
        try:
            w_stat, w_p_val = wilcoxon(winsor, robust)
            print(f"           Wilcoxon: statistic={w_stat:.3f}, p-value={w_p_val:.4f}")
        except ValueError:
            print("           Wilcoxon test not applicable (no differences or zero variance)")
            
compare_preprocessing(final_df, metric='Accuracy')
compare_preprocessing(final_df, metric='F1 Score')
compare_preprocessing(final_df, metric='ROC AUC')
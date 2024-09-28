import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter

# Set random seed for reproducibility
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

def train_models(group_column, models, data_path, meta_data_path, output_dir, corr_type='pearson', top_features_percent=100):
    """
    Train and evaluate multiple machine learning models on microbial data.

    Parameters:
    - group_column (str): The column name in metadata indicating the group labels.
    - models (list): List of model names to train.
    - data_path (str): Path to the feature data CSV file.
    - meta_data_path (str): Path to the metadata CSV file.
    - output_dir (str): Directory to save output files.
    - corr_type (str): Correlation type for feature selection ('pearson', 'spearman', 'kendall').
    - top_features_percent (float): Percentage of top correlated features to select.

    Returns:
    - results (dict): Dictionary containing ROC AUC scores and confusion matrices for each model.
    """
    # Load data
    features = pd.read_csv(data_path)
    metadata = pd.read_csv(meta_data_path)
    
    # Preprocess metadata
    metadata = metadata[['SampleID', group_column]]
    if group_column == 'group_1':
        metadata = metadata[metadata[group_column] != 'CR']
    
    # Merge feature data with metadata
    data = pd.merge(features, metadata, on='SampleID')
    data.dropna(inplace=True)

    # Compute correlation matrix
    if corr_type in ['pearson', 'spearman', 'kendall']:
        corr_matrix = data.iloc[:, 1:-1].corr(method=corr_type)
    else:
        raise ValueError("Unsupported correlation type. Choose 'pearson', 'spearman', or 'kendall'.")

    # Select top correlated features
    abs_corr_matrix = corr_matrix.abs()
    mean_corr = abs_corr_matrix.mean(axis=0)
    selected_features = mean_corr.sort_values(ascending=False).head(int(len(mean_corr) * (top_features_percent / 100))).index

    X = data[selected_features]
    y = LabelEncoder().fit_transform(data[group_column])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED_VALUE)

    # Balance the training data using oversampling
    counter = Counter(y_train)
    max_count = max(counter.values())
    
    X_train_balanced = pd.DataFrame()
    y_train_balanced = np.array([])
    
    for cls, count in counter.items():
        cls_indices = np.where(y_train == cls)[0]
        n_repeats = max_count // count
        remainder = max_count % count
        X_cls_repeated = pd.concat([X_train.iloc[cls_indices]] * n_repeats + [X_train.iloc[cls_indices][:remainder]])
        y_cls_repeated = np.tile(y_train[cls_indices], n_repeats + 1)[:max_count]
        X_train_balanced = pd.concat([X_train_balanced, X_cls_repeated])
        y_train_balanced = np.concatenate([y_train_balanced, y_cls_repeated])
    
    # Standardize features
    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train_balanced)
    X_test = scaler.transform(X_test)

    # Define available models
    model_dict = {
        'Logistic': LogisticRegression(C=0.1, solver='saga', max_iter=1000),
        'Linear_SVM': SVC(kernel='linear', C=0.5, probability=True),
        'Naive_Bayes': GaussianNB(var_smoothing=1e-9),
        'Radial_SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        'Decision_Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=10),
        'Random_Forest': RandomForestClassifier(n_estimators=200, max_depth=1000, class_weight={0: 1, 1: 200}),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.05, max_depth=6),
        'Neural_Network': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=100)
    }

    # Select requested models
    selected_models = {name: model for name, model in model_dict.items() if name in models}
    results = {}

    for name, model in selected_models.items():
        # Train the model
        model.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        # Evaluate the model
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {'ROC_AUC': roc_auc, 'Confusion_Matrix': cm}
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f'{name}_roc_curve.png'))
        plt.close()
        
        # Save ROC Curve Data
        roc_data = pd.DataFrame({
            'False_Positive_Rate': fpr,
            'True_Positive_Rate': tpr
        })
        roc_data.to_excel(os.path.join(output_dir, f'{name}_roc_data.xlsx'), index=False)
        
        # Print Confusion Matrix
        print(f'Confusion Matrix for {name}:\n{cm}\n')
        
    return results

def main():
    # Define configurations for different datasets
    configurations = [
        {
            'data_path': 'taxa.genus.Abd.csv',
            'meta_data_path': 'group1_hc_rcc.csv',
            'group_column': 'group1',
            'output_dir': 'genus_rcc'
        },
        {
            'data_path': 'taxa.species.Abd.csv',
            'meta_data_path': 'group1_hc_rcc.csv',
            'group_column': 'group1',
            'output_dir': 'species_rcc'
        },
        {
            'data_path': 'taxa.species.Abd.csv',
            'meta_data_path': 'group2_hc_blca.csv',
            'group_column': 'group1',
            'output_dir': 'species_blca'
        },
        {
            'data_path': 'taxa.genus.Abd.csv',
            'meta_data_path': 'group2_hc_blca.csv',
            'group_column': 'group1',
            'output_dir': 'genus_blca'
        }
    ]

    # Common parameters
    models = [
        'Logistic', 'Linear_SVM', 'Naive_Bayes', 'Radial_SVM',
        'Decision_Tree', 'Random_Forest', 'XGBoost', 'Neural_Network'
    ]
    corr_type = 'pearson'
    top_features_percent = 95

    # Execute training for each configuration
    for config in configurations:
        output_dir = config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Training models for dataset: {config['data_path']} with metadata: {config['meta_data_path']}")
        results = train_models(
            group_column=config['group_column'],
            models=models,
            data_path=config['data_path'],
            meta_data_path=config['meta_data_path'],
            output_dir=output_dir,
            corr_type=corr_type,
            top_features_percent=top_features_percent
        )
        print(f"Results for {output_dir}: {results}\n")

if __name__ == "__main__":
    main()

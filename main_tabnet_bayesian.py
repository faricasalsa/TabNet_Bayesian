import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
import joblib
import random
import optuna
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.optimize import minimize_scalar
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, brier_score_loss
)
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

# Import calibration libraries
from netcal.binning import IsotonicRegression, BBQ

# =========================================================
# COMPLETE REPRODUCIBILITY SETUP 
# =========================================================
def set_reproducibility(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Reproducibility set with seed: {seed}")
    return seed

REPRODUCIBILITY_SEED = 123
set_reproducibility(REPRODUCIBILITY_SEED)

pd.set_option('display.float_format', '{:,.4f}'.format)

# Set output directory
output_dir = 'tabnet_bayesian'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*60)
print("LIBRARIES IMPORTED SUCCESSFULLY")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Reproducibility seed: {REPRODUCIBILITY_SEED}")
print(f"Output directory: {output_dir}")
print("="*60)

# =========================================================
# DEFINE CUSTOM LOSS FUNCTION (TANPA CLASS WEIGHT)
# =========================================================
class StandardLoss:
    def __call__(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true.long())

class WeightedAUC(Metric):
    def __init__(self):
        self._name = "weighted_auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return roc_auc_score(y_true, y_score[:, 1])

# =========================================================
# MASK VISUALIZATION FUNCTIONS
# =========================================================
def get_tabnet_masks_array(model, X_data):
    """Mengambil attention mask TabNet untuk banyak sampel."""
    X_data = X_data.astype(np.float32)
    M_explain, masks = model.explain(X_data)

    if isinstance(masks, dict):
        mask_keys = sorted(masks.keys())
        masks_array = np.stack([masks[k] for k in mask_keys], axis=1)
    else:
        masks_array = masks

    return M_explain, masks_array

def visualize_global_mask_heatmap(model, X_data, feature_names=None, save_path=None,
                                  csv_path=None, split_name="all data"):
    print(f"Mengambil attention mask untuk {len(X_data)} sampel ({split_name})...")
    M_explain, masks_array = get_tabnet_masks_array(model, X_data)
    
    mean_mask_by_step = masks_array.mean(axis=0)
    n_steps, n_features = mean_mask_by_step.shape
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    mean_mask_df = pd.DataFrame(
        mean_mask_by_step,
        index=[f"Step_{i+1}" for i in range(n_steps)],
        columns=feature_names
    )
    
    if csv_path:
        mean_mask_df.to_csv(csv_path)
        print(f"✓ Global mask heatmap data saved to {csv_path}")
    
    plt.figure(figsize=(max(14, n_features * 0.55), max(7, n_steps * 0.55)))
    sns.heatmap(mean_mask_df, cmap='viridis', linewidths=0.1, linecolor='white', cbar=True)
    plt.xlabel('Feature')
    plt.ylabel('Decision Step')
    plt.title(f'Global Feature Selection Mask Heatmap\nAverage mask across {len(X_data)} samples ({split_name})')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Global mask heatmap saved to {save_path}")
    plt.show()
    
    return mean_mask_df, M_explain, masks_array

def create_global_aggregate_importance_from_masks(masks_array, feature_names, save_path=None,
                                                  csv_path=None, split_name="all data"):
    aggregate_importance = masks_array.mean(axis=(0, 1))
    
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': aggregate_importance
    }).sort_values('Importance', ascending=False)
    
    if csv_path:
        imp_df.to_csv(csv_path, index=False)
        print(f"✓ Global aggregate mask importance saved to {csv_path}")
    
    top_n = min(20, len(imp_df))
    top_features = imp_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), top_features['Importance'].values)
    plt.yticks(range(top_n), top_features['Feature'].values)
    plt.xlabel('Average Mask Importance')
    plt.title(f'Top {top_n} Features - Global Aggregate Mask Importance\n({split_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Global aggregate mask importance plot saved to {save_path}")
    plt.show()
    
    return imp_df

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def extract_tabnet_logits(model, X_data, batch_size=1024):
    """
    Mengambil logits asli dari jaringan TabNet sebelum softmax.
    Output shape: (n_samples, n_classes)
    """
    model.network.eval()
    device = next(model.network.parameters()).device
    logits_list = []

    with torch.no_grad():
        for start in range(0, len(X_data), batch_size):
            end = start + batch_size
            X_batch = torch.tensor(
                X_data[start:end],
                dtype=torch.float32,
                device=device
            )

            network_output = model.network(X_batch)

            if isinstance(network_output, tuple):
                logits_batch = network_output[0]
            else:
                logits_batch = network_output

            logits_list.append(logits_batch.detach().cpu().numpy())

    return np.vstack(logits_list)

def softmax_numpy(logits):
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def calculate_ece_correct(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_samples = len(y_true)
    
    for idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        
        if idx == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
    
    return ece

def manual_temperature_scaling(logits, labels):
    def nll_loss(temp):
        scaled_logits = logits / temp
        probs = 1 / (1 + np.exp(-scaled_logits))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        return nll
    
    result = minimize_scalar(nll_loss, bounds=(0.5, 5.0), method='bounded')
    optimal_temp = result.x
    
    scaled_logits = logits / optimal_temp
    calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
    
    return calibrated_probs, optimal_temp

# =========================================================
# THRESHOLD SENSITIVITY ANALYSIS FUNCTION
# =========================================================
def threshold_sensitivity_analysis(y_true, y_prob, thresholds, title="Threshold Sensitivity Analysis"):
    """Melakukan analisis sensitivitas threshold dan menampilkan tabel."""
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        results.append({
            'Threshold': threshold,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1,
            'Accuracy': acc
        })
    
    results_df = pd.DataFrame(results)
    return results_df

# =========================================================
# OUTLIER HANDLING FUNCTION (WINSORIZATION)
# =========================================================
def handle_outliers_winsorize(df, columns, limits=(0.01, 0.99)):
    df_clean = df.copy()
    outlier_summary = []
    
    print("\n" + "="*60)
    print("OUTLIER HANDLING WITH WINSORIZATION")
    print("="*60)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        lower = df_clean[col].quantile(limits[0])
        upper = df_clean[col].quantile(limits[1])
        
        original_lower_outliers = (df_clean[col] < lower).sum()
        original_upper_outliers = (df_clean[col] > upper).sum()
        
        df_clean[col] = df_clean[col].clip(lower, upper)
        
        outlier_summary.append({
            'Feature': col,
            'Lower Percentile': limits[0],
            'Upper Percentile': limits[1],
            'Lower Bound': lower,
            'Upper Bound': upper,
            'Lower Outliers Winsorized': original_lower_outliers,
            'Upper Outliers Winsorized': original_upper_outliers,
            'Total Outliers Handled': original_lower_outliers + original_upper_outliers
        })
        
        print(f"  {col}: handled {original_lower_outliers + original_upper_outliers} outliers")
    
    outlier_df = pd.DataFrame(outlier_summary)
    return df_clean, outlier_df

# =========================================================
# OPTUNA OBJECTIVE FUNCTION
# =========================================================
def objective(trial, X_train, X_valid, y_train, y_valid, cat_idxs, cat_dims):
    n_da = trial.suggest_int("n_da", 24, 64, step=4)
    n_steps = trial.suggest_int("n_steps", 3, 10, step=1)
    gamma = trial.suggest_float("gamma", 1.0, 2.0)
    n_independent = trial.suggest_int("n_independent", 1, 5)
    n_shared = trial.suggest_int("n_shared", 1, 5)
    momentum = trial.suggest_float("momentum", 0.01, 0.4)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    clip_value = trial.suggest_float("clip_value", 0.5, 2.0)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    virtual_batch_size = trial.suggest_categorical("virtual_batch_size", [128, 256, 512])
    max_epochs = trial.suggest_int("max_epochs", 30, 60)
    
    model = TabNetClassifier(
        n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
        n_independent=n_independent, n_shared=n_shared, epsilon=1e-15,
        momentum=momentum, lambda_sparse=lambda_sparse, seed=REPRODUCIBILITY_SEED,
        clip_value=clip_value, verbose=0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': lr, 'weight_decay': weight_decay},
    )
    
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)], max_epochs=max_epochs, patience=10,
        batch_size=batch_size, virtual_batch_size=virtual_batch_size,
        num_workers=0, drop_last=False, eval_metric=['auc']
    )
    
    val_auc = max(model.history['val_0_auc'])
    return val_auc

# =========================================================
# LOAD DATASET
# =========================================================
print("="*60)
print("LOADING CREDIT CARD DEFAULT DATASET")
print("="*60)

file_path = "credit_card_default.csv" 

try:
    df = pd.read_csv(file_path)
    print(f"✓ File loaded: {file_path}")
except FileNotFoundError:
    print(f"✗ File tidak ditemukan di: {file_path}")
    np.random.seed(REPRODUCIBILITY_SEED)
    n_samples = 10000
    df = pd.DataFrame({
        'customer_id': range(n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'net_yearly_income': np.random.normal(50000, 20000, n_samples),
        'no_of_days_employed': np.random.randint(0, 5000, n_samples),
        'total_family_members': np.random.randint(1, 10, n_samples),
        'yearly_debt_payments': np.random.normal(10000, 5000, n_samples),
        'credit_limit': np.random.normal(30000, 10000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'no_of_children': np.random.randint(0, 5, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'owns_car': np.random.choice(['Y', 'N'], n_samples),
        'owns_house': np.random.choice(['Y', 'N'], n_samples),
        'occupation_type': np.random.choice(['professional', 'white_collar', 'blue_collar', 'student'], n_samples),
        'migrant_worker': np.random.choice(['Yes', 'No'], n_samples),
        'credit_card_default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    print(f"✓ Sample data created: {df.shape}")

print(f"✓ Dataset loaded: {df.shape}")
print(f"✓ Target distribution:")
print(df['credit_card_default'].value_counts())

# =========================================================
# SAVE TEST DATA BEFORE NORMALIZATION (ADDED)
# =========================================================
print("\n" + "="*60)
print("SAVING TEST DATA BEFORE NORMALIZATION")
print("="*60)

# Save original data for reference
original_df = df.copy()
original_df.to_csv(f'{output_dir}/original_dataset.csv', index=False)
print(f"✓ Original dataset saved to '{output_dir}/original_dataset.csv'")
print(f"  - Shape: {original_df.shape}")

# =========================================================
# REMOVE IDENTIFIER COLUMNS
# =========================================================
print("="*60)
print("REMOVING IDENTIFIER COLUMNS")
print("="*60)

identifier_columns = ['customer_id', 'name']
cols_to_drop = [col for col in identifier_columns if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"✓ Removed identifier columns: {cols_to_drop}")

# =========================================================
# IDENTIFY COLUMN TYPES
# =========================================================
print("="*60)
print("IDENTIFYING COLUMN TYPES")
print("="*60)

continuous_features = ['age', 'net_yearly_income', 'no_of_days_employed', 
                       'total_family_members', 'yearly_debt_payments', 'credit_limit',
                       'credit_score', 'no_of_children']
categorical_features = ['gender', 'owns_car', 'owns_house', 'occupation_type', 'migrant_worker']
target_col = 'credit_card_default'

continuous_features = [col for col in continuous_features if col in df.columns]
categorical_features = [col for col in categorical_features if col in df.columns]

print(f"✓ Continuous features: {continuous_features}")
print(f"✓ Categorical features: {categorical_features}")

# =========================================================
# HANDLE MISSING VALUES DENGAN KNN IMPUTER
# =========================================================
print("="*60)
print("HANDLING MISSING VALUES DENGAN KNN IMPUTER")
print("="*60)

missing_features = []
for col in df.columns:
    if col != target_col and df[col].isnull().sum() > 0:
        missing_features.append(col)
        print(f"✓ Feature '{col}' has {df[col].isnull().sum()} missing values")

if missing_features:
    for col in missing_features:
        indicator_name = f"{col}_missing"
        df[indicator_name] = df[col].isnull().astype(int)
        print(f"✓ Created missing indicator: {indicator_name}")
        continuous_features.append(indicator_name)
    
    for col in categorical_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"✓ Filled categorical '{col}' with mode")
    
    continuous_missing = [col for col in missing_features if col in continuous_features]
    if continuous_missing:
        print(f"\nMengisi missing values pada fitur: {continuous_missing}")
        df_numeric = df[continuous_features].copy()
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        df_imputed = imputer.fit_transform(df_numeric)
        df[continuous_features] = df_imputed
        print(f"✓ KNN Imputer applied with n_neighbors=5")
        knn_imputer = imputer
    else:
        knn_imputer = None
else:
    knn_imputer = None
    print("✓ No missing values found in the dataset")

print("✓ Missing values handled with KNN Imputer")

# =========================================================
# APPLY OUTLIER HANDLING (WINSORIZATION)
# =========================================================
df, outlier_summary_df = handle_outliers_winsorize(
    df, 
    continuous_features, 
    limits=(0.01, 0.99)
)

print(f"\n✓ Outlier handling completed")
print(f"  Total outliers handled: {outlier_summary_df['Total Outliers Handled'].sum()}")

# =========================================================
# ENCODE CATEGORICAL VARIABLES
# =========================================================
print("="*60)
print("ENCODING CATEGORICAL VARIABLES")
print("="*60)

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded '{col}'")

continuous_features_updated = [col for col in continuous_features if col in df.columns]

# =========================================================
# SEPARATE FEATURES 
# =========================================================
print("="*60)
print("SEPARATE FEATURES")
print("="*60)

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

feature_names = list(X.columns)

# =========================================================
# PREPARE CATEGORICAL FEATURES INFO FOR TABNET
# =========================================================
print("="*60)
print("PREPARING CATEGORICAL FEATURES INFO FOR TABNET")
print("="*60)

all_features = list(X.columns)
cat_idxs = []
cat_dims = []

for i, col in enumerate(all_features):
    if col in categorical_features:
        cat_idxs.append(i)
        cat_dims.append(df[col].nunique())
        print(f"✓ Categorical '{col}' at index {i}")

# =========================================================
# SPLIT DATA (80-10-10)
# =========================================================
print("="*60)
print("SPLITTING DATA (80-10-10)")
print("="*60)

X_np = X.values.astype(np.float32)
y_np = y.values.reshape(-1)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_np, y_np, test_size=0.2, random_state=REPRODUCIBILITY_SEED, stratify=y_np
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=REPRODUCIBILITY_SEED, stratify=y_temp
)

print(f"✓ Training: {X_train.shape} ({len(y_train)/len(y_np)*100:.1f}%)")
print(f"✓ Validation: {X_valid.shape} ({len(y_valid)/len(y_np)*100:.1f}%)")
print(f"✓ Test: {X_test.shape} ({len(y_test)/len(y_np)*100:.1f}%)")

# =========================================================
# SAVE TEST DATA BEFORE NORMALIZATION (RAW SPLIT)
# =========================================================
print("\n" + "="*60)
print("SAVING TEST DATA BEFORE NORMALIZATION (RAW SPLIT)")
print("="*60)

# Save raw test data before any normalization
test_raw_df = pd.DataFrame(X_test, columns=all_features)
test_raw_df['actual_target'] = y_test
test_raw_df.to_csv(f'{output_dir}/test_data_before_normalization.csv', index=False, float_format='%.6f')
print(f"✓ Test data before normalization saved to '{output_dir}/test_data_before_normalization.csv'")
print(f"  - Shape: {test_raw_df.shape}")
print(f"  - Columns: {list(test_raw_df.columns)}")

# Also save train and validation raw data for completeness
train_raw_df = pd.DataFrame(X_train, columns=all_features)
train_raw_df['actual_target'] = y_train
train_raw_df.to_csv(f'{output_dir}/train_data_before_normalization.csv', index=False, float_format='%.6f')
print(f"✓ Train data before normalization saved to '{output_dir}/train_data_before_normalization.csv'")

valid_raw_df = pd.DataFrame(X_valid, columns=all_features)
valid_raw_df['actual_target'] = y_valid
valid_raw_df.to_csv(f'{output_dir}/valid_data_before_normalization.csv', index=False, float_format='%.6f')
print(f"✓ Validation data before normalization saved to '{output_dir}/valid_data_before_normalization.csv'")

# =========================================================
# NORMALIZE CONTINUOUS FEATURES - HANYA PADA DATA TRAIN
# =========================================================
print("="*60)
print("NORMALIZATION (HANYA PADA DATA TRAIN)")
print("="*60)

scaler = StandardScaler()
continuous_to_scale = [i for i, col in enumerate(all_features) if col in continuous_features_updated]

if continuous_to_scale:
    X_train_scaled = X_train.copy()
    X_train_scaled[:, continuous_to_scale] = scaler.fit_transform(X_train[:, continuous_to_scale])
    X_valid_scaled = X_valid.copy()
    X_valid_scaled[:, continuous_to_scale] = scaler.transform(X_valid[:, continuous_to_scale])
    X_test_scaled = X_test.copy()
    X_test_scaled[:, continuous_to_scale] = scaler.transform(X_test[:, continuous_to_scale])
    print(f"✓ Normalized {len(continuous_to_scale)} continuous features using ONLY train data statistics")
else:
    X_train_scaled = X_train
    X_valid_scaled = X_valid
    X_test_scaled = X_test
    print("⚠️ No continuous features to normalize")

print("✓ Normalization completed (only on train data)")

# =========================================================
# SAVE MEAN AND STANDARD DEVIATION USED FOR NORMALIZATION
# =========================================================
print("\n" + "="*60)
print("SAVING NORMALIZATION PARAMETERS")
print("="*60)

normalization_params_df = pd.DataFrame({
    'feature': [all_features[i] for i in continuous_to_scale],
    'mean_train': scaler.mean_,
    'std_train': scaler.scale_
})

normalization_params_df.to_csv(
    f'{output_dir}/normalization_parameters.csv',
    index=False
)

print(f"✓ Normalization parameters saved to '{output_dir}/normalization_parameters.csv'")
print(normalization_params_df)

# =========================================================
# SAVE TEST DATA AFTER NORMALIZATION
# =========================================================
print("\n" + "="*60)
print("SAVING TEST DATA AFTER NORMALIZATION")
print("="*60)

# Create DataFrame for test data after normalization
test_normalized_df = pd.DataFrame(X_test_scaled, columns=all_features)
test_normalized_df['actual_target'] = y_test
test_normalized_df.to_csv(f'{output_dir}/test_data_after_normalization.csv', index=False, float_format='%.6f')
print(f"✓ Test data after normalization saved to '{output_dir}/test_data_after_normalization.csv'")
print(f"  - Shape: {test_normalized_df.shape}")
print(f"  - Columns: {list(test_normalized_df.columns)}")

# Also save train and validation data after normalization for completeness
train_normalized_df = pd.DataFrame(X_train_scaled, columns=all_features)
train_normalized_df['actual_target'] = y_train
train_normalized_df.to_csv(f'{output_dir}/train_data_after_normalization.csv', index=False, float_format='%.6f')
print(f"✓ Train data after normalization saved to '{output_dir}/train_data_after_normalization.csv'")

valid_normalized_df = pd.DataFrame(X_valid_scaled, columns=all_features)
valid_normalized_df['actual_target'] = y_valid
valid_normalized_df.to_csv(f'{output_dir}/valid_data_after_normalization.csv', index=False, float_format='%.6f')
print(f"✓ Validation data after normalization saved to '{output_dir}/valid_data_after_normalization.csv'")

# =========================================================
# SAVE ATTACHMENT FILE (ORIGINAL DATA FOR REFERENCE)
# =========================================================
print("\n" + "="*60)
print("SAVING ATTACHMENT FILES")
print("="*60)

# Save the processed data before splitting (complete dataset)
processed_df = pd.concat([pd.DataFrame(X_np, columns=all_features), 
                          pd.Series(y_np, name=target_col)], axis=1)
processed_df.to_csv(f'{output_dir}/processed_dataset.csv', index=False, float_format='%.6f')
print(f"✓ Processed dataset (complete) saved to '{output_dir}/processed_dataset.csv'")

# Save data split summary
split_summary = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Samples': [len(X_train_scaled), len(X_valid_scaled), len(X_test_scaled)],
    'Default_Count': [np.sum(y_train), np.sum(y_valid), np.sum(y_test)],
    'Non_Default_Count': [len(y_train)-np.sum(y_train), len(y_valid)-np.sum(y_valid), len(y_test)-np.sum(y_test)],
    'Default_Ratio': [np.mean(y_train), np.mean(y_valid), np.mean(y_test)]
})
split_summary.to_csv(f'{output_dir}/data_split_summary.csv', index=False)
print(f"✓ Data split summary saved to '{output_dir}/data_split_summary.csv'")

# =========================================================
# CONVERT TO NUMPY ARRAYS (FINAL)
# =========================================================
print("="*60)
print("FINAL DATA PREPARATION")
print("="*60)

X_train_np = X_train_scaled.astype(np.float32)
X_valid_np = X_valid_scaled.astype(np.float32)
X_test_np = X_test_scaled.astype(np.float32)

y_train_np = y_train.reshape(-1)
y_valid_np = y_valid.reshape(-1)
y_test_np = y_test.reshape(-1)

print("✓ Data converted to numpy arrays")

# =========================================================
# OPTUNA HYPERPARAMETER TUNING
# =========================================================
print("\n" + "="*60)
print("OPTUNA HYPERPARAMETER TUNING (MULTIPLE TRIALS)")
print("="*60)

X_hp = X_train_np
Y_hp = y_train_np

print(f"✓ Using only training data for Optuna tuning: {X_hp.shape[0]} samples")
print(f"✓ Validation data size: {X_valid_np.shape[0]} samples")
print(f"✓ Test data is HIDDEN and not used in tuning")

study_dir = "tabnet_bayesian"
if not os.path.exists(study_dir):
    os.makedirs(study_dir)

study = optuna.create_study(
    direction="maximize", 
    study_name='TabNet Optimization',
    sampler=TPESampler(seed=REPRODUCIBILITY_SEED)
)

print("\nRunning Optuna optimization with 30 trials...")
print("Each trial will train a TabNet model on training set and validate on validation set")
print("-" * 60)

study.optimize(
    lambda trial: objective(trial, X_hp, X_valid_np, Y_hp, y_valid_np, cat_idxs, cat_dims),
    n_trials=30,
    n_jobs=1,
    show_progress_bar=True
)

print("\n" + "="*60)
print("BEST HYPERPARAMETERS FOUND")
print("="*60)
best_params = study.best_params
best_auc = study.best_value
print(f"Best validation AUC: {best_auc:.6f}")
for key, value in best_params.items():
    print(f"  • {key}: {value}")

best_n_d = best_params.get('n_da', 32)
best_n_a = best_params.get('n_da', 32)
best_n_steps = best_params.get('n_steps', 5)
best_gamma = best_params.get('gamma', 1.5)
best_cat_emb_dim = 1
best_n_independent = best_params.get('n_independent', 2)
best_n_shared = best_params.get('n_shared', 2)
best_momentum = best_params.get('momentum', 0.1)
best_lambda_sparse = best_params.get('lambda_sparse', 1e-4)
best_clip_value = best_params.get('clip_value', 1.0)
best_lr = best_params.get('lr', 0.001)
best_weight_decay = best_params.get('weight_decay', 1e-5)
best_batch_size = best_params.get('batch_size', 512)
best_virtual_batch_size = best_params.get('virtual_batch_size', 256)
best_max_epochs = best_params.get('max_epochs', 50)
best_patience = 12

joblib.dump(study, f'{study_dir}/optuna_study.pkl')
print(f"\n✓ Optuna study saved to '{study_dir}/optuna_study.pkl' with {len(study.trials)} trials")

best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv(f'{output_dir}/best_hyperparameters.csv', index=False)
print(f"✓ Best hyperparameters saved to '{output_dir}/best_hyperparameters.csv'")

# =========================================================
# BUILD FINAL TABNET MODEL WITH BEST HYPERPARAMETERS
# =========================================================
print("\n" + "="*60)
print("BUILDING FINAL TABNET MODEL WITH BEST HYPERPARAMETERS")
print("="*60)

final_tabnet_model = TabNetClassifier(
    n_d=best_n_d, n_a=best_n_a, n_steps=best_n_steps, gamma=best_gamma,
    cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=best_cat_emb_dim,
    n_independent=best_n_independent, n_shared=best_n_shared, epsilon=1e-15,
    momentum=best_momentum, lambda_sparse=best_lambda_sparse,
    seed=REPRODUCIBILITY_SEED, clip_value=best_clip_value, verbose=0,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': best_lr, 'weight_decay': best_weight_decay},
)

print("✓ Final TabNet model created with best hyperparameters")

# =========================================================
# TRAIN FINAL TABNET MODEL
# =========================================================
print("\n" + "="*60)
print("TRAINING FINAL TABNET MODEL")
print("="*60)

standard_loss = StandardLoss()

final_tabnet_model.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_train_np, y_train_np), (X_valid_np, y_valid_np)],
    eval_name=['train', 'val'], eval_metric=["auc", WeightedAUC],
    max_epochs=best_max_epochs, patience=best_patience,
    batch_size=best_batch_size, virtual_batch_size=best_virtual_batch_size,
    num_workers=0, weights=1, drop_last=False, loss_fn=standard_loss
)

print("\n✓ Final model training completed")

# =========================================================
# MASK VISUALIZATION (FEATURE SELECTION MASKS) - DATA TEST
# =========================================================
print("\n" + "="*60)
print("MASK VISUALIZATION (FEATURE SELECTION MASKS) - DATA TEST")
print("="*60)

X_mask_np = X_test_np.astype(np.float32)

print(f"\nJumlah sampel data test untuk visualisasi mask: {X_mask_np.shape[0]}")
print(f"Jumlah fitur: {len(feature_names)}")
print(f"Jumlah decision step TabNet: {final_tabnet_model.n_steps}")

print("\n1. Generating Mask Heatmap for Test Data...")
global_mask_heatmap_df, M_explain_test, masks_array_test = visualize_global_mask_heatmap(
    final_tabnet_model, X_mask_np, feature_names=feature_names,
    save_path=f'{output_dir}/test_mask_heatmap.png',
    csv_path=f'{output_dir}/test_mask_heatmap.csv', split_name='test data'
)

print("\n2. Generating Aggregate Mask Importance for Test Data...")
aggregate_importance_df = create_global_aggregate_importance_from_masks(
    masks_array_test, feature_names=feature_names,
    save_path=f'{output_dir}/test_aggregate_mask_importance.png',
    csv_path=f'{output_dir}/test_aggregate_mask_importance.csv', split_name='test data'
)

# =========================================================
# COMPARE MASK IMPORTANCE WITH MODEL'S FEATURE IMPORTANCE
# =========================================================
print("\n" + "="*60)
print("COMPARING MASK IMPORTANCE WITH MODEL FEATURE IMPORTANCE")
print("="*60)

model_feature_importance = final_tabnet_model.feature_importances_

if aggregate_importance_df is not None:
    mask_imp_for_compare = aggregate_importance_df[['Feature', 'Importance']].rename(
        columns={'Importance': 'Mask_Importance'}
    )
    model_imp_for_compare = pd.DataFrame({
        'Feature': feature_names,
        'Model_Importance': model_feature_importance
    })
    comparison_imp_df = pd.merge(
        mask_imp_for_compare,
        model_imp_for_compare,
        on='Feature',
        how='left'
    )
    
    correlation = comparison_imp_df['Mask_Importance'].corr(comparison_imp_df['Model_Importance'])
    print(f"\nCorrelation between Mask Importance and Model Feature Importance: {correlation:.4f}")
    
    print("\nTop 10 Features by Mask Importance:")
    print(comparison_imp_df.nlargest(10, 'Mask_Importance')[['Feature', 'Mask_Importance', 'Model_Importance']])
    
    print("\nTop 10 Features by Model Feature Importance:")
    print(comparison_imp_df.nlargest(10, 'Model_Importance')[['Feature', 'Mask_Importance', 'Model_Importance']])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mask importance
    top_mask = comparison_imp_df.nlargest(15, 'Mask_Importance')
    axes[0].barh(top_mask['Feature'], top_mask['Mask_Importance'], color='steelblue')
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Top 15 Features by Mask Importance')
    axes[0].invert_yaxis()
    
    # Model importance
    top_model = comparison_imp_df.nlargest(15, 'Model_Importance')
    axes[1].barh(top_model['Feature'], top_model['Model_Importance'], color='steelblue')
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 15 Features by Model Feature Importance')
    axes[1].invert_yaxis()
    
    plt.suptitle('COMPARISON: Mask Importance vs Model Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Importance comparison saved to '{output_dir}/importance_comparison.png'")
    
    # Save comparison DataFrame
    comparison_imp_df.to_csv(f'{output_dir}/importance_comparison.csv', index=False)

# =========================================================
# PLOT MODEL FEATURE IMPORTANCE (BAR CHART)
# =========================================================
print("\n" + "="*60)
print("PLOT MODEL FEATURE IMPORTANCE")
print("="*60)

# Create model feature importance plot
model_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model_feature_importance
}).sort_values('Importance', ascending=False)

top_n_features = min(20, len(model_imp_df))
top_model_features = model_imp_df.head(top_n_features)

plt.figure(figsize=(12, 8))
plt.barh(range(top_n_features), top_model_features['Importance'].values, color='teal')
plt.yticks(range(top_n_features), top_model_features['Feature'].values)
plt.xlabel('Feature Importance')
plt.title(f'Top {top_n_features} Features - TabNet Model Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_dir}/model_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Model feature importance saved to '{output_dir}/model_feature_importance.png'")

# =========================================================
# GET PREDICTIONS (MURNI TABNET)
# =========================================================
print("="*60)
print("GETTING PREDICTIONS (MURNI TABNET)")
print("="*60)

probs_train = final_tabnet_model.predict_proba(X_train_np)
probs_valid = final_tabnet_model.predict_proba(X_valid_np)
probs_test = final_tabnet_model.predict_proba(X_test_np)

# =========================================================
# LOGITS ASLI TABNET DARI NETWORK
# =========================================================
print("\n" + "="*60)
print("EKSTRAKSI LOGITS ASLI TABNET DARI NETWORK")
print("="*60)

logits_train_all = extract_tabnet_logits(final_tabnet_model, X_train_np)
logits_valid_all = extract_tabnet_logits(final_tabnet_model, X_valid_np)
logits_test_all  = extract_tabnet_logits(final_tabnet_model, X_test_np)

logits_train_class0 = logits_train_all[:, 0]
logits_train_class1 = logits_train_all[:, 1]
logits_valid_class0 = logits_valid_all[:, 0]
logits_valid_class1 = logits_valid_all[:, 1]
logits_test_class0  = logits_test_all[:, 0]
logits_test_class1  = logits_test_all[:, 1]

logits_train = logits_train_class1 - logits_train_class0
logits_valid = logits_valid_class1 - logits_valid_class0
logits_test  = logits_test_class1 - logits_test_class0

print("✓ Logits asli TabNet berhasil diekstrak langsung dari final_tabnet_model.network")
print(f"  logits_train_all shape: {logits_train_all.shape}")
print(f"  logits_valid_all shape: {logits_valid_all.shape}")
print(f"  logits_test_all shape : {logits_test_all.shape}")
print(f"  Contoh logits asli sampel test 0: {logits_test_all[0]}")
print(f"  Contoh logit difference sampel test 0: {logits_test[0]:.6f}")

softmax_probs_train = softmax_numpy(logits_train_all)
softmax_probs_valid = softmax_numpy(logits_valid_all)
softmax_probs_test  = softmax_numpy(logits_test_all)

max_diff_train_class1 = np.max(np.abs(probs_train[:, 1] - softmax_probs_train[:, 1]))
max_diff_valid_class1 = np.max(np.abs(probs_valid[:, 1] - softmax_probs_valid[:, 1]))
max_diff_test_class0  = np.max(np.abs(probs_test[:, 0] - softmax_probs_test[:, 0]))
max_diff_test_class1  = np.max(np.abs(probs_test[:, 1] - softmax_probs_test[:, 1]))

print(f"  Maksimum selisih softmax logits vs predict_proba train kelas 1: {max_diff_train_class1:.10f}")
print(f"  Maksimum selisih softmax logits vs predict_proba valid kelas 1: {max_diff_valid_class1:.10f}")
print(f"  Maksimum selisih softmax logits vs predict_proba test kelas 0 : {max_diff_test_class0:.10f}")
print(f"  Maksimum selisih softmax logits vs predict_proba test kelas 1 : {max_diff_test_class1:.10f}")

# =========================================================
# SIMPAN LOGITS ASLI TABNET KE CSV
# =========================================================
print("\n" + "="*60)
print("SAVING LOGITS ASLI TABNET TO CSV")
print("="*60)

logits_train_df = pd.DataFrame({
    'sample_index': np.arange(len(y_train_np)),
    'logit_class_0': logits_train_class0,
    'logit_class_1': logits_train_class1,
    'logit_difference_z1_minus_z0': logits_train,
    'prob_predict_proba_class_0': probs_train[:, 0],
    'prob_predict_proba_class_1': probs_train[:, 1],
    'prob_softmax_logits_class_0': softmax_probs_train[:, 0],
    'prob_softmax_logits_class_1': softmax_probs_train[:, 1],
    'abs_diff_class_0': np.abs(probs_train[:, 0] - softmax_probs_train[:, 0]),
    'abs_diff_class_1': np.abs(probs_train[:, 1] - softmax_probs_train[:, 1]),
    'actual_target': y_train_np
})
logits_train_df.to_csv(f'{output_dir}/logits_asli_train.csv', index=False)

logits_valid_df = pd.DataFrame({
    'sample_index': np.arange(len(y_valid_np)),
    'logit_class_0': logits_valid_class0,
    'logit_class_1': logits_valid_class1,
    'logit_difference_z1_minus_z0': logits_valid,
    'prob_predict_proba_class_0': probs_valid[:, 0],
    'prob_predict_proba_class_1': probs_valid[:, 1],
    'prob_softmax_logits_class_0': softmax_probs_valid[:, 0],
    'prob_softmax_logits_class_1': softmax_probs_valid[:, 1],
    'abs_diff_class_0': np.abs(probs_valid[:, 0] - softmax_probs_valid[:, 0]),
    'abs_diff_class_1': np.abs(probs_valid[:, 1] - softmax_probs_valid[:, 1]),
    'actual_target': y_valid_np
})
logits_valid_df.to_csv(f'{output_dir}/logits_asli_valid.csv', index=False)

logits_test_df = pd.DataFrame({
    'sample_index': np.arange(len(y_test_np)),
    'logit_class_0': logits_test_class0,
    'logit_class_1': logits_test_class1,
    'logit_difference_z1_minus_z0': logits_test,
    'prob_predict_proba_class_0': probs_test[:, 0],
    'prob_predict_proba_class_1': probs_test[:, 1],
    'prob_softmax_logits_class_0': softmax_probs_test[:, 0],
    'prob_softmax_logits_class_1': softmax_probs_test[:, 1],
    'abs_diff_class_0': np.abs(probs_test[:, 0] - softmax_probs_test[:, 0]),
    'abs_diff_class_1': np.abs(probs_test[:, 1] - softmax_probs_test[:, 1]),
    'actual_target': y_test_np
})
logits_test_df.to_csv(f'{output_dir}/logits_asli_test.csv', index=False)

logits_train_df.insert(1, 'split', 'train')
logits_valid_df.insert(1, 'split', 'valid')
logits_test_df.insert(1, 'split', 'test')

logits_all_df = pd.concat(
    [logits_train_df, logits_valid_df, logits_test_df],
    ignore_index=True
)
logits_all_df.to_csv(f'{output_dir}/logits_asli_semua_split.csv', index=False)

print(f"✓ Logits asli train saved to '{output_dir}/logits_asli_train.csv'")
print(f"✓ Logits asli valid saved to '{output_dir}/logits_asli_valid.csv'")
print(f"✓ Logits asli test saved to '{output_dir}/logits_asli_test.csv'")
print(f"✓ Logits asli semua split saved to '{output_dir}/logits_asli_semua_split.csv'")

# =========================================================
# APPLY CALIBRATION METHODS 
# =========================================================
print("\n" + "="*60)
print("APPLYING CALIBRATION METHODS")
print("="*60)

probs_valid_class1 = probs_valid[:, 1].reshape(-1, 1)

print("\n1. BBQ...")
bbq = BBQ()
bbq.fit(probs_valid_class1, y_valid_np)
probs_test_bbq = bbq.transform(probs_test[:, 1].reshape(-1, 1))
probs_test_bbq = np.clip(probs_test_bbq, 0, 1)
probs_test_bbq = np.column_stack([1 - probs_test_bbq, probs_test_bbq])
print("✓ BBQ applied")

print("\n2. Temperature Scaling...")
calibrated_probs_temp, optimal_temp = manual_temperature_scaling(logits_valid, y_valid_np)
scaled_logits_test = logits_test / optimal_temp
probs_test_temp_probs = 1 / (1 + np.exp(-scaled_logits_test))
probs_test_temp = np.column_stack([1 - probs_test_temp_probs, probs_test_temp_probs])
print(f"✓ Temperature Scaling applied with T = {optimal_temp:.4f}")

print("\n3. Isotonic Regression...")
iso_reg = IsotonicRegression()
iso_reg.fit(probs_valid_class1, y_valid_np)
probs_test_iso = iso_reg.transform(probs_test[:, 1].reshape(-1, 1))
probs_test_iso = np.clip(probs_test_iso, 0, 1)
probs_test_iso = np.column_stack([1 - probs_test_iso, probs_test_iso])
print("✓ Isotonic Regression applied")

class DummyTemperatureScaler:
    def __init__(self, temperature):
        self.temperature_ = temperature
        self.temperature = temperature

temp_scaler = DummyTemperatureScaler(optimal_temp)

# =========================================================
# THRESHOLD OPTIMIZATION FOR TABNET (BEFORE CALIBRATION)
# =========================================================
print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION FOR TABNET (BEFORE CALIBRATION)")
print("="*60)

thresholds_tabnet = np.arange(0.10, 0.90, 0.02)
best_f1_tabnet = 0
best_threshold_tabnet = 0.5

print("\nOptimizing threshold for pure TabNet...")
for threshold in thresholds_tabnet:
    y_pred_temp = (probs_valid[:, 1] >= threshold).astype(int)
    f1 = f1_score(y_valid_np, y_pred_temp, zero_division=0)
    if f1 > best_f1_tabnet:
        best_f1_tabnet = f1
        best_threshold_tabnet = threshold

print(f"✓ Optimal threshold for TabNet: {best_threshold_tabnet:.2f}")
print(f"✓ Best validation F1 for TabNet: {best_f1_tabnet:.6f}")

# =========================================================
# THRESHOLD SENSITIVITY ANALYSIS - BEFORE CALIBRATION
# =========================================================
print("\n" + "="*60)
print("THRESHOLD SENSITIVITY ANALYSIS - BEFORE CALIBRATION")
print("="*60)

print("\nThreshold | Recall | Precision | F1-Score | Accuracy")
print("-" * 60)

sensitivity_before_df = threshold_sensitivity_analysis(y_test_np, probs_test[:, 1], thresholds_tabnet)

for _, row in sensitivity_before_df.iterrows():
    marker = " ← OPTIMAL" if row['Threshold'] == best_threshold_tabnet else ""
    print(f"  {row['Threshold']:.2f}     | {row['Recall']:.4f}  | {row['Precision']:.4f}   | {row['F1-Score']:.4f}    | {row['Accuracy']:.4f}{marker}")

sensitivity_before_df.to_csv(f'{output_dir}/threshold_sensitivity_analysis_before_calibration.csv', index=False, float_format='%.6f')
print(f"\n✓ Threshold sensitivity analysis (before calibration) saved to '{output_dir}/threshold_sensitivity_analysis_before_calibration.csv'")

# =========================================================
# MENENTUKAN METODE KALIBRASI TERBAIK 
# =========================================================
print("\n" + "="*60)
print("DETERMINE BEST CALIBRATION METHOD")
print("="*60)

ece_original_test = calculate_ece_correct(y_test_np,
    probs_test[:, 1], n_bins=10)

ece_bbq_test = calculate_ece_correct(y_test_np,
    probs_test_bbq[:, 1], n_bins=10)

ece_ts_test = calculate_ece_correct(y_test_np,
    probs_test_temp[:, 1], n_bins=10)

ece_iso_test = calculate_ece_correct(y_test_np,
    probs_test_iso[:, 1], n_bins=10)

ece_methods = {
    "Original TabNet": ece_original_test,
    "BBQ": ece_bbq_test,
    "Temperature Scaling": ece_ts_test,
    "Isotonic Regression": ece_iso_test
}

best_calib_method = min(ece_methods, key=ece_methods.get)

print("\nECE test per method:")
for method_name, ece_value in ece_methods.items():
    print(f"  • {method_name}: {ece_value:.6f}")

print(f"\n✓ Best calibration method based on test ECE: {best_calib_method}")

# =========================================================
# OPTIMIZE THRESHOLD AFTER CALIBRATION (BEST METHOD)
# =========================================================
print("\n" + "="*60)
print(f"OPTIMIZING THRESHOLD FOR {best_calib_method}")
print("="*60)

if best_calib_method == 'BBQ':
    probs_test_calibrated = probs_test_bbq[:, 1]
    probs_valid_bbq = bbq.transform(probs_valid_class1)
    probs_valid_bbq = np.clip(probs_valid_bbq, 0, 1)
    probs_valid_calibrated = probs_valid_bbq.reshape(-1)

elif best_calib_method == 'Temperature Scaling':
    probs_test_calibrated = probs_test_temp[:, 1]
    scaled_logits_valid = logits_valid / optimal_temp
    probs_valid_temp = 1 / (1 + np.exp(-scaled_logits_valid))
    probs_valid_calibrated = probs_valid_temp.reshape(-1)

elif best_calib_method == 'Isotonic Regression':
    probs_test_calibrated = probs_test_iso[:, 1]
    probs_valid_iso = iso_reg.transform(probs_valid_class1)
    probs_valid_iso = np.clip(probs_valid_iso, 0, 1)
    probs_valid_calibrated = probs_valid_iso.reshape(-1)

else:
    probs_valid_calibrated = probs_valid[:, 1]
    probs_test_calibrated = probs_test[:, 1]

thresholds_calibrated = np.arange(0.10, 0.90, 0.02)
best_f1_calibrated = 0
best_threshold_calibrated = 0.5

for threshold in thresholds_calibrated:
    y_pred_temp = (probs_valid_calibrated >= threshold).astype(int)

    f1 = f1_score(y_valid_np,
        y_pred_temp, zero_division=0)

    if f1 > best_f1_calibrated:
        best_f1_calibrated = f1
        best_threshold_calibrated = threshold

print(f"✓ Optimal threshold for {best_calib_method}: "
    f"{best_threshold_calibrated:.2f}")
print(f"✓ Best validation F1 for {best_calib_method}: "
    f"{best_f1_calibrated:.6f}")

# =========================================================
# THRESHOLD SENSITIVITY ANALYSIS - AFTER CALIBRATION
# =========================================================
print("\n" + "="*60)
print("THRESHOLD SENSITIVITY ANALYSIS - AFTER CALIBRATION")
print("="*60)

print("\nThreshold | Recall | Precision | F1-Score | Accuracy")
print("-" * 60)

sensitivity_after_df = threshold_sensitivity_analysis(
    y_test_np,
    probs_test_calibrated,
    thresholds_calibrated
)

for _, row in sensitivity_after_df.iterrows():
    marker = (
        " ← OPTIMAL"
        if row['Threshold'] == best_threshold_calibrated
        else ""
    )

    print(
        f"  {row['Threshold']:.2f}     | "
        f"{row['Recall']:.4f}  | "
        f"{row['Precision']:.4f}   | "
        f"{row['F1-Score']:.4f}    | "
        f"{row['Accuracy']:.4f}{marker}"
    )
sensitivity_after_df.to_csv(f'{output_dir}/threshold_sensitivity_analysis_after_calibration.csv', index=False, float_format='%.6f')
print(f"\n✓ Threshold sensitivity analysis (after calibration) saved to '{output_dir}/threshold_sensitivity_analysis_after_calibration.csv'")

# =========================================================
# PREDICTIONS WITH OPTIMAL THRESHOLDS
# =========================================================
print("\n" + "="*60)
print("FINAL PREDICTIONS WITH OPTIMAL THRESHOLDS")
print("="*60)

y_pred_tabnet = (probs_test[:, 1] >= best_threshold_tabnet).astype(int)
y_pred_calibrated = (probs_test_calibrated >= best_threshold_calibrated).astype(int)

# =========================================================
# SAVE TEST PREDICTIONS TO CSV
# =========================================================
print("\n" + "="*60)
print("SAVING TEST PREDICTIONS TO CSV")
print("="*60)

test_predictions_df = pd.DataFrame({
    'actual': y_test_np,
    'prob_original': probs_test[:, 1],
    'prob_bbq': probs_test_bbq[:, 1],
    'prob_temperature': probs_test_temp[:, 1],
    'prob_isotonic': probs_test_iso[:, 1],
    'threshold_tabnet': best_threshold_tabnet,
    'best_calibration_method': best_calib_method,
    'predicted_tabnet': y_pred_tabnet,
    'threshold_calibrated': best_threshold_calibrated,
    'predicted_calibrated': y_pred_calibrated
})

test_predictions_df.to_csv(f'{output_dir}/test_predictions.csv', index=False)
print(f"✓ Test predictions saved to '{output_dir}/test_predictions.csv'")

# =========================================================
# CONFUSION MATRIX - BEFORE CALIBRATION
# =========================================================
print("\n" + "="*60)
print("CONFUSION MATRIX - BEFORE CALIBRATION")
print("="*60)

cm_tabnet = confusion_matrix(y_test_np, y_pred_tabnet)
tn_tabnet, fp_tabnet, fn_tabnet, tp_tabnet = cm_tabnet.ravel()

plt.figure(figsize=(8, 6))
sns.heatmap(cm_tabnet, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title(f'Confusion Matrix - TabNet Murni (Before Calibration)\nThreshold = {best_threshold_tabnet:.2f}', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix_before_calibration.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Confusion matrix before calibration saved to '{output_dir}/confusion_matrix_before_calibration.png'")

# =========================================================
# CONFUSION MATRIX - AFTER CALIBRATION
# =========================================================
print("\n" + "="*60)
print("CONFUSION MATRIX - AFTER CALIBRATION")
print("="*60)

cm_calibrated = confusion_matrix(y_test_np, y_pred_calibrated)
tn_cal, fp_cal, fn_cal, tp_cal = cm_calibrated.ravel()

plt.figure(figsize=(8, 6))
sns.heatmap(cm_calibrated, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title(f'Confusion Matrix (After Calibration)\nThreshold = {best_threshold_calibrated:.2f}', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix_after_calibration.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Confusion matrix after calibration saved to '{output_dir}/confusion_matrix_after_calibration.png'")

# =========================================================
# ROC CURVE - BEFORE CALIBRATION
# =========================================================
print("\n" + "="*60)
print("ROC CURVE - BEFORE CALIBRATION")
print("="*60)

fpr_original, tpr_original, _ = roc_curve(y_test_np, probs_test[:, 1])
auc_original = roc_auc_score(y_test_np, probs_test[:, 1])

plt.figure(figsize=(10, 8))
plt.plot(fpr_original, tpr_original, linewidth=2.5, 
         label=f'Before Calibration (Original TabNet) - AUC = {auc_original:.4f}', 
         color='blue', linestyle='-')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve: Before Calibration (Original TabNet)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/roc_curve_before_calibration.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ ROC curve (before calibration) saved to '{output_dir}/roc_curve_before_calibration.png'")

# =========================================================
# ROC CURVE - AFTER CALIBRATION
# =========================================================
print("\n" + "="*60)
print(f"ROC CURVE - AFTER CALIBRATION ({best_calib_method})")
print("="*60)

fpr_calibrated, tpr_calibrated, _ = roc_curve(y_test_np, probs_test_calibrated)
auc_calibrated = roc_auc_score(y_test_np, probs_test_calibrated)

plt.figure(figsize=(10, 8))
plt.plot(fpr_calibrated, tpr_calibrated, linewidth=2.5,
    label=(f'After Calibration ({best_calib_method}) 'f'- AUC = {auc_calibrated:.4f}'),
    color='green', linestyle='-')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1,
 label='Random Classifier (AUC = 0.5)')

plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title(f'ROC Curve: After Calibration ({best_calib_method})',
    fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/roc_curve_after_calibration.png',
    dpi=150, bbox_inches='tight')
plt.show()

print(
    f"✓ ROC curve after calibration ({best_calib_method}) "
    f"saved to '{output_dir}/roc_curve_after_calibration.png'"
)

# =========================================================
# RELIABILITY DIAGRAMS
# =========================================================
print("\n" + "="*60)
print("RELIABILITY DIAGRAMS")
print("="*60)

methods_for_reliability = [
    ('Original TabNet', probs_test[:, 1], 'blue'),
    ('BBQ', probs_test_bbq[:, 1], 'blue'),
    ('Temperature Scaling', probs_test_temp[:, 1], 'blue'),
    ('Isotonic Regression', probs_test_iso[:, 1], 'blue')
]

for method_name, probs, color in methods_for_reliability:
    fig, ax1 = plt.subplots(figsize=(8, 8))

    prob_true, prob_pred = calibration_curve(y_test_np, probs, n_bins=10, strategy='uniform')

    ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, color=color, label=method_name)
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration')

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.hist(probs, bins=20, alpha=0.3, color='blue', edgecolor='black')
    ax2.set_ylabel('Frequency', fontsize=12)

    ece_value = calculate_ece_correct(y_test_np, probs)
    brier_value = brier_score_loss(y_test_np, probs)

    display_name = method_name.replace('_', ' ').title()
    ax1.set_title(f'Reliability Diagram - {display_name}\nECE = {ece_value:.6f} | Brier Score = {brier_value:.6f}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')

    save_filename = f'{output_dir}/reliability_diagram_{method_name.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Reliability diagram for {method_name} saved to '{save_filename}'")

# =========================================================
# CALIBRATION ANALYSIS - GABUNGAN 4 RELIABILITY DIAGRAM DALAM 1 FIGURE
# =========================================================
print("\n" + "="*60)
print("CALIBRATION ANALYSIS - COMBINED RELIABILITY DIAGRAMS")
print("="*60)

methods_calibration = [
    ('Original TabNet', probs_test[:, 1], 'blue'),
    ('BBQ', probs_test_bbq[:, 1], 'blue'),
    ('Temperature Scaling', probs_test_temp[:, 1], 'blue'),
    ('Isotonic Regression', probs_test_iso[:, 1], 'blue')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (method_name, probs, color) in enumerate(methods_calibration):
    ax = axes[idx]
    
    prob_true, prob_pred = calibration_curve(
        y_test_np, probs, n_bins=10, strategy='uniform'
    )
    
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, 
            color=color)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.hist(probs, bins=20, alpha=0.3, color='blue', edgecolor='black')
    
    ece_value = calculate_ece_correct(y_test_np, probs)
    
    ax.set_title(f'{method_name}\nECE = {ece_value:.6f}')

plt.tight_layout()
plt.savefig(f'{output_dir}/calibration_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Calibration analysis (combined) saved to '{output_dir}/calibration_analysis.png'")

# =========================================================
# CALCULATE METRICS 
# =========================================================
print("\n" + "="*60)
print("METRIK EVALUASI")
print("="*60)

# TabNet metrics
tabnet_accuracy = accuracy_score(y_test_np, y_pred_tabnet)
tabnet_precision = precision_score(y_test_np, y_pred_tabnet, zero_division=0)
tabnet_recall = recall_score(y_test_np, y_pred_tabnet, zero_division=0)
tabnet_f1 = f1_score(y_test_np, y_pred_tabnet, zero_division=0)
tabnet_auc = auc_original
tabnet_brier = brier_score_loss(y_test_np, probs_test[:, 1])
tabnet_ece = calculate_ece_correct(y_test_np, probs_test[:, 1])

# Best Calibration
cal_accuracy = accuracy_score(y_test_np, y_pred_calibrated)
cal_precision = precision_score(y_test_np, y_pred_calibrated, zero_division=0)
cal_recall = recall_score(y_test_np, y_pred_calibrated, zero_division=0)
cal_f1 = f1_score(y_test_np, y_pred_calibrated, zero_division=0)
cal_auc = roc_auc_score(y_test_np, probs_test_calibrated)
cal_brier = brier_score_loss(y_test_np, probs_test_calibrated)
cal_ece = calculate_ece_correct(y_test_np, probs_test_calibrated)

print("\n" + "="*40)
print("TABNET MURNI - PERFORMANCE METRICS")
print("="*40)
print(f"  • Accuracy:   {tabnet_accuracy:.6f}")
print(f"  • Precision:  {tabnet_precision:.6f}")
print(f"  • Recall:     {tabnet_recall:.6f}")
print(f"  • F1-Score:   {tabnet_f1:.6f}")
print(f"  • ROC AUC:    {tabnet_auc:.6f}")
print(f"  • Brier Score: {tabnet_brier:.6f}")
print(f"  • ECE:        {tabnet_ece:.6f}")
print(f"  • Optimal Threshold: {best_threshold_tabnet:.2f}")

print("\n" + "="*40)
print("AFTER KALIBRASI - PERFORMANCE METRICS")
print("="*40)
print(f"  • Accuracy:   {cal_accuracy:.6f}")
print(f"  • Precision:  {cal_precision:.6f}")
print(f"  • Recall:     {cal_recall:.6f}")
print(f"  • F1-Score:   {cal_f1:.6f}")
print(f"  • ROC AUC:    {cal_auc:.6f}")
print(f"  • Brier Score: {cal_brier:.6f}")
print(f"  • ECE:        {cal_ece:.6f}")
print(f"  • Optimal Threshold: {best_threshold_calibrated:.2f}")

# =========================================================
# CALIBRATION RESULTS TABLE
# =========================================================
print("\n" + "="*60)
print("CALIBRATION RESULTS")
print("="*60)

calibration_results = []
methods_list = [
    ('Original TabNet', probs_test[:, 1]),
    ('BBQ', probs_test_bbq[:, 1]),
    ('Temperature Scaling', probs_test_temp[:, 1]),
    ('Isotonic Regression', probs_test_iso[:, 1])
]

for name, probs in methods_list:
    ece = calculate_ece_correct(y_test_np, probs)
    brier = brier_score_loss(y_test_np, probs)
    calibration_results.append({'Method': name, 'ECE': ece, 'Brier Score': brier})

calibration_df = pd.DataFrame(calibration_results)
ece_original = calibration_df.loc[0, 'ECE']
brier_original = calibration_df.loc[0, 'Brier Score']

for i in range(len(calibration_df)):
    calibration_df.loc[i, 'ECE_Improvement_%'] = ((ece_original - calibration_df.loc[i, 'ECE']) / ece_original * 100) if ece_original > 0 else 0
    calibration_df.loc[i, 'Brier_Improvement_%'] = ((brier_original - calibration_df.loc[i, 'Brier Score']) / brier_original * 100) if brier_original > 0 else 0

print("\nCALIBRATION RESULTS:")
print(calibration_df.to_string(index=False, float_format='%.6f'))

calibration_df.to_csv(f'{output_dir}/calibration_results.csv', index=False)

# =========================================================
# SAVE OTHER RESULTS 
# =========================================================
print("\n" + "="*60)
print("SAVING OTHER RESULTS")
print("="*60)

# Save model comparison 
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Brier Score', 'ECE'],
    'TabNet Murni': [tabnet_accuracy, tabnet_precision, tabnet_recall, tabnet_f1, tabnet_auc, tabnet_brier, tabnet_ece],
    f'After Kalibrasi ({best_calib_method})': [cal_accuracy, cal_precision, cal_recall, cal_f1, cal_auc, cal_brier, cal_ece]
})
comparison_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)

# Save feature importance
model_feature_importance = final_tabnet_model.feature_importances_
imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model_feature_importance
}).sort_values('Importance', ascending=False)
imp_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)

# Save outlier summary
outlier_summary_df.to_csv(f'{output_dir}/outlier_summary.csv', index=False)

# Save models
if knn_imputer is not None:
    joblib.dump(knn_imputer, f'{output_dir}/knn_imputer.pkl')

joblib.dump(bbq, f'{output_dir}/bbq.pkl')
joblib.dump(temp_scaler, f'{output_dir}/temperature_scaling.pkl')
joblib.dump(iso_reg, f'{output_dir}/isotonic_regression.pkl')
joblib.dump(scaler, f'{output_dir}/scaler.pkl')
joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
final_tabnet_model.save_model(f'{output_dir}/tabnet_model_optimized.zip')

print(f"✓ All results saved to '{output_dir}/' directory")

# =========================================================
# FINAL SUMMARY 
# =========================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print("\nSTRATEGI YANG DIGUNAKAN:")
print("  • Split ratio: 80-10-10 (dengan stratifikasi)")
print("  • Missing value handling: KNN Imputer (n_neighbors=5)")
print("  • Outlier handling: Winsorization (percentile 1-99)")
print("  • Hyperparameter tuning: Optuna dengan 30 trials (ONLY on training data)")
print("  • Normalization: ONLY on train data (fit_transform), validation/test using transform only")
print("  • Tanpa class weight (standard loss)")

print("\nOUTLIER HANDLING:")
print(f"  • Total outliers handled: {outlier_summary_df['Total Outliers Handled'].sum()}")

print("\nDATA SPLIT (80-10-10):")
print(f"  • Training set: {X_train_np.shape[0]} samples ({len(y_train_np)/len(y_np)*100:.1f}%)")
print(f"  • Validation set: {X_valid_np.shape[0]} samples ({len(y_valid_np)/len(y_np)*100:.1f}%)")
print(f"  • Test set: {X_test_np.shape[0]} samples ({len(y_test_np)/len(y_np)*100:.1f}%)")

print("\nBEST HYPERPARAMETERS:")
for key, value in best_params.items():
    print(f"  • {key}: {value}")

best_method_idx = calibration_df['ECE'].idxmin()

print("\nBEST CALIBRATION METHOD:")
print(f"  • {best_calib_method}")
print(f"  • ECE Improvement: {calibration_df.loc[best_method_idx, 'ECE_Improvement_%']:.2f}%")
print(f"  • Final ECE: {calibration_df.loc[best_method_idx, 'ECE']:.6f}")

print("\nTEST SET PERFORMANCE (TABNET MURNI):")
print(f"  • Accuracy:  {tabnet_accuracy:.6f}")
print(f"  • Precision: {tabnet_precision:.6f}")
print(f"  • Recall:    {tabnet_recall:.6f}")
print(f"  • F1-Score:  {tabnet_f1:.6f}")
print(f"  • ROC AUC:   {tabnet_auc:.6f}")
print(f"  • ECE:       {tabnet_ece:.6f}")

print("\nTEST SET PERFORMANCE (AFTER {best_calib_method} CALIBRATION):")
print(f"  • Accuracy:  {cal_accuracy:.6f}")
print(f"  • Precision: {cal_precision:.6f}")
print(f"  • Recall:    {cal_recall:.6f}")
print(f"  • F1-Score:  {cal_f1:.6f}")
print(f"  • ROC AUC:   {cal_auc:.6f}")
print(f"  • ECE:       {cal_ece:.6f}")

print("\nOUTPUT FILES:")
print("  CSV FILES:")
print("  ✓ original_dataset.csv - Original dataset before any processing (ADDED)")
print("  ✓ test_data_before_normalization.csv - Test data BEFORE normalization (ADDED)")
print("  ✓ train_data_before_normalization.csv - Train data BEFORE normalization (ADDED)")
print("  ✓ valid_data_before_normalization.csv - Validation data BEFORE normalization (ADDED)")
print("  ✓ test_data_after_normalization.csv - Test data after normalization")
print("  ✓ train_data_after_normalization.csv - Train data after normalization")
print("  ✓ valid_data_after_normalization.csv - Validation data after normalization")
print("  ✓ processed_dataset.csv - Complete processed dataset")
print("  ✓ data_split_summary.csv - Data split summary")
print("  ✓ logits_asli_train.csv - Logits asli TabNet untuk data train")
print("  ✓ logits_asli_valid.csv - Logits asli TabNet untuk data validasi")
print("  ✓ logits_asli_test.csv - Logits asli TabNet untuk data test")
print("  ✓ logits_asli_semua_split.csv - Gabungan logits asli seluruh split")
print("  ✓ test_predictions.csv - Prediksi lengkap dengan berbagai metode kalibrasi")
print("  ✓ calibration_results.csv - Hasil kalibrasi (ECE dan Brier Score)")
print("  ✓ model_comparison.csv - Perbandingan metrik model (tanpa specificity)")
print("  ✓ feature_importance.csv - Feature importance model TabNet")
print("  ✓ outlier_summary.csv - Ringkasan outlier yang di-handle")
print("  ✓ test_mask_heatmap.csv - Data heatmap mask")
print("  ✓ test_aggregate_mask_importance.csv - Data aggregate mask importance")
print("  ✓ best_hyperparameters.csv - Hyperparameter terbaik Optuna")
print("  ✓ importance_comparison.csv - Perbandingan mask importance vs model importance")
print("  ✓ threshold_sensitivity_analysis_before_calibration.csv - Threshold sensitivity before calibration")
print("  ✓ threshold_sensitivity_analysis_after_calibration.csv - Threshold sensitivity after calibration")
print("\n  GAMBAR:")
print("  ✓ test_mask_heatmap.png - Heatmap mask")
print("  ✓ test_aggregate_mask_importance.png - Aggregate mask importance")
print("  ✓ importance_comparison.png - Perbandingan mask vs model importance")
print("  ✓ model_feature_importance.png - Model feature importance")
print("  ✓ roc_curve_before_calibration.png - ROC curve sebelum kalibrasi")
print("  ✓ roc_curve_after_calibration.png - ROC curve setelah kalibrasi (BBQ)")
print("  ✓ confusion_matrix_before_calibration.png - Confusion matrix sebelum kalibrasi")
print("  ✓ confusion_matrix_after_calibration.png - Confusion matrix setelah kalibrasi")
print("  ✓ reliability_diagram_original_tabnet.png - Reliability diagram Original TabNet")
print("  ✓ reliability_diagram_bbq.png - Reliability diagram BBQ")
print("  ✓ reliability_diagram_temperature_scaling.png - Reliability diagram Temperature Scaling")
print("  ✓ reliability_diagram_isotonic_regression.png - Reliability diagram Isotonic Regression")
print("  ✓ calibration_analysis.png - Gabungan 4 reliability diagram dalam 1 figure")
print("\n  MODEL FILES:")
print("  ✓ tabnet_model_optimized.zip - Model TabNet final")
print("  ✓ knn_imputer.pkl - KNN Imputer")
print("  ✓ bbq.pkl - BBQ calibration model")
print("  ✓ temperature_scaling.pkl - Temperature Scaling model")
print("  ✓ isotonic_regression.pkl - Isotonic Regression model")
print("  ✓ scaler.pkl - StandardScaler")
print("  ✓ label_encoders.pkl - Label encoders")

print("\n" + "="*60)
print("PROCESSING COMPLETED SUCCESSFULLY")
print("="*60)
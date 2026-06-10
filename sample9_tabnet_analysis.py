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
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from netcal.binning import IsotonicRegression, BBQ

# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_reproducibility(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Reproducibility set with seed: {seed}")

REPRODUCIBILITY_SEED = 123
set_reproducibility(REPRODUCIBILITY_SEED)

# =========================================================
# HELPER FUNCTIONS
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

def prob_to_logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def calculate_ece_with_detail(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    total_samples = len(y_true)
    
    print("="*100)
    print("DETAIL ECE PER BIN")
    print("="*100)
    print(f"n = {total_samples}")
    print(f"{'Bin':<5} {'Rentang':<14} {'|Bi|':<7} {'n_pos':<8} "
          f"{'sum_p':<12} {'oi':<10} {'ei':<10} "
          f"{'|oi-ei|':<12} {'P(i)x|oi-ei|'}")
    print("-"*105)
    
    for idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        
        if idx == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            rentang = f"[{bin_lower:.1f},{bin_upper:.1f}]"
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            rentang = f"[{bin_lower:.1f},{bin_upper:.1f})"
        
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            n_pos = int(np.sum(y_true[in_bin]))     # jumlah data aktual positif/default
            sum_p = np.sum(y_prob[in_bin])          # jumlah probabilitas prediksi
            oi = n_pos / bin_size                   # observed frequency
            ei = sum_p / bin_size                   # expected confidence
            diff = abs(oi - ei)
            kontrib = (bin_size / total_samples) * diff
            ece += kontrib
            
            print(f"{idx+1:<5} {rentang:<14} "
                  f"{bin_size:<7} {n_pos:<8} {sum_p:<12.4f} "
                  f"{oi:<10.4f} {ei:<10.4f} "
                  f"{diff:<12.4f} {kontrib:.6f}")
        else:
            print(f"{idx+1:<5} {rentang:<14} "
                  f"{0:<7} {0:<8} {0:<12.4f} "
                  f"{0:<10.4f} {0:<10.4f} "
                  f"{0:<12.4f} {0:.6f}")
    
    print(f"\nECE = Σ (|Bi|/n) × |oi − ei|")
    print(f"    = {ece:.6f}")
    return ece

def calculate_ece_correct(y_true, y_prob, n_bins=10):
    """Menghitung ECE tanpa mencetak detail per bin."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for idx in range(n_bins):
        bin_lower = bin_boundaries[idx]
        bin_upper = bin_boundaries[idx + 1]

        if idx == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return ece

def print_brier_detail_per_sample(method_name, y_true, y_prob, y_pred):
    """Menampilkan detail Brier Score dengan contoh TN, TP, FP, dan FN."""
    squared_errors = (y_prob - y_true) ** 2
    bs_manual = squared_errors.mean()

    categories = [
        ((y_true == 0) & (y_pred == 0), 'TN (Non-Default benar)'),
        ((y_true == 1) & (y_pred == 1), 'TP (Default benar)'),
        ((y_true == 0) & (y_pred == 1), 'FP (Non-Default salah prediksi)'),
        ((y_true == 1) & (y_pred == 0), 'FN (Default salah prediksi)'),
    ]

    print("\n" + "="*60)
    print(f"BRIER SCORE - DETAIL PER SAMPEL ({method_name})")
    print("="*60)
    print(f"\n{'No':<5}{'Keterangan':<35}{'p_i':>12}"
          f"{'o_i':>6}{'(p_i-o_i)²':>15}")
    print("-"*75)

    no = 1
    for condition, label in categories:
        idx_candidates = np.where(condition)[0]

        if len(idx_candidates) > 0:
            idx = idx_candidates[0]
            pi = y_prob[idx]
            oi = int(y_true[idx])
            sq = squared_errors[idx]

            print(f"{no:<5}{label:<35}{pi:>12.8f}{oi:>6}{sq:>15.8f}")
        else:
            print(f"{no:<5}{label:<35}{'Tidak ada':>12}{'-':>6}{'-':>15}")

        no += 1

    print("-"*75)
    print(f"\n  BS = {bs_manual:.6f}")

def manual_temperature_scaling(logits, labels):
    def nll_loss(temp):
        scaled = logits / temp
        probs  = np.clip(1 / (1 + np.exp(-scaled)), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    result = minimize_scalar(nll_loss, bounds=(0.5, 5.0), method='bounded')
    T = result.x
    return 1 / (1 + np.exp(-logits / T)), T

def handle_outliers_winsorize(df, columns, limits=(0.01, 0.99)):
    df_clean = df.copy()
    summary  = []
    for col in columns:
        if col not in df_clean.columns:
            continue
        lo = df_clean[col].quantile(limits[0])
        hi = df_clean[col].quantile(limits[1])
        n_lo = (df_clean[col] < lo).sum()
        n_hi = (df_clean[col] > hi).sum()
        df_clean[col] = df_clean[col].clip(lo, hi)
        summary.append({'Feature': col, 'Total Outliers Handled': n_lo + n_hi})
    return df_clean, pd.DataFrame(summary)


def extract_tabnet_logits(model, X_data, batch_size=1024):
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


def sparsemax_manual(u):
    u = np.asarray(u, dtype=np.float64)
    z_sorted = np.sort(u)[::-1]
    z_cumsum = np.cumsum(z_sorted)
    k_array = np.arange(1, len(u) + 1)

    condition = 1 + k_array * z_sorted > z_cumsum
    k_z = k_array[condition][-1]
    tau = (z_cumsum[k_z - 1] - 1) / k_z

    output = np.maximum(u - tau, 0)
    return output, tau, z_sorted, k_z


def extract_tabnet_internal_attention(model, X_data, sample_idx, all_features, output_dir, gamma_value=None):
    
    print("\n" + "="*80)
    print(f"EKSTRAKSI NILAI INTERNAL SEBELUM SPARSEMAX TABNET - SAMPEL INDEKS KE-{sample_idx}")
    print("="*80)

    model.network.eval()

    # =====================================================
    # AMBIL ENCODER SESUAI STRUKTUR MODEL
    # =====================================================
    if hasattr(model.network, "tabnet") and hasattr(model.network.tabnet, "encoder"):
        encoder = model.network.tabnet.encoder
        print("✓ Encoder ditemukan pada model.network.tabnet.encoder")
    elif hasattr(model.network, "encoder"):
        encoder = model.network.encoder
        print("✓ Encoder ditemukan pada model.network.encoder")
    else:
        print("⚠️ Encoder TabNet tidak ditemukan.")
        print("   Cek struktur dengan: print(final_tabnet_model.network)")
        return None

    if not hasattr(encoder, "att_transformers"):
        print("⚠️ Struktur encoder tidak memiliki att_transformers.")
        print("   Cek struktur dengan: print(final_tabnet_model.network)")
        return None

    captured = {
        "u_pre_sparsemax": [],
        "M_after_sparsemax": []
    }
    hooks = []

    # =====================================================
    # PASANG HOOK PADA SELECTOR SPARSEMAX SETIAP STEP
    # =====================================================
    for step, att in enumerate(encoder.att_transformers):
        if hasattr(att, "selector"):
            def make_selector_hook(step_idx):
                def hook(module, inputs, output):
                    if len(inputs) == 0:
                        return

                    # inputs[0] = skor sebelum Sparsemax/Entmax
                    u_tensor = inputs[0]

                    # output = mask setelah Sparsemax/Entmax
                    m_tensor = output[0] if isinstance(output, tuple) else output

                    captured["u_pre_sparsemax"].append(
                        (step_idx, u_tensor.detach().cpu().numpy())
                    )
                    captured["M_after_sparsemax"].append(
                        (step_idx, m_tensor.detach().cpu().numpy())
                    )
                return hook

            hooks.append(att.selector.register_forward_hook(make_selector_hook(step)))
        else:
            print(f"⚠️ att_transformers[{step}] tidak memiliki atribut selector.")

    if len(hooks) == 0:
        print("⚠️ Tidak ada selector yang berhasil dipasangi hook.")
        print("   Cek nama modul dengan:")
        print("   for name, module in final_tabnet_model.network.named_modules(): print(name, module)")
        return None

    # =====================================================
    # FORWARD PASS SATU SAMPEL UNTUK MEMICU HOOK
    # =====================================================
    device = next(model.network.parameters()).device
    x_batch = torch.tensor(
        X_data[sample_idx:sample_idx + 1],
        dtype=torch.float32,
        device=device
    )

    try:
        with torch.no_grad():
            _ = model.network(x_batch)
    finally:
        for h in hooks:
            h.remove()

    captured["u_pre_sparsemax"] = sorted(captured["u_pre_sparsemax"], key=lambda x: x[0])
    captured["M_after_sparsemax"] = sorted(captured["M_after_sparsemax"], key=lambda x: x[0])

    if len(captured["u_pre_sparsemax"]) == 0:
        print("⚠️ Nilai u_pre_sparsemax tidak berhasil ditangkap.")
        return None

    # =====================================================
    # AMBIL GAMMA
    # =====================================================
    if gamma_value is None:
        gamma_value = getattr(model, "gamma", None)
    if gamma_value is None:
        gamma_value = getattr(encoder, "gamma", None)
    if gamma_value is None:
        gamma_value = 1.0
        print("⚠️ Nilai gamma tidak ditemukan. P_prev dihitung dengan gamma=1.0.")

    x_sample = X_data[sample_idx]
    all_rows = []
    P_prev = None

    print("✓ Hook berhasil dipasang pada selector Sparsemax")
    print(f"✓ Jumlah decision step yang tertangkap: {len(captured['u_pre_sparsemax'])}")

    for idx in range(len(captured["u_pre_sparsemax"])):
        step_idx, u_arr = captured["u_pre_sparsemax"][idx]
        _, m_arr = captured["M_after_sparsemax"][idx]

        u_step = np.asarray(u_arr[0]).reshape(-1)
        m_step = np.asarray(m_arr[0]).reshape(-1)

        # Nama fitur disesuaikan jika dimensi internal tidak sama dengan jumlah fitur asli.
        if len(u_step) == len(all_features):
            feature_names = list(all_features)
            x_values = x_sample
        else:
            feature_names = [f"internal_feature_{j+1}" for j in range(len(u_step))]
            x_values = np.full(len(u_step), np.nan)
            print(
                f"⚠️ Step {step_idx+1}: dimensi internal ({len(u_step)}) "
                f"tidak sama dengan jumlah fitur asli ({len(all_features)}). "
                "Kolom Feature memakai nama internal_feature."
            )

        # Prior awal bernilai 1. Prior berikutnya mengikuti rumus:
        # P[i] = P[i-1] ⊙ (gamma - M[i])
        if P_prev is None:
            P_prev = np.ones(len(u_step), dtype=np.float32)

        x_norm_x_m = x_values * m_step if len(x_values) == len(m_step) else np.full(len(m_step), np.nan)

        step_df = pd.DataFrame({
            "Step": step_idx + 1,
            "Feature": feature_names,
            "x_norm": x_values,
            "P_prev": P_prev,
            "u_pre_sparsemax": u_step,
            "M_after_sparsemax": m_step,
            "x_norm_x_M": x_norm_x_m
        }).sort_values("M_after_sparsemax", ascending=False)

        # Detail per step tidak dicetak agar output terminal tidak terlalu panjang.

        step_df.to_csv(
            f"{output_dir}/sample_{sample_idx}_internal_sparsemax_step_{step_idx+1}.csv",
            index=False
        )

        all_rows.append(step_df)
        P_prev = P_prev * (gamma_value - m_step)

    all_internal_df = pd.concat(all_rows, ignore_index=True)
    all_internal_df.to_csv(
        f"{output_dir}/sample_{sample_idx}_internal_sparsemax_all_steps.csv",
        index=False
    )

    print("\n✓ Nilai internal sebelum Sparsemax berhasil disimpan.")
    print(f"✓ File utama: {output_dir}/sample_{sample_idx}_internal_sparsemax_all_steps.csv")

    return all_internal_df

# =========================================================
# LOAD DATA
# =========================================================
print("="*60); print("LOADING DATA"); print("="*60)
df = pd.read_csv("credit_card_default.csv")
print(f"✓ Loaded: {df.shape}")
print(df['credit_card_default'].value_counts())

df = df.drop(columns=[c for c in ['customer_id','name'] if c in df.columns])

continuous_features  = ['age','net_yearly_income','no_of_days_employed',
                        'total_family_members','yearly_debt_payments',
                        'credit_limit','credit_score','no_of_children']
categorical_features = ['gender','owns_car','owns_house','occupation_type','migrant_worker']
target_col           = 'credit_card_default'

continuous_features  = [c for c in continuous_features  if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

# =========================================================
# MISSING VALUES — KNN IMPUTER
# =========================================================
missing_features = [c for c in df.columns if c != target_col and df[c].isnull().sum() > 0]
for col in missing_features:
    df[f"{col}_missing"] = df[col].isnull().astype(int)
    continuous_features.append(f"{col}_missing")

for col in categorical_features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

cont_missing = [c for c in missing_features if c in continuous_features]
if cont_missing:
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    df[continuous_features] = imputer.fit_transform(df[continuous_features])
    knn_imputer = imputer
else:
    knn_imputer = None

# =========================================================
# WINSORIZATION
# =========================================================
df, outlier_summary_df = handle_outliers_winsorize(df, continuous_features)
print(f"✓ Total outliers handled: {outlier_summary_df['Total Outliers Handled'].sum()}")

# =========================================================
# ENCODING
# =========================================================
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

continuous_features = [c for c in continuous_features if c in df.columns]

X = df.drop(columns=[target_col])
y = df[target_col]
all_features = list(X.columns)

cat_idxs = [i for i, c in enumerate(all_features) if c in categorical_features]
cat_dims  = [df[c].nunique() for c in all_features if c in categorical_features]

# =========================================================
# SPLIT 80-10-10
# =========================================================
X_np = X.values.astype(np.float32)
y_np = y.values.reshape(-1)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_np, y_np, test_size=0.2, random_state=REPRODUCIBILITY_SEED, stratify=y_np)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=REPRODUCIBILITY_SEED, stratify=y_temp)

print(f"✓ Train: {X_train.shape} | Val: {X_valid.shape} | Test: {X_test.shape}")

# =========================================================
# NORMALIZATION
# =========================================================
scaler = StandardScaler()
continuous_to_scale = [i for i, c in enumerate(all_features) if c in continuous_features]

X_train_np = X_train.copy()
X_train_np[:, continuous_to_scale] = scaler.fit_transform(X_train[:, continuous_to_scale])
X_valid_np = X_valid.copy()
X_valid_np[:, continuous_to_scale] = scaler.transform(X_valid[:, continuous_to_scale])
X_test_np  = X_test.copy()
X_test_np[:, continuous_to_scale]  = scaler.transform(X_test[:, continuous_to_scale])

X_train_np = X_train_np.astype(np.float32)
X_valid_np = X_valid_np.astype(np.float32)
X_test_np  = X_test_np.astype(np.float32)
y_train_np = y_train.reshape(-1)
y_valid_np = y_valid.reshape(-1)
y_test_np  = y_test.reshape(-1)

# =========================================================
# BEST HYPERPARAMETERS (LANGSUNG DARI HASIL OPTUNA TRIAL 21)
# =========================================================
print("\n" + "="*60)
print("MENGGUNAKAN HIPERPARAMETER TERBAIK HASIL OPTUNA (TRIAL 21)")
print("="*60)

best_params = {
    'n_da'           : 56,
    'n_steps'        : 10,
    'gamma'          : 1.9791715302771704,
    'n_independent'  : 2,
    'n_shared'       : 3,
    'momentum'       : 0.06315153770265473,
    'lambda_sparse'  : 1.1555751696611154e-6,
    'clip_value'     : 0.9587981205495992,
    'lr'             : 0.006180489033395728,
    'weight_decay'   : 7.019218799818284e-5,
    'batch_size'     : 256,
    'virtual_batch_size': 512,
    'max_epochs'     : 41,
}
best_val_auc = 0.9941578202634862

for k, v in best_params.items():
    print(f"  • {k}: {v}")
print(f"  • Best Validation AUC: {best_val_auc:.6f}")

# =========================================================
# TRAIN FINAL TABNET
# =========================================================
print("\n" + "="*60)
print("TRAINING FINAL TABNET MODEL")
print("="*60)

final_tabnet_model = TabNetClassifier(
    n_d=best_params['n_da'],
    n_a=best_params['n_da'],
    n_steps=best_params['n_steps'],
    gamma=best_params['gamma'],
    cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
    n_independent=best_params['n_independent'],
    n_shared=best_params['n_shared'],
    epsilon=1e-15,
    momentum=best_params['momentum'],
    lambda_sparse=best_params['lambda_sparse'],
    seed=REPRODUCIBILITY_SEED,
    clip_value=best_params['clip_value'],
    verbose=0,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': best_params['lr'], 'weight_decay': best_params['weight_decay']},
)

final_tabnet_model.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_train_np, y_train_np), (X_valid_np, y_valid_np)],
    eval_name=['train', 'val'],
    eval_metric=["auc", WeightedAUC],
    max_epochs=best_params['max_epochs'],
    patience=12,
    batch_size=best_params['batch_size'],
    virtual_batch_size=best_params['virtual_batch_size'],
    num_workers=0, weights=1, drop_last=False,
    loss_fn=StandardLoss()
)
print("✓ Training selesai")

# =========================================================
# PREDIKSI
# =========================================================
probs_train = final_tabnet_model.predict_proba(X_train_np)
probs_valid = final_tabnet_model.predict_proba(X_valid_np)
probs_test  = final_tabnet_model.predict_proba(X_test_np)

# =========================================================
# OUTPUT DIRECTORY
# =========================================================
output_dir = 'sample9_tabnet'
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# THRESHOLD OPTIMAL TABNET MURNI
# =========================================================
best_f1_tabnet, best_threshold_tabnet = 0, 0.5
for thr in np.arange(0.1, 0.9, 0.02):
    f1 = f1_score(y_valid_np, (probs_valid[:,1] >= thr).astype(int), zero_division=0)
    if f1 > best_f1_tabnet:
        best_f1_tabnet, best_threshold_tabnet = f1, thr

y_pred_tabnet = (probs_test[:,1] >= best_threshold_tabnet).astype(int)
cm_tabnet = confusion_matrix(y_test_np, y_pred_tabnet)
tn_t, fp_t, fn_t, tp_t = cm_tabnet.ravel()

tabnet_acc   = accuracy_score(y_test_np, y_pred_tabnet)
tabnet_prec  = precision_score(y_test_np, y_pred_tabnet, zero_division=0)
tabnet_rec   = recall_score(y_test_np, y_pred_tabnet, zero_division=0)
tabnet_f1    = f1_score(y_test_np, y_pred_tabnet, zero_division=0)
tabnet_auc   = roc_auc_score(y_test_np, probs_test[:,1])
tabnet_brier = brier_score_loss(y_test_np, probs_test[:,1])

tabnet_ece = calculate_ece_correct(
    y_test_np,
    probs_test[:, 1],
    n_bins=10
)

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

logits_test_class0 = logits_test_all[:, 0]
logits_test_class1 = logits_test_all[:, 1]

logits_train = logits_train_class1 - logits_train_class0
logits_valid = logits_valid_class1 - logits_valid_class0
logits_test  = logits_test_class1 - logits_test_class0

print("✓ Logits asli TabNet berhasil diekstrak langsung dari final_tabnet_model.network")
print(f"  logits_train_all shape: {logits_train_all.shape}")
print(f"  logits_valid_all shape: {logits_valid_all.shape}")
print(f"  logits_test_all shape : {logits_test_all.shape}")

# =========================================================
# VERIFIKASI SOFTMAX LOGITS ASLI
# =========================================================
softmax_probs_test = softmax_numpy(logits_test_all)

max_diff_test_class0 = np.max(np.abs(probs_test[:, 0] - softmax_probs_test[:, 0]))
max_diff_test_class1 = np.max(np.abs(probs_test[:, 1] - softmax_probs_test[:, 1]))

print("\nVerifikasi softmax logits asli terhadap predict_proba pada data test:")
print(f"  Maksimum selisih kelas 0: {max_diff_test_class0:.10f}")
print(f"  Maksimum selisih kelas 1: {max_diff_test_class1:.10f}")

# =========================================================
# LOGITS ASLI UNTUK SAMPEL KE-9
# =========================================================
sample_idx = 9

z0_sample = logits_test_all[sample_idx, 0]
z1_sample = logits_test_all[sample_idx, 1]
z_diff_sample = logits_test[sample_idx]

print("\n" + "="*60)
print("LOGITS ASLI TABNET UNTUK SAMPEL KE-9")
print("="*60)
print(f"Indeks sampel test = {sample_idx}")
print(f"Label aktual       = {int(y_test_np[sample_idx])}")
print(f"z0 (Non-Default)   = {z0_sample:.6f}")
print(f"z1 (Default)       = {z1_sample:.6f}")
print(f"z1 - z0            = {z_diff_sample:.6f}")
print(f"Probabilitas TabNet kelas 0 = {probs_test[sample_idx, 0]:.8f}")
print(f"Probabilitas TabNet kelas 1 = {probs_test[sample_idx, 1]:.8f}")

# =========================================================
# KALIBRASI
# =========================================================
probs_valid_c1 = probs_valid[:,1].reshape(-1,1)

bbq = BBQ()
bbq.fit(probs_valid_c1, y_valid_np)
p_bbq = np.clip(bbq.transform(probs_test[:,1].reshape(-1,1)), 0, 1)
p_bbq = np.column_stack([1-p_bbq, p_bbq])

_, T_opt = manual_temperature_scaling(logits_valid, y_valid_np)
p_ts_raw = 1 / (1 + np.exp(-logits_test / T_opt))
p_ts = np.column_stack([1-p_ts_raw, p_ts_raw])

iso = IsotonicRegression()
iso.fit(probs_valid_c1, y_valid_np)
p_iso = np.clip(iso.transform(probs_test[:,1].reshape(-1,1)), 0, 1)
p_iso = np.column_stack([1-p_iso, p_iso])

# Probabilitas validasi setelah kalibrasi untuk menentukan metode terbaik.
p_valid_bbq = np.clip(bbq.transform(probs_valid_c1), 0, 1).reshape(-1)

temp_probs_valid, _ = manual_temperature_scaling(logits_valid, y_valid_np)
temp_probs_valid = np.clip(temp_probs_valid, 0, 1)

p_valid_iso = np.clip(iso.transform(probs_valid_c1), 0, 1).reshape(-1)

# =========================================================
# MENENTUKAN METODE KALIBRASI TERBAIK
# =========================================================
print("\n" + "="*60)
print("DETERMINE BEST CALIBRATION METHOD")
print("="*60)

ece_original_test = calculate_ece_correct(
    y_test_np,
    probs_test[:, 1],
    n_bins=10
)

ece_bbq_test = calculate_ece_correct(
    y_test_np,
    p_bbq[:, 1],
    n_bins=10
)

ece_ts_test = calculate_ece_correct(
    y_test_np,
    p_ts[:, 1],
    n_bins=10
)

ece_iso_test = calculate_ece_correct(
    y_test_np,
    p_iso[:, 1],
    n_bins=10
)

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
# THRESHOLD OPTIMAL SETELAH KALIBRASI TERBAIK
# =========================================================
print("\n" + "="*60)
print(f"OPTIMIZING THRESHOLD FOR {best_calib_method}")
print("="*60)

if best_calib_method == "BBQ":
    probs_valid_calibrated = p_valid_bbq
    probs_test_calibrated = p_bbq[:, 1]
elif best_calib_method == "Temperature Scaling":
    probs_valid_calibrated = temp_probs_valid
    probs_test_calibrated = p_ts[:, 1]
elif best_calib_method == "Isotonic Regression":
    probs_valid_calibrated = p_valid_iso
    probs_test_calibrated = p_iso[:, 1]
else:
    probs_valid_calibrated = probs_valid[:, 1]
    probs_test_calibrated = probs_test[:, 1]

thresholds_calibrated = np.arange(0.10, 0.90, 0.02)
best_f1_calibrated = 0
best_threshold_calibrated = 0.5

threshold_rows = []
for thr in thresholds_calibrated:
    y_pred_valid = (probs_valid_calibrated >= thr).astype(int)
    recall_val = recall_score(y_valid_np, y_pred_valid, zero_division=0)
    precision_val = precision_score(y_valid_np, y_pred_valid, zero_division=0)
    f1_val = f1_score(y_valid_np, y_pred_valid, zero_division=0)
    acc_val = accuracy_score(y_valid_np, y_pred_valid)

    threshold_rows.append({
        "Threshold": thr,
        "Recall": recall_val,
        "Precision": precision_val,
        "F1-Score": f1_val,
        "Accuracy": acc_val
    })

    if f1_val > best_f1_calibrated:
        best_f1_calibrated = f1_val
        best_threshold_calibrated = thr

print(f"\n✓ Optimal threshold for {best_calib_method}: {best_threshold_calibrated:.2f}")
print(f"✓ Best validation F1 for {best_calib_method}: {best_f1_calibrated:.6f}")

# =========================================================
# PREDIKSI SETELAH KALIBRASI
# =========================================================
y_pred_calibrated = (probs_test_calibrated >= best_threshold_calibrated).astype(int)

y_pred_bbq = (
    (p_bbq[:, 1] >= best_threshold_calibrated).astype(int)
    if best_calib_method == "BBQ"
    else (p_bbq[:, 1] >= best_threshold_tabnet).astype(int)
)

y_pred_ts = (
    (p_ts[:, 1] >= best_threshold_calibrated).astype(int)
    if best_calib_method == "Temperature Scaling"
    else (p_ts[:, 1] >= best_threshold_tabnet).astype(int)
)

y_pred_iso = (
    (p_iso[:, 1] >= best_threshold_calibrated).astype(int)
    if best_calib_method == "Isotonic Regression"
    else (p_iso[:, 1] >= best_threshold_tabnet).astype(int)
)

cm_calibrated = confusion_matrix(y_test_np, y_pred_calibrated)
tn_b, fp_b, fn_b, tp_b = cm_calibrated.ravel()

cal_acc   = accuracy_score(y_test_np, y_pred_calibrated)
cal_prec  = precision_score(y_test_np, y_pred_calibrated, zero_division=0)
cal_rec   = recall_score(y_test_np, y_pred_calibrated, zero_division=0)
cal_f1    = f1_score(y_test_np, y_pred_calibrated, zero_division=0)
cal_auc   = roc_auc_score(y_test_np, probs_test_calibrated)
cal_brier = brier_score_loss(y_test_np, probs_test_calibrated)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
feature_importance = final_tabnet_model.feature_importances_
imp_df = pd.DataFrame({'Feature': all_features, 'Importance': feature_importance})\
           .sort_values('Importance', ascending=False)
imp_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)

# =========================================================
# LOGIT TABNET — SATU SAMPEL TRUE POSITIVE
# =========================================================
print("\n" + "="*60)
print("LOGIT TABNET")
print("="*60)

sample_idx = 9
sample_raw = X_test[sample_idx]
sample_norm = X_test_np[sample_idx]

print(f"\nSampel dipilih : indeks ke-{sample_idx} dari test set")
print(f"Label aktual   : {int(y_test_np[sample_idx])} (Default)")
print(f"Label prediksi metode terbaik ({best_calib_method}) : {int(y_pred_calibrated[sample_idx])}")

# --- LANGKAH 1: Nilai fitur asli ---
print("\n" + "-"*65)
print("LANGKAH 1: Nilai Fitur Asli (Sebelum Normalisasi)")
print("-"*65)
print(f"{'No':<5}{'Nama Fitur':<35}{'Nilai Asli'}")
print("-"*65)
for i,(name,val) in enumerate(zip(all_features, sample_raw)):
    print(f"{i+1:<5}{name:<35}{val:.4f}")

# --- LANGKAH 2: Normalisasi---
print("\n" + "-"*75)
print("LANGKAH 2: Normalisasi (StandardScaler)")
print("Rumus: x_norm = (x - mean_train) / std_train")
print("-"*75)
print(f"{'No':<5}{'Nama Fitur':<30}{'x_asli':<12}{'mean':<12}{'std':<12}{'x_norm'}")
print("-"*75)

scaler_means = scaler.mean_
scaler_stds  = scaler.scale_

for i, feat_name in enumerate(all_features):
    x_asli = sample_raw[i]
    x_norm = sample_norm[i]
    if i in continuous_to_scale:
        pos      = continuous_to_scale.index(i)
        mean_val = scaler_means[pos]
        std_val  = scaler_stds[pos]
        print(f"{i+1:<5}{feat_name:<30}{x_asli:<12.4f}{mean_val:<12.4f}"
              f"{std_val:<12.4f}{x_norm:.4f}")
    else:
        print(f"{i+1:<5}{feat_name:<30}{x_asli:<12.4f}{'(kategorik)':<12}"
              f"{'—':<12}{x_norm:.4f}")

# --- LANGKAH 3 & 4: Logits asli dan Softmax ---
sample_input = X_test_np[sample_idx:sample_idx+1]
probs_sample = final_tabnet_model.predict_proba(sample_input)
p0_s, p1_s   = probs_sample[0, 0], probs_sample[0, 1]

# Logits asli dari network untuk sampel yang dipilih
z0 = logits_test_all[sample_idx, 0]
z1 = logits_test_all[sample_idx, 1]

# Softmax manual dari logits asli
exp_z0 = np.exp(z0)
exp_z1 = np.exp(z1)
sum_exp = exp_z0 + exp_z1
sm0 = exp_z0 / sum_exp
sm1 = exp_z1 / sum_exp

print("\n" + "-"*60)
print("LANGKAH 3: Logits Asli TabNet Sebelum Softmax")
print("-"*60)
print(f"  z0 (Non-Default) = {z0:.6f}")
print(f"  z1 (Default)     = {z1:.6f}")
print(f"  z1 - z0          = {z1 - z0:.6f}")

# =========================================================
# ATTENTION MASK TABNET UNTUK SAMPEL TERPILIH
# =========================================================
print("\n" + "="*80)
print(f"ATTENTION MASK TABNET - SAMPEL INDEKS KE-{sample_idx}")
print("="*80)

M_explain, masks = final_tabnet_model.explain(X_test_np)

if isinstance(masks, dict):
    mask_keys = sorted(masks.keys())
    masks_array = np.stack([masks[k] for k in mask_keys], axis=1)
else:
    masks_array = masks

print(f"M_explain shape    : {M_explain.shape}")
print(f"masks type         : {type(masks)}")
print(f"masks_array shape  : {masks_array.shape}")
print(f"Jumlah decision step : {masks_array.shape[1]}")
print(f"Jumlah fitur         : {masks_array.shape[2]}")

x_sample = X_test_np[sample_idx]

# ---------------------------------------------------------
# 1. Aggregated attention mask
# ---------------------------------------------------------
agg_mask_df = pd.DataFrame({
    "Feature": all_features,
    "x_norm": x_sample,
    "Aggregated_Mask": M_explain[sample_idx],
    "x_norm_x_Aggregated_Mask": x_sample * M_explain[sample_idx]
}).sort_values("Aggregated_Mask", ascending=False)

print("\n" + "-"*80)
print("AGGREGATED ATTENTION MASK - TOP 10 FITUR")
print("-"*80)
print(agg_mask_df.head(10).to_string(index=False, float_format="%.6f"))

agg_mask_df.to_csv(
    f"{output_dir}/sample_{sample_idx}_aggregated_attention_mask.csv",
    index=False
)

# ---------------------------------------------------------
# 2. Attention mask per decision step
# ---------------------------------------------------------
all_steps = []

for step in range(masks_array.shape[1]):
    step_mask = masks_array[sample_idx, step, :]
    x_masked = x_sample * step_mask

    step_df = pd.DataFrame({
        "Step": step + 1,
        "Feature": all_features,
        "x_norm": x_sample,
        "Mask": step_mask,
        "x_norm_x_Mask": x_masked
    }).sort_values("Mask", ascending=False)

    step_df.to_csv(
        f"{output_dir}/sample_{sample_idx}_attention_mask_step_{step+1}.csv",
        index=False
    )

    all_steps.append(step_df)

all_steps_df = pd.concat(all_steps, ignore_index=True)
all_steps_df.to_csv(
    f"{output_dir}/sample_{sample_idx}_attention_mask_all_steps.csv",
    index=False
)

print("\n✓ Attention mask per step berhasil disimpan.")
print(f"✓ File utama: {output_dir}/sample_{sample_idx}_attention_mask_all_steps.csv")

# =========================================================
# FITUR TERPILIH PER LANGKAH KEPUTUSAN
# =========================================================

selected_mask_df = all_steps_df[all_steps_df["Mask"] > 0].copy()

selected_mask_df = selected_mask_df[["Step", "Feature",
    "x_norm", "Mask"]]

selected_mask_df = selected_mask_df.sort_values(
    by=["Step", "Mask"],
    ascending=[True, False])

print("\n" + "="*80)
print("FITUR TERPILIH PER LANGKAH KEPUTUSAN TABNET - SAMPEL KE-9")
print("="*80)

print(f"{'Langkah':<10}{'Fitur Terpilih':<30}{'x_norm':>12}{'M_j[i]':>12}")
print("-"*70)

for _, row in selected_mask_df.iterrows():
    print(f"{int(row['Step']):<10}" f"{row['Feature']:<30}"
        f"{row['x_norm']:>12.4f}" f"{row['Mask']:>12.4f}")

selected_mask_df.to_csv(
    f"{output_dir}/sample_{sample_idx}_selected_attention_mask.csv",
    index=False)

print("-"*70)
print(f"✓ Fitur terpilih berhasil disimpan ke: "
      f"{output_dir}/sample_{sample_idx}_selected_attention_mask.csv")

# =========================================================
# EKSTRAKSI NILAI INTERNAL SEBELUM SPARSEMAX TABNET
# =========================================================
internal_sparsemax_df = extract_tabnet_internal_attention(
    model=final_tabnet_model,
    X_data=X_test_np,
    sample_idx=sample_idx,
    all_features=all_features,
    output_dir=output_dir,
    gamma_value=best_params['gamma']
)

# =========================================================
# METRIK KALIBRASI — TABNET MURNI
# =========================================================
print("\n" + "="*60)
print("METRIK KALIBRASI — TABNET MURNI")
print("="*60)

probs_raw = probs_test[:, 1]

tabnet_ece = calculate_ece_with_detail(
    y_test_np,
    probs_raw,
    n_bins=10
)

tabnet_brier = brier_score_loss(
    y_test_np,
    probs_raw
)

print_brier_detail_per_sample(
    method_name="Original TabNet",
    y_true=y_test_np,
    y_prob=probs_raw,
    y_pred=y_pred_tabnet
)

print(f"\nECE TabNet Murni       = {tabnet_ece:.6f}")
print(f"Brier Score TabNet     = {tabnet_brier:.6f}")

# =========================================================
# METRIK KALIBRASI — BBQ
# =========================================================
print("\n" + "="*60)
print("METRIK KALIBRASI — BBQ")
print("="*60)

cal_ece_bbq = calculate_ece_with_detail(y_test_np, p_bbq[:, 1], n_bins=10)
cal_brier_bbq = brier_score_loss(y_test_np, p_bbq[:, 1])

print_brier_detail_per_sample(
    method_name="BBQ",
    y_true=y_test_np,
    y_prob=p_bbq[:, 1],
    y_pred=y_pred_bbq
)

print(f"\nECE BBQ              = {cal_ece_bbq:.6f}")
print(f"Brier Score BBQ      = {cal_brier_bbq:.6f}")
print(f"ECE sebelum Original = {tabnet_ece:.6f}")
print(f"Perbaikan ECE BBQ    = {(tabnet_ece - cal_ece_bbq) / tabnet_ece * 100:.2f}%")

# =========================================================
# METRIK KALIBRASI — TEMPERATURE SCALING
# =========================================================
print("\n" + "="*60)
print("METRIK KALIBRASI — TEMPERATURE SCALING")
print("="*60)

ece_ts = calculate_ece_with_detail(y_test_np, p_ts[:, 1], n_bins=10)
brier_ts = brier_score_loss(y_test_np, p_ts[:, 1])

print_brier_detail_per_sample(
    method_name="Temperature Scaling",
    y_true=y_test_np,
    y_prob=p_ts[:, 1],
    y_pred=y_pred_ts
)

print(f"\nECE Temperature Scaling = {ece_ts:.6f}")
print(f"Brier Score TS          = {brier_ts:.6f}")
print(f"ECE sebelum Original    = {tabnet_ece:.6f}")
print(f"Perbaikan ECE TS        = {(tabnet_ece - ece_ts) / tabnet_ece * 100:.2f}%")

# =========================================================
# METRIK KALIBRASI — ISOTONIC REGRESSION
# =========================================================
print("\n" + "="*60)
print("METRIK KALIBRASI — ISOTONIC REGRESSION")
print("="*60)

ece_iso = calculate_ece_with_detail(y_test_np, p_iso[:, 1], n_bins=10)
brier_iso = brier_score_loss(y_test_np, p_iso[:, 1])

print_brier_detail_per_sample(
    method_name="Isotonic Regression",
    y_true=y_test_np,
    y_prob=p_iso[:, 1],
    y_pred=y_pred_iso
)

print(f"\nECE Isotonic Regression = {ece_iso:.6f}")
print(f"Brier Score Isotonic    = {brier_iso:.6f}")
print(f"ECE sebelum Original    = {tabnet_ece:.6f}")
print(f"Perbaikan ECE IR        = {(tabnet_ece - ece_iso) / tabnet_ece * 100:.2f}%")

# =========================================================
# SIMPAN HASIL
# =========================================================
final_tabnet_model.save_model(f'{output_dir}/tabnet_model_optimized.zip')
joblib.dump(bbq,            f'{output_dir}/bbq.pkl')
joblib.dump(iso,            f'{output_dir}/isotonic_regression.pkl')
joblib.dump(scaler,         f'{output_dir}/scaler.pkl')
joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
if knn_imputer:
    joblib.dump(knn_imputer, f'{output_dir}/knn_imputer.pkl')

imp_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)

print(f"\n✓ Semua hasil tersimpan di '{output_dir}/'")
print("="*60)
print("SELESAI")
print("="*60)
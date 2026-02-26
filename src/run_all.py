"""
Notes:
- This script loads a lenient ARFF , cleans it, and trains:
  (1) Logistic Regression baseline
  (2) Random Forest final model
- It generates:
  outputs/figures/*.png
  outputs/tables/*.csv
  outputs/models/*.joblib
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


RANDOM_STATE = 42


def ensure_dirs(repo_root: Path) -> None:
    (repo_root / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "models").mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)



def _normalize_cell(raw: str) -> str:

    if raw is None:
        return "?"
    s = str(raw)
    s = s.replace("\ufeff", "").replace("\xa0", " ")
    s = s.strip()
    if s == "":
        return "?"
    return s


def _sanitize_value(x):

    if pd.isna(x):
        return np.nan
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        x = x.replace("\t", " ").strip()
        if x == "?":
            return np.nan
        x = re.sub(r"\s+", " ", x)
    return x


def _clean_label(x):
 
    if pd.isna(x):
        return np.nan
    s = str(x).strip().strip("'\"").lower()
    s = re.sub(r"\s+", "", s)
    if s in {"?", "", "nan", "none", "null"}:
        return np.nan
    return s


def load_arff_lenient(arff_path: Path) -> pd.DataFrame:

    attributes = []
    rows = []
    in_data = False

    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            low = line.lower()

            # Header section
            if not in_data:
                if low.startswith("@attribute"):
                    parts = line.split(None, 2)
                    if len(parts) >= 2:
                        name = parts[1].strip().strip("'\"")
                        attributes.append(name)
                elif low.startswith("@data"):
                    in_data = True
                continue

            # Data section
            if line.startswith("%"):
                continue

            tokens = [_normalize_cell(tok) for tok in line.split(",")]

            # Pad/truncate to match header length (safety)
            if len(tokens) < len(attributes):
                tokens = tokens + (["?"] * (len(attributes) - len(tokens)))
            if len(tokens) > len(attributes):
                tokens = tokens[:len(attributes)]

            rows.append(tokens)

    df = pd.DataFrame(rows, columns=attributes)

    # normalize column names
    df.columns = [str(c).strip().strip("'\"").lower() for c in df.columns]

    # sanitize cells
    for c in df.columns:
        df[c] = df[c].map(_sanitize_value)

    return df


def load_ckd_dataset(arff_path: Path) -> pd.DataFrame:

    df = load_arff_lenient(arff_path)

    # 1) find target column
    if "class" not in df.columns:
        candidate_labels = {"ckd", "notckd", "yes", "no"}
        best_col = None
        best_score = -1.0
        for col in df.columns:
            vals = df[col].dropna().map(_clean_label).dropna()
            if len(vals) == 0:
                continue
            score = float(vals.isin(candidate_labels).mean())
            if score > best_score:
                best_score = score
                best_col = col
        if best_col is None:
            raise ValueError("Could not identify target column.")
        df = df.rename(columns={best_col: "class"})

    # 2) map labels → target (prefer ckd/notckd if present)
    labels = df["class"].map(_clean_label)
    counts = labels.value_counts(dropna=True).to_dict()

    ckd_share = (counts.get("ckd", 0) + counts.get("notckd", 0)) / max(1, len(labels.dropna()))
    yn_share = (counts.get("yes", 0) + counts.get("no", 0)) / max(1, len(labels.dropna()))

    if ckd_share >= 0.90:
        mapping = {"ckd": 1, "notckd": 0}
    elif yn_share >= 0.90:
        mapping = {"yes": 1, "no": 0}
    else:
        mapping = {"ckd": 1, "notckd": 0, "yes": 1, "no": 0}

    labels = labels.where(labels.isin(mapping.keys()), np.nan)
    df["target"] = labels.map(mapping)

    # 3) drop invalid/missing labels
    before = len(df)
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    after = len(df)

    print(f"[INFO] Target label counts (cleaned): {df['target'].value_counts().to_dict()}")
    print(f"[INFO] Dropped {before - after} rows due to invalid/missing class labels.")

    # 4) enforce known categorical domains
    domain_rules = {
        "rbc": {"normal", "abnormal"},
        "pc": {"normal", "abnormal"},
        "pcc": {"present", "notpresent"},
        "ba": {"present", "notpresent"},
        "htn": {"yes", "no"},
        "dm": {"yes", "no"},
        "cad": {"yes", "no"},
        "appet": {"good", "poor"},
        "pe": {"yes", "no"},
        "ane": {"yes", "no"},
    }
    for col, allowed in domain_rules.items():
        if col in df.columns:
            s = df[col].map(_clean_label)
            s = s.where(s.isna() | s.isin(allowed), np.nan)
            df[col] = s

    return df


# -------------------------
# Column inference
# -------------------------
def infer_columns(df: pd.DataFrame):

    # NOTE: include both rc and rbcc if present
    numeric_guess = [
        "age", "bp", "bgr", "bu", "sc", "sod", "pot",
        "hemo", "pcv", "wc", "rc", "rbcc",
    ]

    exclude = {"class", "target"}
    feature_cols = [c for c in df.columns if c not in exclude]

    numeric_cols = [c for c in numeric_guess if c in df.columns]
    numeric_set = set(numeric_cols)
    cat_cols = [c for c in feature_cols if c not in numeric_set]

    # Convert numeric columns safely
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return feature_cols, numeric_cols, cat_cols


# -------------------------
# EDA plots
# -------------------------
def plot_class_balance(y: pd.Series, out_path: Path) -> None:
    counts = y.value_counts().sort_index()
    labels = ["notckd (0)", "ckd (1)"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

"""
def plot_missingness(X: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    miss_pct = (X.isna().mean() * 100.0).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Missing Values by Feature (%)")
    
    plt.close(fig)

    out = miss_pct.reset_index()
    out.columns = ["feature", "missing_pct"]
    return out
"""


def plot_missingness(X: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    miss_pct = (X.isna().mean() * 100.0).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(miss_pct.index.astype(str), miss_pct.values)
    ax.set_title("Missing Values by Feature (%)")
    ax.set_ylabel("Missing (%)")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    out = miss_pct.reset_index()
    out.columns = ["feature", "missing_pct"]
    return out


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list, out_path: Path) -> None:
    if len(numeric_cols) == 0:
        return

    cols = 4
    rows = int(np.ceil(len(numeric_cols) / cols))
    fig = plt.figure(figsize=(cols * 3.2, rows * 2.8))

    for i, c in enumerate(numeric_cols, start=1):
        ax = plt.subplot(rows, cols, i)
        vals = df[c].dropna()
        ax.hist(vals, bins=20)
        ax.set_title(c)

    plt.suptitle("Numeric Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmap(df: pd.DataFrame, numeric_cols: list, out_path: Path) -> None:
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=75, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# -------------------------
# Pipelines
# -------------------------
def build_models(numeric_cols, cat_cols):
    """
    Build two pipelines:
    - Logistic Regression: numeric impute + scale, categorical one-hot
    - Random Forest: numeric impute, categorical ordinal encoding (SHAP-friendly)
    """

    # Logistic Regression preprocessing
    num_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre_lr = ColumnTransformer([
        ("num", num_lr, numeric_cols),
        ("cat", cat_lr, cat_cols),
    ])

    lr = Pipeline([
        ("preprocess", pre_lr),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    # Random Forest preprocessing
    num_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    pre_rf = ColumnTransformer([
        ("num", num_rf, numeric_cols),
        ("cat", cat_rf, cat_cols),
    ])

    rf = Pipeline([
        ("preprocess", pre_rf),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return lr, rf


# -------------------------
# Evaluation + plots
# -------------------------
def crossval_metrics(model, X_train, y_train) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "acc": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "ap": "average_precision",
    }

    cv = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)

    out = {}
    for k in scoring.keys():
        out[f"cv_{k}_mean"] = float(np.mean(cv[f"test_{k}"]))
        out[f"cv_{k}_std"] = float(np.std(cv[f"test_{k}"]))
    return out


def test_metrics(y_true, y_pred, proba, thr: float, name: str) -> dict:
    return {
        "model": name,
        "threshold": thr,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "avg_precision": float(average_precision_score(y_true, proba)),
    }


def make_eval_figures(
    y_test: pd.Series,
    lr_proba: np.ndarray,
    rf_proba: np.ndarray,
    lr_pred: np.ndarray,
    rf_pred: np.ndarray,
    out_fig_dir: Path,
    thr: float = 0.5,
):
   

    # Final model
    final_name = "RandomForest"
    final_pred = rf_pred
    final_proba = rf_proba

    # Confusion matrix
    cm = confusion_matrix(y_test, final_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["notckd", "ckd"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d", colorbar=True)
    ax.set_title(f"Confusion Matrix (Test) — {final_name} @ thr={thr}")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig_confusion_matrix.png", dpi=220)
    plt.close(fig)

    # ROC curve 
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, proba in [("LogReg", lr_proba), ("RandomForest", rf_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.step(fpr, tpr, where="post", label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")

"""
    if roc_auc_score(y_test, rf_proba) >= 0.98:
        ax.set_xlim(0,)
        ax.set_ylim(0.8, 1.01)

    fig.tight_layout()
    fig.savefig()
    plt.close(fig)
"""
   
    if roc_auc_score(y_test, rf_proba) >= 0.98:
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0.8, 1.01)

    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig_roc_curve.png", dpi=220)
    plt.close(fig)

    # PR curve (step + baseline prevalence)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    base_rate = float(np.mean(y_test))
    ax.hlines(base_rate, 0, 1, linestyles="--", label=f"Baseline (pos rate={base_rate:.3f})")

    for name, proba in [("LogReg", lr_proba), ("RandomForest", rf_proba)]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.step(rec, prec, where="post", label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve (Test Set)")
    ax.legend(loc="lower left")
    ax.set_ylim(max(0.0, base_rate - 0.05), 1.01)

    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig_pr_curve.png", dpi=220)
    plt.close(fig)

    # Calibration curve + bin counts
    n_bins = 5
    frac_pos, mean_pred = calibration_curve(y_test, final_proba, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o", label=final_name)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve (Test Set) — {final_name} (bins={n_bins})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig_calibration_curve.png", dpi=220)
    plt.close(fig)

    # Bin count plot 
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(final_proba, bins, right=True)
    bin_counts = np.array([(bin_ids == i).sum() for i in range(1, n_bins + 1)], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 2.8))
    centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(centers, bin_counts, width=(bins[1] - bins[0]) * 0.9)
    ax.set_xlabel("Predicted probability bin (uniform)")
    ax.set_ylabel("Count")
    ax.set_title(f"Calibration Bin Counts — {final_name} (n={len(y_test)})")
    fig.tight_layout()
    fig.savefig(out_fig_dir / "fig_calibration_bin_counts.png", dpi=220)
    plt.close(fig)


# -------------------------
# SHAP (Random Forest)
# -------------------------
def xai_shap_for_rf(rf_pipeline, X_train, X_test, out_fig_dir: Path, out_table_dir: Path):
    if not SHAP_AVAILABLE:
        print("SHAP not installed. Skipping SHAP figures.")
        return

    pre = rf_pipeline.named_steps["preprocess"]
    model = rf_pipeline.named_steps["model"]

    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)

    # feature names: numeric then cat (ColumnTransformer order)
    num_cols = list(pre.transformers_[0][2])
    cat_cols = list(pre.transformers_[1][2])
    feature_names = num_cols + cat_cols

    X_test_t_df = pd.DataFrame(X_test_t, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_t_df)
    base_value = explainer.expected_value

    """
    # Handle SHAP output shape for binary classification robustly
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            base_value = base_value[1]
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            base_value = base_value[1]
    """

    # Handle SHAP output shape for binary classification robustly
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            base_value = base_value[1]
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            base_value = base_value[1]

    sv = np.asarray(sv)
    if sv.ndim != 2:
        sv = np.squeeze(sv)
    if sv.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

    # Global summary plot
    plt.figure()
    shap.summary_plot(sv, X_test_t_df, show=False)
    plt.title("SHAP Summary (RandomForest, Test Set)")
    plt.tight_layout()
    plt.savefig(out_fig_dir / "fig_shap_summary.png", dpi=220, bbox_inches="tight")
    plt.close()

    # Top features CSV
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = pd.DataFrame({
        "feature": np.array(feature_names)[order],
        "mean_abs_shap": mean_abs[order],
    }).head(15)
    save_csv(top, out_table_dir / "table_shap_top_features.csv")

    # Dependence plots (top 2)
    top2 = top["feature"].tolist()[:2]
    for feat in top2:
        plt.figure()
        shap.dependence_plot(feat, sv, X_test_t_df, show=False)
        plt.title(f"SHAP Dependence: {feat}")
        plt.tight_layout()
        plt.savefig(out_fig_dir / f"fig_shap_dependence_{feat}.png", dpi=220, bbox_inches="tight")
        plt.close()

    # Local waterfall: one high-risk (max proba) and one low-risk (min proba)
    # This avoids claiming "error" when RF is perfect.
    proba = rf_pipeline.predict_proba(X_test)[:, 1]
    i_high = int(np.argmax(proba))
    i_low = int(np.argmin(proba))

    def save_waterfall(i: int, fname: str):
        exp = shap.Explanation(
            values=sv[i],
            base_values=base_value,
            data=X_test_t_df.iloc[i].values,
            feature_names=feature_names,
        )
        plt.figure()
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        plt.savefig(out_fig_dir / fname, dpi=220, bbox_inches="tight")
        plt.close()

    save_waterfall(i_high, "fig_shap_waterfall_tp.png")
    # Keep old filename for compatibility with your Overleaf
    save_waterfall(i_low, "fig_shap_waterfall_error.png")


# -------------------------
# Subgroup table (CSV)
# -------------------------
def subgroup_metrics(X_test: pd.DataFrame, y_test: pd.Series, proba: np.ndarray, out_csv: Path):
    """
    Make subgroup table for ethics section.
    (plotting recall is not useful when everything is 1.0.)
    """
    tmp = X_test.copy()
    tmp["y_true"] = y_test.values
    tmp["proba"] = proba

    rows = []

    # Age quartiles
    if "age" in tmp.columns:
        age_num = pd.to_numeric(tmp["age"], errors="coerce")
        qbins = pd.qcut(age_num, q=4, duplicates="drop")
        for grp, sub in tmp.groupby(qbins):
            if len(sub) < 10:
                continue
            yhat = (sub["proba"] >= 0.5).astype(int)
            rows.append({
                "group_type": "age_quartile",
                "group": str(grp),
                "n": int(len(sub)),
                "recall": float(recall_score(sub["y_true"], yhat, zero_division=0)),
                "precision": float(precision_score(sub["y_true"], yhat, zero_division=0)),
                "f1": float(f1_score(sub["y_true"], yhat, zero_division=0)),
            })

    # Binary groups
    for col in ["dm", "htn", "ane"]:
        if col not in tmp.columns:
            continue
        col_vals = tmp[col].astype(str).str.strip().str.lower()
        for val, sub in tmp.groupby(col_vals):
            if len(sub) < 10:
                continue
            yhat = (sub["proba"] >= 0.5).astype(int)
            rows.append({
                "group_type": col,
                "group": str(val),
                "n": int(len(sub)),
                "recall": float(recall_score(sub["y_true"], yhat, zero_division=0)),
                "precision": float(precision_score(sub["y_true"], yhat, zero_division=0)),
                "f1": float(f1_score(sub["y_true"], yhat, zero_division=0)),
            })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        save_csv(out, out_csv)


# -------------------------
# Ablation: sg/al/su encoding
# -------------------------
def run_encoding_ablation(df: pd.DataFrame, feature_cols: list, numeric_cols_base: list, cat_cols_base: list, out_table_dir: Path):
    """
    Ablation: sg/al/su as categorical vs ordinal numeric.
    Writes outputs/tables/table_ablation_encoding.csv
    """
    X = df[feature_cols].copy()
    y = df["target"].copy()

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "acc": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "ap": "average_precision",
    }

    def cv_row(variant, model_name, pipeline):
        cv = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        row = {"variant": variant, "model": model_name}
        for k in scoring.keys():
            row[f"{k}_mean"] = float(cv[f"test_{k}"].mean())
            row[f"{k}_std"] = float(cv[f"test_{k}"].std())
        return row

    rows = []

    # Baseline: categorical
    lr_base, rf_base = build_models(numeric_cols_base, cat_cols_base)
    rows.append(cv_row("categorical (baseline)", "LogReg", lr_base))
    rows.append(cv_row("categorical (baseline)", "RandomForest", rf_base))

    # Variant: ordinal numeric for sg/al/su
    df2 = df.copy()
    for col in ["sg", "al", "su"]:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

    ordinal_cols = [c for c in ["sg", "al", "su"] if c in df2.columns]
    numeric_cols_ord = list(dict.fromkeys(list(numeric_cols_base) + ordinal_cols))
    cat_cols_ord = [c for c in cat_cols_base if c not in set(ordinal_cols)]

    X2 = df2[feature_cols].copy()
    X_train2 = X2.loc[X_train.index].copy()

    lr_ord, rf_ord = build_models(numeric_cols_ord, cat_cols_ord)
    rows.append(cv_row("ordinal numeric (sg/al/su)", "LogReg", lr_ord))
    rows.append(cv_row("ordinal numeric (sg/al/su)", "RandomForest", rf_ord))

    out = pd.DataFrame(rows)
    save_csv(out, out_table_dir / "table_ablation_encoding.csv")

    print(" Ablation saved:")
    print(f" - {out_table_dir / 'table_ablation_encoding.csv'}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/chronic_kidney_disease_full.arff",
        help="Path to ARFF file (relative to repo root)."
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ensure_dirs(repo_root)

    fig_dir = repo_root / "outputs" / "figures"
    tab_dir = repo_root / "outputs" / "tables"
    model_dir = repo_root / "outputs" / "models"

    arff_path = repo_root / args.data
    if not arff_path.exists():
        raise FileNotFoundError(f"ARFF not found: {arff_path}")

    # 1) Load + clean
    df = load_ckd_dataset(arff_path)

    # 2) Decide columns
    feature_cols, numeric_cols, cat_cols = infer_columns(df)

    # 3) Ablation 
    run_encoding_ablation(df, feature_cols, numeric_cols, cat_cols, tab_dir)

    # 4) Dataset overview CSV 
    overview = pd.DataFrame([{
        "instances": int(len(df)),
        "features": int(len(feature_cols)),
        "numeric_features": int(len(numeric_cols)),
        "categorical_features": int(len(cat_cols)),
        "ckd_count": int(df["target"].sum()),
        "notckd_count": int((df["target"] == 0).sum()),
        "missing_any_pct": float(df[feature_cols].isna().any(axis=1).mean() * 100.0),
    }])
    save_csv(overview, tab_dir / "table_dataset_overview.csv")

    # 5) EDA figures
    X = df[feature_cols].copy()
    y = df["target"].copy()

    plot_class_balance(y, fig_dir / "fig_class_balance.png")

    miss_table = plot_missingness(X, fig_dir / "fig_missingness.png")
    save_csv(miss_table, tab_dir / "table_missingness.csv")

    plot_numeric_distributions(df, numeric_cols, fig_dir / "fig_numeric_distributions.png")
    plot_corr_heatmap(df, numeric_cols, fig_dir / "fig_corr_heatmap.png")

    # 6) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 7) Build pipelines
    lr, rf = build_models(numeric_cols, cat_cols)

    # 8) CV metrics table
    rows = []
    for name, model in [("LogReg", lr), ("RandomForest", rf)]:
        row = {"model": name}
        row.update(crossval_metrics(model, X_train, y_train))
        rows.append(row)

    cv_df = pd.DataFrame(rows)
    save_csv(cv_df, tab_dir / "table_cv_metrics.csv")

    # 9) Fit models + save
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    joblib.dump(lr, model_dir / "model_logreg.joblib")
    joblib.dump(rf, model_dir / "model_random_forest.joblib")

    # 10) Test metrics
    thr = 0.5
    lr_proba = lr.predict_proba(X_test)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]

    lr_pred = (lr_proba >= thr).astype(int)
    rf_pred = (rf_proba >= thr).astype(int)

    test_df = pd.DataFrame([
        test_metrics(y_test, lr_pred, lr_proba, thr, "LogReg"),
        test_metrics(y_test, rf_pred, rf_proba, thr, "RandomForest"),
    ])
    save_csv(test_df, tab_dir / "table_test_metrics.csv")

    # 11) Eval figures (ROC/PR/Calib/CM)
    make_eval_figures(
        y_test=y_test,
        lr_proba=lr_proba,
        rf_proba=rf_proba,
        lr_pred=lr_pred,
        rf_pred=rf_pred,
        out_fig_dir=fig_dir,
        thr=thr,
    )

    # 12) SHAP 
    xai_shap_for_rf(rf, X_train, X_test, fig_dir, tab_dir)

    # 13) Subgroup table 
    subgroup_metrics(
        X_test=X_test,
        y_test=y_test,
        proba=rf_proba,
        out_csv=tab_dir / "table_subgroup_metrics.csv",
    )

    print("\n Done. Generated outputs:")
    print(f"Figures: {fig_dir}")
    print(f"Tables : {tab_dir}")
    print(f"Models : {model_dir}")
    if not SHAP_AVAILABLE:
        print("Note: SHAP not installed → SHAP figures skipped (pip install shap).")


if __name__ == "__main__":
    main()
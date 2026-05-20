"""
Entrenamiento modelo LightGBM v2 — clasificación binaria is_disrupted.

Mejoras respecto a v1:
  - seller_tasa_disrupcion      : tasa histórica de disrupción del seller
  - seller_n_ordenes            : volumen histórico del seller
  - categoria_tasa_disrupcion   : tasa histórica por service_category
  - categoria_cac_tasa_disrupcion: tasa por categoría + click&collect
  - dia_semana_tasa_disrupcion  : tasa histórica por día de la semana
  - franja_horaria              : mañana / tarde / noche
  - franja_tasa_disrupcion      : tasa histórica por franja horaria

Dataset: staging_marts.ml_dataset_v2
Split:   temporal (train=2025-01/11, val=2025-12/2026-01, test=2026-02)

Uso:
    python train_model_v2.py

Outputs en ml_outputs/:
    model_lgbm_v2.pkl
    results_v2.json
    feature_importance_v2.csv
    comparacion_v1_v2.json
"""

import os, gc, json, time, pickle, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Features ────────────────────────────────────────────────────────────────

FEATURES_NUMERICAS = [
    # originales v1
    "is_click_and_collect",
    "is_high_season",
    "has_insurance",
    "total_items",
    "distinct_skus",
    "dia_semana_creacion",
    "hora_creacion",
    "dia_mes_creacion",
    "sla_horas_prometidas",
    # nuevas v2 — features históricas
    "seller_n_ordenes",
    "seller_tasa_disrupcion",
    "categoria_tasa_disrupcion",
    "categoria_cac_tasa_disrupcion",
    "dia_semana_tasa_disrupcion",
    "franja_tasa_disrupcion",
]

FEATURES_CATEGORICAS = [
    "service_category",
    "seller_id",
    "created_by",
    "franja_horaria",
]

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
TARGET       = "target_binario"
CHUNK_SIZE   = 500_000

COLS_A_LEER = ALL_FEATURES + [TARGET, "split_set", "logistic_order_id"]


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def cargar_split(engine, split_name: str) -> pd.DataFrame:
    print(f"  Cargando '{split_name}'...", flush=True)
    t0 = time.time()
    cols_sql = ", ".join(COLS_A_LEER)
    query = f"""
        SELECT {cols_sql}
        FROM staging_marts.ml_dataset_v2
        WHERE split_set = '{split_name}'
    """
    chunks, total = [], 0
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(text(query))
        while True:
            rows = result.fetchmany(CHUNK_SIZE)
            if not rows:
                break
            chunks.append(pd.DataFrame(rows, columns=COLS_A_LEER))
            total += len(chunks[-1])
            print(f"    ...{total:,} filas", flush=True)
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    print(f"  '{split_name}' listo: {len(df):,} filas en {time.time()-t0:.1f}s", flush=True)
    return df


def preprocesar(df, encoders=None, fit=False):
    df = df.copy()

    # Numéricas: nulos a -1
    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")

    # Categóricas: label encoding
    if encoders is None:
        encoders = {}
    for col in FEATURES_CATEGORICAS:
        df[col] = df[col].fillna("DESCONOCIDO").astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]).astype("int32")
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "DESCONOCIDO")
            df[col] = le.transform(df[col]).astype("int32")

    X = df[ALL_FEATURES]
    y = df[TARGET].astype("int8")
    return X, y, encoders


def entrenar(X_train, y_train, X_val, y_val):
    print("\nEntrenando LightGBM v2...", flush=True)
    params = {
        "objective":         "binary",
        "metric":            ["binary_logloss", "auc"],
        "boosting_type":     "gbdt",
        "n_estimators":      1000,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "max_depth":         -1,
        "min_child_samples": 200,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l1":         0.1,
        "lambda_l2":         0.1,
        "scale_pos_weight":  1.0,
        "n_jobs":            -1,
        "verbose":           -1,
        "random_state":      42,
    }
    cat_idx = [ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS]
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
        categorical_feature=cat_idx,
    )
    print(f"\nMejor iteración: {model.best_iteration_}", flush=True)
    return model


def evaluar(model, X, y, split_name):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, y_prob)
    auc_pr  = average_precision_score(y, y_prob)
    report  = classification_report(y, y_pred, output_dict=True)
    cm      = confusion_matrix(y, y_pred).tolist()

    # % dentro de distintos umbrales de probabilidad
    high_risk  = (y_prob >= 0.7).mean() * 100
    very_high  = (y_prob >= 0.9).mean() * 100

    metricas = {
        "split":                split_name,
        "n":                    int(len(y)),
        "auc_roc":              round(float(auc_roc), 4),
        "auc_pr":               round(float(auc_pr),  4),
        "accuracy":             round(float(report["accuracy"]), 4),
        "precision_disrupted":  round(float(report["1"]["precision"]), 4),
        "recall_disrupted":     round(float(report["1"]["recall"]),    4),
        "f1_disrupted":         round(float(report["1"]["f1-score"]),  4),
        "precision_ok":         round(float(report["0"]["precision"]), 4),
        "recall_ok":            round(float(report["0"]["recall"]),    4),
        "f1_ok":                round(float(report["0"]["f1-score"]),  4),
        "confusion_matrix":     cm,
        "pct_high_risk_70":     round(float(high_risk), 2),
        "pct_very_high_risk_90":round(float(very_high), 2),
    }

    print(f"\n{'='*55}", flush=True)
    print(f"MÉTRICAS — {split_name.upper()}", flush=True)
    print(f"{'='*55}", flush=True)
    print(f"  AUC-ROC:           {auc_roc:.4f}", flush=True)
    print(f"  AUC-PR:            {auc_pr:.4f}", flush=True)
    print(f"  Accuracy:          {report['accuracy']:.4f}", flush=True)
    print(f"  Precision disrupt: {report['1']['precision']:.4f}", flush=True)
    print(f"  Recall    disrupt: {report['1']['recall']:.4f}", flush=True)
    print(f"  F1        disrupt: {report['1']['f1-score']:.4f}", flush=True)
    print(f"  Confusion matrix:", flush=True)
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}", flush=True)
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}", flush=True)
    print(f"  % órdenes prob>=0.7: {high_risk:.1f}%", flush=True)
    print(f"  % órdenes prob>=0.9: {very_high:.1f}%", flush=True)
    return metricas


def guardar_fi(model, path):
    fi = pd.DataFrame({
        "feature":          ALL_FEATURES,
        "importance_gain":  model.booster_.feature_importance("gain"),
        "importance_split": model.booster_.feature_importance("split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(path, index=False)
    print(f"\nFeature importance guardada en: {path}", flush=True)
    print(fi.to_string(index=False), flush=True)
    return fi


def comparar_con_v1(m_val, m_test):
    """Carga resultados del v1 y compara."""
    v1_path = OUTPUT_DIR / "results_v1.json"
    if not v1_path.exists():
        print("\n  results_v1.json no encontrado, saltando comparación.", flush=True)
        return

    with open(v1_path) as f:
        v1 = json.load(f)

    print(f"\n{'='*55}", flush=True)
    print("COMPARACIÓN v1 vs v2", flush=True)
    print(f"{'='*55}", flush=True)
    header = f"{'Métrica':<25} {'v1 val':>10} {'v2 val':>10} {'Δ val':>10}"
    print(header, flush=True)
    print("-"*55, flush=True)

    metricas_cmp = [
        ("auc_roc",             "AUC-ROC"),
        ("accuracy",            "Accuracy"),
        ("f1_disrupted",        "F1 disrupted"),
        ("recall_disrupted",    "Recall disrupted"),
        ("precision_disrupted", "Precision disrupt"),
    ]
    comp = {}
    for key, label in metricas_cmp:
        v1_val = v1["val"].get(key, 0)
        v2_val = m_val.get(key, 0)
        delta  = v2_val - v1_val
        sign   = "+" if delta >= 0 else ""
        print(f"  {label:<23} {v1_val:>10.4f} {v2_val:>10.4f} {sign}{delta:>9.4f}", flush=True)
        comp[key] = {"v1_val": v1_val, "v2_val": v2_val, "delta": round(delta, 4)}

    print(f"\n  Test:", flush=True)
    for key, label in [("auc_roc", "AUC-ROC"), ("f1_disrupted", "F1 disrupted")]:
        v1_t = v1["test"].get(key, 0)
        v2_t = m_test.get(key, 0)
        delta = v2_t - v1_t
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<23} v1={v1_t:.4f}  v2={v2_t:.4f}  Δ={sign}{delta:.4f}", flush=True)

    comp_path = OUTPUT_DIR / "comparacion_v1_v2.json"
    with open(comp_path, "w") as f:
        json.dump(comp, f, indent=2)
    print(f"\n  Comparación guardada: {comp_path}", flush=True)


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("ENTRENAMIENTO LGBM v2 — CLASIFICACIÓN DISRUPCIONES", flush=True)
    print(f"Features: {len(ALL_FEATURES)} ({len(FEATURES_NUMERICAS)} num, {len(FEATURES_CATEGORICAS)} cat)", flush=True)
    print(f"Nuevas vs v1: seller_tasa_disrupcion, categoria_tasa_disrupcion,", flush=True)
    print(f"              dia_semana_tasa_disrupcion, franja_horaria, franja_tasa_disrupcion,", flush=True)
    print(f"              seller_n_ordenes, categoria_cac_tasa_disrupcion", flush=True)
    print("="*60, flush=True)

    engine = get_engine()

    # ── 1. Cargar ────────────────────────────────────────────────
    print("\n[1/5] Cargando datos desde ml_dataset_v2...", flush=True)
    df_train = cargar_split(engine, "train")
    df_val   = cargar_split(engine, "val")
    df_test  = cargar_split(engine, "test")
    print(f"\n  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}", flush=True)

    # ── 2. Preprocesar ───────────────────────────────────────────
    print("\n[2/5] Preprocesando...", flush=True)
    X_train, y_train, encoders = preprocesar(df_train, fit=True)
    X_val,   y_val,   _        = preprocesar(df_val,   encoders=encoders)
    X_test,  y_test,  _        = preprocesar(df_test,  encoders=encoders)

    del df_train, df_val, df_test; gc.collect()
    print(f"  Train disrupted: {y_train.sum():,} / {len(y_train):,} ({100*y_train.mean():.1f}%)", flush=True)
    print(f"  Val   disrupted: {y_val.sum():,} / {len(y_val):,} ({100*y_val.mean():.1f}%)", flush=True)
    print(f"  Test  disrupted: {y_test.sum():,} / {len(y_test):,} ({100*y_test.mean():.1f}%)", flush=True)

    # ── 3. Entrenar ──────────────────────────────────────────────
    print("\n[3/5] Entrenando...", flush=True)
    t_train = time.time()
    model = entrenar(X_train, y_train, X_val, y_val)
    print(f"  Entrenamiento: {(time.time()-t_train)/60:.1f} minutos", flush=True)

    # ── 4. Evaluar ───────────────────────────────────────────────
    print("\n[4/5] Evaluando...", flush=True)
    m_val  = evaluar(model, X_val,  y_val,  "val")
    m_test = evaluar(model, X_test, y_test, "test")
    comparar_con_v1(m_val, m_test)

    # ── 5. Guardar ───────────────────────────────────────────────
    print("\n[5/5] Guardando outputs...", flush=True)

    model_path = OUTPUT_DIR / "model_lgbm_v2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": ALL_FEATURES, "version": "v2"}, f)
    print(f"  Modelo: {model_path}", flush=True)

    results_path = OUTPUT_DIR / "results_v2.json"
    with open(results_path, "w") as f:
        json.dump({"val": m_val, "test": m_test}, f, indent=2)
    print(f"  Métricas: {results_path}", flush=True)

    fi_path = OUTPUT_DIR / "feature_importance_v2.csv"
    guardar_fi(model, fi_path)

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("RESUMEN FINAL — CLASIFICACIÓN v2", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Tiempo total:      {total:.1f} minutos", flush=True)
    print(f"  AUC-ROC val:       {m_val['auc_roc']}", flush=True)
    print(f"  AUC-ROC test:      {m_test['auc_roc']}", flush=True)
    print(f"  F1 disrupted val:  {m_val['f1_disrupted']}", flush=True)
    print(f"  F1 disrupted test: {m_test['f1_disrupted']}", flush=True)
    print(f"\n  Outputs en: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

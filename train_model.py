"""
Entrenamiento de modelo LightGBM para predicción de disrupciones en órdenes.

Problema: clasificación binaria (is_disrupted = 0/1)
Dataset:  staging_marts.ml_dataset_v1
Split:    temporal (train=2025-01/11, val=2025-12/2026-01, test=2026-02)

Uso:
    python train_model.py

Outputs:
    model_lgbm_v1.pkl        → modelo entrenado
    results_v1.json          → métricas de evaluación
    feature_importance_v1.csv→ importancia de features
"""

import os
import gc
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────

load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Features que usa el modelo (ex-ante, sin leakage)
FEATURES_NUMERICAS = [
    "is_click_and_collect",
    "is_high_season",
    "has_insurance",
    "total_items",
    "distinct_skus",
    "dia_semana_creacion",
    "hora_creacion",
    "dia_mes_creacion",
    "sla_horas_prometidas",
]

FEATURES_CATEGORICAS = [
    "service_category",
    "source_order_type",
    "seller_id",
    "created_by",
]

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

TARGET = "target_binario"

# Columnas a leer de la tabla (features + target + split + id)
COLS_A_LEER = ALL_FEATURES + [TARGET, "split_set", "logistic_order_id"]

CHUNK_SIZE = 500_000  # filas por chunk al leer


# ──────────────────────────────────────────────
# CONEXIÓN A RDS
# ──────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


# ──────────────────────────────────────────────
# CARGA DE DATOS EN CHUNKS
# ──────────────────────────────────────────────

def cargar_split(engine, split_name: str) -> pd.DataFrame:
    """Lee un split completo desde Postgres en chunks para no agotar memoria."""
    print(f"  Leyendo split '{split_name}'...", flush=True)
    t0 = time.time()

    cols_sql = ", ".join(COLS_A_LEER)
    query = f"""
        SELECT {cols_sql}
        FROM staging_marts.ml_dataset_v1
        WHERE split_set = '{split_name}'
    """

    chunks = []
    total_filas = 0
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(text(query))
        while True:
            rows = result.fetchmany(CHUNK_SIZE)
            if not rows:
                break
            chunk = pd.DataFrame(rows, columns=COLS_A_LEER)
            chunks.append(chunk)
            total_filas += len(chunk)
            print(f"    ...{total_filas:,} filas leídas", flush=True)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    elapsed = time.time() - t0
    print(f"  '{split_name}' listo: {len(df):,} filas en {elapsed:.1f}s", flush=True)
    return df


# ──────────────────────────────────────────────
# PREPROCESAMIENTO
# ──────────────────────────────────────────────

def preprocesar(df: pd.DataFrame, encoders: dict = None, fit: bool = False):
    """
    Prepara el dataframe para LightGBM.
    - Numéricas: cast a float, rellena nulos con -1
    - Categóricas: Label Encoding, nulos como string "DESCONOCIDO"
    
    Si fit=True, ajusta los encoders (para train).
    Si fit=False, usa encoders ya ajustados (para val/test).
    Devuelve (X, y, encoders).
    """
    df = df.copy()

    # Numéricas: nulos a -1 (LightGBM los maneja bien)
    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")

    # Categóricas: nulos a string, luego label encode
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
            # Valores no vistos en train → categoria "DESCONOCIDO"
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "DESCONOCIDO")
            df[col] = le.transform(df[col]).astype("int32")

    X = df[ALL_FEATURES]
    y = df[TARGET].astype("int8")
    return X, y, encoders


# ──────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────

def entrenar(X_train, y_train, X_val, y_val):
    """Entrena LightGBM con early stopping en val."""
    print("\nEntrenando LightGBM...", flush=True)

    # Parámetros: conservadores para primera corrida
    # Después podemos hacer hyperparameter tuning con Optuna
    params = {
        "objective":        "binary",
        "metric":           ["binary_logloss", "auc"],
        "boosting_type":    "gbdt",
        "n_estimators":     1000,       # early stopping va a cortar antes
        "learning_rate":    0.05,
        "num_leaves":       63,         # 2^6 - 1; bueno para datasets grandes
        "max_depth":        -1,         # sin límite explícito
        "min_child_samples": 200,       # evita overfit en hojas pequeñas
        "feature_fraction": 0.8,        # subsampling de features por árbol
        "bagging_fraction": 0.8,        # subsampling de filas
        "bagging_freq":     5,
        "lambda_l1":        0.1,
        "lambda_l2":        0.1,
        "scale_pos_weight": 1.0,        # dataset ~balanceado, no necesitamos ajuste
        "n_jobs":           -1,         # usar todos los cores del EC2
        "verbose":          -1,
        "random_state":     42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=callbacks,
        categorical_feature=[ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS],
    )

    print(f"\nMejor iteración: {model.best_iteration_}", flush=True)
    return model


# ──────────────────────────────────────────────
# EVALUACIÓN
# ──────────────────────────────────────────────

def evaluar(model, X, y, split_name: str) -> dict:
    """Evalúa el modelo y devuelve las métricas principales."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc  = roc_auc_score(y, y_pred_proba)
    auc_pr   = average_precision_score(y, y_pred_proba)
    report   = classification_report(y, y_pred, output_dict=True)
    cm       = confusion_matrix(y, y_pred).tolist()

    metricas = {
        "split":     split_name,
        "n":         len(y),
        "auc_roc":   round(auc_roc,  4),
        "auc_pr":    round(auc_pr,   4),
        "precision_disrupted": round(report["1"]["precision"], 4),
        "recall_disrupted":    round(report["1"]["recall"],    4),
        "f1_disrupted":        round(report["1"]["f1-score"],  4),
        "precision_ok":        round(report["0"]["precision"], 4),
        "recall_ok":           round(report["0"]["recall"],    4),
        "f1_ok":               round(report["0"]["f1-score"],  4),
        "accuracy":            round(report["accuracy"],       4),
        "confusion_matrix":    cm,
    }

    print(f"\n{'='*50}", flush=True)
    print(f"MÉTRICAS — {split_name.upper()}", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  AUC-ROC:              {auc_roc:.4f}", flush=True)
    print(f"  AUC-PR (avg prec):    {auc_pr:.4f}", flush=True)
    print(f"  Accuracy:             {report['accuracy']:.4f}", flush=True)
    print(f"\n  Clase DISRUPTED (1):", flush=True)
    print(f"    Precision:  {report['1']['precision']:.4f}", flush=True)
    print(f"    Recall:     {report['1']['recall']:.4f}", flush=True)
    print(f"    F1:         {report['1']['f1-score']:.4f}", flush=True)
    print(f"\n  Clase OK (0):", flush=True)
    print(f"    Precision:  {report['0']['precision']:.4f}", flush=True)
    print(f"    Recall:     {report['0']['recall']:.4f}", flush=True)
    print(f"    F1:         {report['0']['f1-score']:.4f}", flush=True)
    print(f"\n  Confusion matrix:", flush=True)
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}", flush=True)
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}", flush=True)

    return metricas


# ──────────────────────────────────────────────
# FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def guardar_feature_importance(model, path: Path):
    fi = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "importance_gain":  model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    fi.to_csv(path, index=False)
    print(f"\nFeature importance guardada en: {path}", flush=True)
    print(fi.to_string(index=False), flush=True)
    return fi


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    t_inicio = time.time()
    print("=" * 60, flush=True)
    print("ENTRENAMIENTO LGBM — CLASIFICACIÓN DISRUPCIONES", flush=True)
    print("=" * 60, flush=True)

    engine = get_engine()

    # ── 1. Cargar datos ──
    print("\n[1/5] Cargando datos...", flush=True)
    df_train = cargar_split(engine, "train")
    df_val   = cargar_split(engine, "val")
    df_test  = cargar_split(engine, "test")

    print(f"\nTamaños: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}", flush=True)

    # ── 2. Preprocesar ──
    print("\n[2/5] Preprocesando...", flush=True)
    X_train, y_train, encoders = preprocesar(df_train, fit=True)
    X_val,   y_val,   _        = preprocesar(df_val,   encoders=encoders, fit=False)
    X_test,  y_test,  _        = preprocesar(df_test,  encoders=encoders, fit=False)

    del df_train, df_val, df_test
    gc.collect()

    print(f"  Features: {len(ALL_FEATURES)} ({len(FEATURES_NUMERICAS)} numéricas, {len(FEATURES_CATEGORICAS)} categóricas)", flush=True)
    print(f"  Train disrupted: {y_train.sum():,} / {len(y_train):,} ({100*y_train.mean():.1f}%)", flush=True)
    print(f"  Val   disrupted: {y_val.sum():,} / {len(y_val):,} ({100*y_val.mean():.1f}%)", flush=True)
    print(f"  Test  disrupted: {y_test.sum():,} / {len(y_test):,} ({100*y_test.mean():.1f}%)", flush=True)

    # ── 3. Entrenar ──
    print("\n[3/5] Entrenando...", flush=True)
    t_train = time.time()
    model = entrenar(X_train, y_train, X_val, y_val)
    print(f"\nEntrenamiento: {(time.time()-t_train)/60:.1f} minutos", flush=True)

    # ── 4. Evaluar ──
    print("\n[4/5] Evaluando...", flush=True)
    metricas_val  = evaluar(model, X_val,  y_val,  "val")
    metricas_test = evaluar(model, X_test, y_test, "test")

    # ── 5. Guardar outputs ──
    print("\n[5/5] Guardando outputs...", flush=True)

    # Modelo
    model_path = OUTPUT_DIR / "model_lgbm_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": ALL_FEATURES}, f)
    print(f"  Modelo guardado: {model_path}", flush=True)

    # Métricas
    results_path = OUTPUT_DIR / "results_v1.json"
    with open(results_path, "w") as f:
        json.dump({"val": metricas_val, "test": metricas_test}, f, indent=2)
    print(f"  Métricas guardadas: {results_path}", flush=True)

    # Feature importance
    fi_path = OUTPUT_DIR / "feature_importance_v1.csv"
    guardar_feature_importance(model, fi_path)

    # ── Resumen final ──
    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"RESUMEN FINAL", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Tiempo total:     {total:.1f} minutos", flush=True)
    print(f"  AUC-ROC val:      {metricas_val['auc_roc']}", flush=True)
    print(f"  AUC-ROC test:     {metricas_test['auc_roc']}", flush=True)
    print(f"  F1 disrupted val: {metricas_val['f1_disrupted']}", flush=True)
    print(f"  F1 disrupted test:{metricas_test['f1_disrupted']}", flush=True)
    print(f"\n  Outputs en: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

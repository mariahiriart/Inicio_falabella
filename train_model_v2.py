"""
Entrenamiento v2 — LightGBM clasificación binaria disrupciones.
Agrega features históricas (target encoding) calculadas en staging_marts.ml_historical_features.

Mejoras respecto a v1:
- seller_tasa_disrupcion, seller_n_ordenes
- categoria_tasa_disrupcion
- categoria_cac_tasa_disrupcion (interacción service_category × is_click_and_collect)
- dia_semana_tasa_disrupcion
- franja_horaria + franja_tasa_disrupcion
- seller_id removido (reemplazado por seller_tasa_disrupcion, más informativa)
- source_order_type removido (casi sin importancia en v1)

Uso:
    python train_model_v2.py

Outputs en ml_outputs/:
    model_lgbm_v2.pkl
    results_v2.json
    feature_importance_v2.csv
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

# Features numéricas — v1 + nuevas históricas
FEATURES_NUMERICAS = [
    # v1
    "is_click_and_collect",
    "is_high_season",
    "has_insurance",
    "total_items",
    "distinct_skus",
    "dia_semana_creacion",
    "hora_creacion",
    "dia_mes_creacion",
    "sla_horas_prometidas",
    # nuevas v2
    "seller_n_ordenes",
    "seller_tasa_disrupcion",
    "categoria_tasa_disrupcion",
    "categoria_cac_tasa_disrupcion",
    "dia_semana_tasa_disrupcion",
    "franja_tasa_disrupcion",
]

# Features categóricas — removemos seller_id y source_order_type (baja importancia en v1)
# Agregamos franja_horaria
FEATURES_CATEGORICAS = [
    "service_category",
    "created_by",
    "franja_horaria",
]

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
TARGET = "target_binario"

CHUNK_SIZE = 500_000


# ──────────────────────────────────────────────
# CONEXIÓN
# ──────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


# ──────────────────────────────────────────────
# CARGA — JOIN con features históricas
# ──────────────────────────────────────────────

def cargar_split(engine, split_name: str) -> pd.DataFrame:
    """
    Lee un split joineando ml_dataset_v1 con ml_historical_features.
    Las features históricas se calcularon SOLO sobre train, así que
    no hay leakage hacia val/test.
    """
    print(f"  Leyendo split '{split_name}'...", flush=True)
    t0 = time.time()

    cols_base = [
        "d.logistic_order_id",
        "d.service_category",
        "d.is_click_and_collect",
        "d.is_high_season",
        "d.has_insurance",
        "d.total_items",
        "d.distinct_skus",
        "d.dia_semana_creacion",
        "d.hora_creacion",
        "d.dia_mes_creacion",
        "d.sla_horas_prometidas",
        "d.created_by",
        "d.target_binario",
        "d.split_set",
    ]

    cols_hist = [
        "h.seller_n_ordenes",
        "h.seller_tasa_disrupcion",
        "h.categoria_tasa_disrupcion",
        "h.categoria_cac_tasa_disrupcion",
        "h.dia_semana_tasa_disrupcion",
        "h.franja_horaria",
        "h.franja_tasa_disrupcion",
    ]

    all_cols = cols_base + cols_hist
    select_clause = ", ".join(all_cols)

    query = f"""
        SELECT {select_clause}
        FROM staging_marts.ml_dataset_v1 d
        JOIN staging_marts.ml_historical_features h
          ON d.logistic_order_id = h.logistic_order_id
        WHERE d.split_set = '{split_name}'
    """

    chunks = []
    total_filas = 0
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(text(query))
        col_names = [
            "logistic_order_id", "service_category", "is_click_and_collect",
            "is_high_season", "has_insurance", "total_items", "distinct_skus",
            "dia_semana_creacion", "hora_creacion", "dia_mes_creacion",
            "sla_horas_prometidas", "created_by", "target_binario", "split_set",
            "seller_n_ordenes", "seller_tasa_disrupcion", "categoria_tasa_disrupcion",
            "categoria_cac_tasa_disrupcion", "dia_semana_tasa_disrupcion",
            "franja_horaria", "franja_tasa_disrupcion",
        ]
        while True:
            rows = result.fetchmany(CHUNK_SIZE)
            if not rows:
                break
            chunk = pd.DataFrame(rows, columns=col_names)
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
    df = df.copy()

    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")

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


# ──────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────

def entrenar(X_train, y_train, X_val, y_val):
    print("\nEntrenando LightGBM v2...", flush=True)

    params = {
        "objective":         "binary",
        "metric":            ["binary_logloss", "auc"],
        "boosting_type":     "gbdt",
        "n_estimators":      1000,
        "learning_rate":     0.05,
        "num_leaves":        127,        # más capacidad que v1 (63)
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
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, y_pred_proba)
    auc_pr  = average_precision_score(y, y_pred_proba)
    report  = classification_report(y, y_pred, output_dict=True)
    cm      = confusion_matrix(y, y_pred).tolist()

    metricas = {
        "split":               split_name,
        "n":                   len(y),
        "auc_roc":             round(auc_roc, 4),
        "auc_pr":              round(auc_pr,  4),
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
    print(f"  AUC-ROC:           {auc_roc:.4f}", flush=True)
    print(f"  AUC-PR:            {auc_pr:.4f}",  flush=True)
    print(f"  Accuracy:          {report['accuracy']:.4f}", flush=True)
    print(f"\n  Clase DISRUPTED (1):", flush=True)
    print(f"    Precision: {report['1']['precision']:.4f}", flush=True)
    print(f"    Recall:    {report['1']['recall']:.4f}",    flush=True)
    print(f"    F1:        {report['1']['f1-score']:.4f}",  flush=True)
    print(f"\n  Clase OK (0):", flush=True)
    print(f"    Precision: {report['0']['precision']:.4f}", flush=True)
    print(f"    Recall:    {report['0']['recall']:.4f}",    flush=True)
    print(f"    F1:        {report['0']['f1-score']:.4f}",  flush=True)
    print(f"\n  Confusion matrix:", flush=True)
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}", flush=True)
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}", flush=True)

    return metricas


# ──────────────────────────────────────────────
# COMPARACIÓN v1 vs v2
# ──────────────────────────────────────────────

def comparar_con_v1(metricas_val, metricas_test):
    # Métricas de v1 hardcodeadas para referencia
    v1 = {
        "val":  {"auc_roc": 0.9296, "f1_disrupted": 0.8026, "accuracy": 0.8687},
        "test": {"auc_roc": 0.8948, "f1_disrupted": 0.8421, "accuracy": 0.8303},
    }

    print(f"\n{'='*50}", flush=True)
    print("COMPARACIÓN v1 vs v2", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"{'Métrica':<25} {'v1 val':>8} {'v2 val':>8} {'v1 test':>8} {'v2 test':>8}", flush=True)
    print("-" * 55, flush=True)

    metricas_a_comparar = ["auc_roc", "f1_disrupted", "accuracy"]
    for m in metricas_a_comparar:
        v1v = v1["val"][m]
        v2v = metricas_val[m]
        v1t = v1["test"][m]
        v2t = metricas_test[m]
        delta_val  = f"({'+'if v2v>=v1v else ''}{v2v-v1v:.4f})"
        delta_test = f"({'+'if v2t>=v1t else ''}{v2t-v1t:.4f})"
        print(f"  {m:<23} {v1v:>8.4f} {v2v:>8.4f}{delta_val:>10}   {v1t:>8.4f} {v2t:>8.4f}{delta_test:>10}", flush=True)


# ──────────────────────────────────────────────
# FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def guardar_feature_importance(model, path: Path):
    fi = pd.DataFrame({
        "feature":          ALL_FEATURES,
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
    print("ENTRENAMIENTO LGBM v2 — CON FEATURES HISTÓRICAS", flush=True)
    print("=" * 60, flush=True)
    print(f"Features totales: {len(ALL_FEATURES)}", flush=True)
    print(f"  Numéricas ({len(FEATURES_NUMERICAS)}): {FEATURES_NUMERICAS}", flush=True)
    print(f"  Categóricas ({len(FEATURES_CATEGORICAS)}): {FEATURES_CATEGORICAS}", flush=True)

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
    comparar_con_v1(metricas_val, metricas_test)

    # ── 5. Guardar ──
    print("\n[5/5] Guardando outputs...", flush=True)

    model_path = OUTPUT_DIR / "model_lgbm_v2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":    model,
            "encoders": encoders,
            "features": ALL_FEATURES,
            "version":  "v2",
        }, f)
    print(f"  Modelo guardado: {model_path}", flush=True)

    results_path = OUTPUT_DIR / "results_v2.json"
    with open(results_path, "w") as f:
        json.dump({"val": metricas_val, "test": metricas_test}, f, indent=2)
    print(f"  Métricas guardadas: {results_path}", flush=True)

    fi_path = OUTPUT_DIR / "feature_importance_v2.csv"
    guardar_feature_importance(model, fi_path)

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("RESUMEN FINAL v2", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Tiempo total:      {total:.1f} minutos", flush=True)
    print(f"  AUC-ROC val:       {metricas_val['auc_roc']}", flush=True)
    print(f"  AUC-ROC test:      {metricas_test['auc_roc']}", flush=True)
    print(f"  F1 disrupted val:  {metricas_val['f1_disrupted']}", flush=True)
    print(f"  F1 disrupted test: {metricas_test['f1_disrupted']}", flush=True)
    print(f"\n  Outputs en: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

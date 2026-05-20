"""
Modelo B — Diagnóstico post-facto de disrupciones.

Objetivo: dado que una orden YA ocurrió, identificar en qué tramo
ocurrió la disrupción y quién fue el actor responsable.

Problema: clasificación multiclase
Target:   target_tramo (sin_disrupcion / ultima_milla / deposito / despacho)

Features: todas las del v2 + features del ciclo completo
  (hours_deposito, hours_despacho, hours_ultima_milla,
   delta_deposito_vs_p95, delta_despacho_vs_p95, delta_ultima_milla_vs_p95)

IMPORTANTE: este modelo NO es para predicción ex-ante.
  Usa información que solo existe DESPUÉS de completado el ciclo.
  Sirve para diagnóstico operativo y análisis de causa raíz.

Dataset: staging_marts.ml_dataset_v1 JOIN staging_marts.fct_orders
Split:   mismo temporal que v1/v2

Uso:
    python train_modelo_b.py

Outputs en ml_outputs/:
    model_lgbm_b.pkl
    results_b.json
    feature_importance_b.csv
"""

import os, gc, json, time, pickle, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Features ────────────────────────────────────────────────────────────────

# Features ex-ante (conocidas al crear la orden)
FEATURES_EXANTE_NUM = [
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

# Features post-facto (conocidas DESPUÉS del ciclo — solo para diagnóstico)
FEATURES_POSTFACTO_NUM = [
    "hours_deposito",
    "hours_despacho",
    "hours_ultima_milla",
    "hours_total",
    "delta_deposito_vs_p95",
    "delta_despacho_vs_p95",
    "delta_ultima_milla_vs_p95",
]

FEATURES_CATEGORICAS = [
    "service_category",
    "franja_horaria",
]

ALL_FEATURES = FEATURES_EXANTE_NUM + FEATURES_POSTFACTO_NUM + FEATURES_CATEGORICAS

# Target multiclase
TARGET = "target_tramo"

# Clases del target
CLASES = ["sin_disrupcion", "deposito", "despacho", "ultima_milla"]

CHUNK_SIZE = 400_000


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def cargar_split(engine, split_name: str) -> pd.DataFrame:
    """
    Carga un split haciendo JOIN entre ml_dataset_v1 (features ex-ante + split)
    y fct_orders (features post-facto del ciclo completo).
    Solo incluye órdenes con ciclo completo.
    """
    print(f"  Cargando '{split_name}'...", flush=True)
    t0 = time.time()

    # Construir lista de columnas
    exante_cols   = ", ".join([f"d.{c}" for c in FEATURES_EXANTE_NUM])
    postfacto_cols = ", ".join([f"f.{c}" for c in FEATURES_POSTFACTO_NUM])
    cat_cols      = "d.service_category, d.franja_horaria"

    query = text(f"""
        SELECT
            d.logistic_order_id,
            d.split_set,
            d.target_tramo,
            d.target_actor,
            {exante_cols},
            {postfacto_cols},
            {cat_cols}
        FROM staging_marts.ml_dataset_v2 d
        JOIN staging_marts.fct_orders f
          ON d.logistic_order_id = f.logistic_order_id
        WHERE d.split_set = :split
          AND d.ciclo_completo = 1
          AND f.hours_total > 0
          AND d.target_tramo IS NOT NULL
    """)

    chunks, total = [], 0
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(
            query, {"split": split_name}
        )
        while True:
            rows = result.fetchmany(CHUNK_SIZE)
            if not rows:
                break
            cols = (
                ["logistic_order_id", "split_set", "target_tramo", "target_actor"]
                + FEATURES_EXANTE_NUM
                + FEATURES_POSTFACTO_NUM
                + ["service_category", "franja_horaria"]
            )
            chunks.append(pd.DataFrame(rows, columns=cols))
            total += len(chunks[-1])
            print(f"    ...{total:,} filas", flush=True)

    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    elapsed = time.time() - t0
    print(f"  '{split_name}' listo: {len(df):,} filas en {elapsed:.1f}s", flush=True)

    # Distribución del target
    print(f"  Distribución target_tramo:", flush=True)
    vc = df["target_tramo"].value_counts()
    for tramo, n in vc.items():
        print(f"    {tramo:<20} {n:>8,} ({100*n/len(df):.1f}%)", flush=True)

    return df


def preprocesar(df, encoders=None, label_encoder_target=None, fit=False):
    df = df.copy()

    # Numéricas: nulos a -1
    for col in FEATURES_EXANTE_NUM + FEATURES_POSTFACTO_NUM:
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

    # Target: label encoding multiclase
    if fit:
        label_encoder_target = LabelEncoder()
        y = label_encoder_target.fit_transform(
            df["target_tramo"].fillna("sin_disrupcion").astype(str)
        ).astype("int32")
    else:
        y = label_encoder_target.transform(
            df["target_tramo"].fillna("sin_disrupcion").astype(str)
        ).astype("int32")

    X = df[ALL_FEATURES]
    return X, y, encoders, label_encoder_target


def entrenar(X_train, y_train, X_val, y_val, n_clases):
    print(f"\nEntrenando LightGBM Modelo B ({n_clases} clases)...", flush=True)

    params = {
        "objective":         "multiclass",
        "num_class":         n_clases,
        "metric":            "multi_logloss",
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


def evaluar(model, X, y, split_name, le_target):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc    = accuracy_score(y, y_pred)
    f1_mac = f1_score(y, y_pred, average="macro")
    f1_wt  = f1_score(y, y_pred, average="weighted")
    report = classification_report(y, y_pred,
                                   target_names=le_target.classes_,
                                   output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()

    print(f"\n{'='*60}", flush=True)
    print(f"MÉTRICAS MODELO B — {split_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Accuracy:        {acc:.4f}", flush=True)
    print(f"  F1 macro:        {f1_mac:.4f}", flush=True)
    print(f"  F1 weighted:     {f1_wt:.4f}", flush=True)
    print(f"\n  Por clase:", flush=True)
    for clase in le_target.classes_:
        if clase in report:
            r = report[clase]
            print(f"    {clase:<20} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  n={int(r['support']):,}", flush=True)

    print(f"\n  Matriz de confusión ({list(le_target.classes_)}):", flush=True)
    for i, row in enumerate(cm):
        print(f"    {le_target.classes_[i]:<20} {row}", flush=True)

    metricas = {
        "split":        split_name,
        "n":            int(len(y)),
        "accuracy":     round(float(acc), 4),
        "f1_macro":     round(float(f1_mac), 4),
        "f1_weighted":  round(float(f1_wt), 4),
        "por_clase":    {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                         for k, v in report.items() if k in le_target.classes_},
        "confusion_matrix": cm,
        "clases": list(le_target.classes_),
    }
    return metricas


def guardar_fi(model, path):
    fi = pd.DataFrame({
        "feature":          ALL_FEATURES,
        "importance_gain":  model.booster_.feature_importance("gain"),
        "importance_split": model.booster_.feature_importance("split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(path, index=False)
    print(f"\nFeature importance guardada en: {path}", flush=True)
    print(fi.head(15).to_string(index=False), flush=True)
    return fi


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("MODELO B — DIAGNÓSTICO POST-FACTO DE DISRUPCIONES", flush=True)
    print("Clasificación multiclase: sin_disrupcion / deposito / despacho / ultima_milla", flush=True)
    print(f"Features: {len(ALL_FEATURES)} ({len(FEATURES_EXANTE_NUM)} ex-ante, {len(FEATURES_POSTFACTO_NUM)} post-facto, {len(FEATURES_CATEGORICAS)} cat)", flush=True)
    print("="*60, flush=True)

    engine = get_engine()

    # ── 1. Cargar ────────────────────────────────────────────────
    print("\n[1/5] Cargando datos...", flush=True)
    df_train = cargar_split(engine, "train")
    df_val   = cargar_split(engine, "val")
    df_test  = cargar_split(engine, "test")
    print(f"\n  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}", flush=True)

    # ── 2. Preprocesar ───────────────────────────────────────────
    print("\n[2/5] Preprocesando...", flush=True)
    X_train, y_train, encoders, le_target = preprocesar(df_train, fit=True)
    X_val,   y_val,   _,        _         = preprocesar(df_val,   encoders=encoders, label_encoder_target=le_target)
    X_test,  y_test,  _,        _         = preprocesar(df_test,  encoders=encoders, label_encoder_target=le_target)

    n_clases = len(le_target.classes_)
    print(f"  Clases ({n_clases}): {list(le_target.classes_)}", flush=True)

    del df_train, df_val, df_test; gc.collect()

    # ── 3. Entrenar ──────────────────────────────────────────────
    print("\n[3/5] Entrenando...", flush=True)
    t_train = time.time()
    model = entrenar(X_train, y_train, X_val, y_val, n_clases)
    print(f"  Entrenamiento: {(time.time()-t_train)/60:.1f} minutos", flush=True)

    # ── 4. Evaluar ───────────────────────────────────────────────
    print("\n[4/5] Evaluando...", flush=True)
    m_val  = evaluar(model, X_val,  y_val,  "val",  le_target)
    m_test = evaluar(model, X_test, y_test, "test", le_target)

    # ── 5. Guardar ───────────────────────────────────────────────
    print("\n[5/5] Guardando outputs...", flush=True)

    model_path = OUTPUT_DIR / "model_lgbm_b.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":          model,
            "encoders":       encoders,
            "le_target":      le_target,
            "features":       ALL_FEATURES,
            "features_exante":    FEATURES_EXANTE_NUM,
            "features_postfacto": FEATURES_POSTFACTO_NUM,
            "version":        "modelo_b_v1",
            "target":         "target_tramo",
        }, f)
    print(f"  Modelo: {model_path}", flush=True)

    results_path = OUTPUT_DIR / "results_b.json"
    with open(results_path, "w") as f:
        json.dump({"val": m_val, "test": m_test}, f, indent=2)
    print(f"  Métricas: {results_path}", flush=True)

    fi_path = OUTPUT_DIR / "feature_importance_b.csv"
    guardar_fi(model, fi_path)

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("RESUMEN FINAL — MODELO B", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Tiempo total:     {total:.1f} minutos", flush=True)
    print(f"  Accuracy val:     {m_val['accuracy']}", flush=True)
    print(f"  F1 macro val:     {m_val['f1_macro']}", flush=True)
    print(f"  Accuracy test:    {m_test['accuracy']}", flush=True)
    print(f"  F1 macro test:    {m_test['f1_macro']}", flush=True)
    print(f"\n  Outputs en: {OUTPUT_DIR}", flush=True)
    print("\n  RECORDATORIO: este modelo usa features post-facto.", flush=True)
    print("  NO usar para predicción en tiempo real.", flush=True)
    print("  Usar para diagnóstico después de completado el ciclo.", flush=True)


if __name__ == "__main__":
    main()

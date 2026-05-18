"""
Entrenamiento incremental por mes — LightGBM v2.
Carga un mes a la vez para evitar problemas de RAM.

Uso:
    python train_incremental.py

Outputs en ml_outputs/:
    model_lgbm_v2_incremental.pkl
"""

import os
import gc
import json
import time
import pickle
import warnings
import pandas as pd
import lightgbm as lgb

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

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
    "seller_n_ordenes",
    "seller_tasa_disrupcion",
    "categoria_tasa_disrupcion",
    "categoria_cac_tasa_disrupcion",
    "dia_semana_tasa_disrupcion",
    "franja_tasa_disrupcion",
]

FEATURES_CATEGORICAS = [
    "service_category",
    "created_by",
    "franja_horaria",
]

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
TARGET = "target_binario"

MESES_TRAIN = [
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11",
]

PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "num_leaves":        127,
    "learning_rate":     0.05,
    "min_child_samples": 100,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "n_jobs":            -1,
    "verbose":           -1,
    "random_state":      42,
}


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def cargar_mes(engine, mes, encoders=None, fit=False):
    """Carga un mes completo desde ml_dataset_v2."""
    print(f"  Cargando {mes}...", flush=True)
    cols = ALL_FEATURES + [TARGET]
    cols_sql = ", ".join(cols)
    query = text(f"SELECT {cols_sql} FROM staging_marts.ml_dataset_v2 WHERE year_month = :mes")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"mes": mes})

    # Preprocesar numéricas
    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")

    # Preprocesar categóricas
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

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype("int8").copy()
    del df
    gc.collect()
    return X, y, encoders


def cargar_split(engine, split_name, encoders):
    """Carga val o test en chunks para no explotar RAM."""
    print(f"  Cargando split '{split_name}'...", flush=True)
    cols = ALL_FEATURES + [TARGET]
    cols_sql = ", ".join(cols)
    query = text(f"SELECT {cols_sql} FROM staging_marts.ml_dataset_v2 WHERE split_set = :split")

    chunks = []
    total = 0
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(query, {"split": split_name})
        while True:
            rows = result.fetchmany(200_000)
            if not rows:
                break
            chunk = pd.DataFrame(rows, columns=cols)
            chunks.append(chunk)
            total += len(chunk)
            print(f"    ...{total:,} filas", flush=True)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")
    for col in FEATURES_CATEGORICAS:
        le = encoders[col]
        df[col] = df[col].fillna("DESCONOCIDO").astype(str)
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "DESCONOCIDO")
        df[col] = le.transform(df[col]).astype("int32")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype("int8").copy()
    del df
    gc.collect()
    return X, y


def evaluar(booster, X, y, split_name):
    y_prob = booster.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y, y_prob)
    rep = classification_report(y, y_pred, output_dict=True)
    f1 = rep["1"]["f1-score"]
    acc = rep["accuracy"]
    print(f"\n{'='*50}", flush=True)
    print(f"METRICAS — {split_name.upper()}", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  AUC-ROC:     {auc:.4f}", flush=True)
    print(f"  F1-disrupted:{f1:.4f}", flush=True)
    print(f"  Accuracy:    {acc:.4f}", flush=True)
    print(f"  Precision:   {rep['1']['precision']:.4f}", flush=True)
    print(f"  Recall:      {rep['1']['recall']:.4f}", flush=True)
    return {"auc_roc": round(auc, 4), "f1": round(f1, 4), "accuracy": round(acc, 4)}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("ENTRENAMIENTO INCREMENTAL POR MES — LGBM v2", flush=True)
    print("=" * 60, flush=True)
    print(f"Features: {len(ALL_FEATURES)} ({len(FEATURES_NUMERICAS)} num + {len(FEATURES_CATEGORICAS)} cat)", flush=True)
    print(f"Meses de train: {MESES_TRAIN}", flush=True)

    engine = get_engine()
    booster = None
    encoders = None

    # ── Entrenamiento mes a mes ──────────────────────────────────────────
    for i, mes in enumerate(MESES_TRAIN):
        t_mes = time.time()
        print(f"\n[Mes {i+1}/{len(MESES_TRAIN)}] {mes}", flush=True)

        fit = (i == 0)  # solo en el primer mes ajustamos los encoders
        X, y, encoders = cargar_mes(engine, mes, encoders=encoders, fit=fit)

        print(f"  {len(X):,} filas  ({int(y.mean()*100)}% disrupted)", flush=True)

        cat_idx = [ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS]
        ds = lgb.Dataset(X, label=y, categorical_feature=cat_idx, free_raw_data=True)

        # Primer mes: 200 iteraciones. Meses siguientes: 50 iteraciones adicionales.
        n_iter = 200 if i == 0 else 50

        booster = lgb.train(
            PARAMS,
            ds,
            num_boost_round=n_iter,
            init_model=booster,
            keep_training_booster=True,
        )

        del X, y, ds
        gc.collect()

        elapsed_mes = time.time() - t_mes
        print(f"  Iteraciones totales: {booster.num_trees()}  ({elapsed_mes:.1f}s)", flush=True)

    # ── Evaluación final ─────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("EVALUACION FINAL", flush=True)
    print(f"{'='*60}", flush=True)

    X_val, y_val = cargar_split(engine, "val", encoders)
    metricas_val = evaluar(booster, X_val, y_val, "val")
    del X_val, y_val
    gc.collect()

    X_test, y_test = cargar_split(engine, "test", encoders)
    metricas_test = evaluar(booster, X_test, y_test, "test")
    del X_test, y_test
    gc.collect()

    # Comparación con v1
    print(f"\n{'='*60}", flush=True)
    print("COMPARACION v1 vs v2 incremental", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Metrica':<20} {'v1 val':>8} {'v2 val':>8}   {'v1 test':>8} {'v2 test':>8}", flush=True)
    print(f"  {'-'*56}", flush=True)
    v1 = {"val": {"auc_roc": 0.9296, "f1": 0.8026}, "test": {"auc_roc": 0.8948, "f1": 0.8421}}
    for m in ["auc_roc", "f1"]:
        print(f"  {m:<20} {v1['val'][m]:>8.4f} {metricas_val[m]:>8.4f}   "
              f"{v1['test'][m]:>8.4f} {metricas_test[m]:>8.4f}", flush=True)

    # ── Guardar ──────────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "model_lgbm_v2_incremental.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "booster":  booster,
            "encoders": encoders,
            "features": ALL_FEATURES,
            "version":  "v2_incremental",
        }, f)
    print(f"\nModelo guardado: {model_path}", flush=True)

    results_path = OUTPUT_DIR / "results_v2_incremental.json"
    with open(results_path, "w") as f:
        json.dump({"val": metricas_val, "test": metricas_test}, f, indent=2)

    total = (time.time() - t0) / 60
    print(f"\nTiempo total: {total:.1f} minutos", flush=True)
    print(f"AUC-ROC val={metricas_val['auc_roc']}  test={metricas_test['auc_roc']}", flush=True)


if __name__ == "__main__":
    main()

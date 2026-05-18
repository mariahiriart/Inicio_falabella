"""
Entrenamiento LightGBM — Regresión para predecir hours_total.
Solo usa órdenes con ciclo completo (hours_total IS NOT NULL).

Uso:
    python train_regression.py

Outputs en ml_outputs/:
    model_lgbm_reg_v1.pkl
    results_reg_v1.json
    feature_importance_reg_v1.csv
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mismas features que el clasificador v1
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
TARGET = "target_horas_totales"  # = hours_total
CHUNK_SIZE = 400_000


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def encode_col(series, le):
    """Encodea una columna categórica de forma segura."""
    known = set(str(c) for c in le.classes_)
    clean = [str(v) if str(v) in known else "DESCONOCIDO" for v in list(series)]
    return le.transform(clean)


def cargar_split(engine, split_name, encoders=None, fit=False):
    """
    Carga un split filtrando solo órdenes con ciclo completo
    (hours_total IS NOT NULL) y target > 0.
    """
    print(f"  Cargando split '{split_name}'...", flush=True)
    t0 = time.time()

    cols_extra = ["logistic_order_id", "year_month", "split_set", TARGET]
    cols_leer  = cols_extra + ALL_FEATURES

    # Mapeo: en ml_dataset_v1 el target se llama target_horas_totales
    cols_sql = ", ".join([
        "logistic_order_id",
        "year_month",
        "split_set",
        "target_horas_totales",
    ] + ALL_FEATURES)

    query = text(f"""
        SELECT {cols_sql}
        FROM staging_marts.ml_dataset_v1
        WHERE split_set = :split
          AND ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
    """)

    chunks = []
    total = 0
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(
            query, {"split": split_name}
        )
        while True:
            rows = result.fetchmany(CHUNK_SIZE)
            if not rows:
                break
            chunk = pd.DataFrame(rows, columns=cols_leer)
            chunks.append(chunk)
            total += len(chunk)
            print(f"    ...{total:,} filas", flush=True)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # Numéricas
    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")

    # Categóricas
    if encoders is None:
        encoders = {}
    for col in FEATURES_CATEGORICAS:
        if fit:
            le = LabelEncoder()
            vals = df[col].fillna("DESCONOCIDO").astype(str).tolist()
            le.fit(vals)
            df[col] = le.transform(vals)
            encoders[col] = le
        else:
            df[col] = encode_col(df[col], encoders[col])
        df[col] = df[col].astype("int32")

    # Target: log1p para estabilizar la distribución (hours_total puede tener cola larga)
    y_raw  = df[TARGET].astype("float32").values
    y      = np.log1p(y_raw)  # entrenamos en escala log, evaluamos en escala original

    X = df[ALL_FEATURES].copy()

    elapsed = time.time() - t0
    print(f"  '{split_name}' listo: {len(X):,} filas en {elapsed:.1f}s", flush=True)
    print(f"  hours_total — media: {y_raw.mean():.1f}h  mediana: {np.median(y_raw):.1f}h  max: {y_raw.max():.1f}h", flush=True)

    del df
    gc.collect()
    return X, y, y_raw, encoders


def entrenar(X_train, y_train, X_val, y_val):
    print("\nEntrenando LightGBM regresión...", flush=True)

    params = {
        "objective":         "regression",
        "metric":            ["rmse", "mae"],
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

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=callbacks,
        categorical_feature=[ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS],
    )

    print(f"\nMejor iteración: {model.best_iteration_}", flush=True)
    return model


def evaluar(model, X, y_log, y_raw, split_name):
    """Predice en escala log y evalúa en escala original (horas)."""
    y_pred_log = model.predict(X)
    y_pred     = np.expm1(y_pred_log)   # volver a horas reales
    y_pred     = np.clip(y_pred, 0, None)  # no puede ser negativo

    mae  = mean_absolute_error(y_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_raw, y_pred))
    r2   = r2_score(y_raw, y_pred)

    # Error en días (más interpretable para el negocio)
    mae_dias  = mae  / 24
    rmse_dias = rmse / 24

    # % de predicciones dentro de ±24h del valor real
    dentro_24h = np.mean(np.abs(y_pred - y_raw) <= 24) * 100
    dentro_48h = np.mean(np.abs(y_pred - y_raw) <= 48) * 100

    metricas = {
        "split":        split_name,
        "n":            len(y_raw),
        "mae_horas":    round(float(mae),       2),
        "rmse_horas":   round(float(rmse),      2),
        "mae_dias":     round(float(mae_dias),  2),
        "rmse_dias":    round(float(rmse_dias), 2),
        "r2":           round(float(r2),        4),
        "dentro_24h":   round(float(dentro_24h),2),
        "dentro_48h":   round(float(dentro_48h),2),
    }

    print(f"\n{'='*50}", flush=True)
    print(f"MÉTRICAS REGRESIÓN — {split_name.upper()}", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  MAE:           {mae:.2f} horas  ({mae_dias:.2f} días)", flush=True)
    print(f"  RMSE:          {rmse:.2f} horas  ({rmse_dias:.2f} días)", flush=True)
    print(f"  R²:            {r2:.4f}", flush=True)
    print(f"  Dentro ±24h:   {dentro_24h:.2f}%", flush=True)
    print(f"  Dentro ±48h:   {dentro_48h:.2f}%", flush=True)

    # Distribución de errores por cuartiles
    errores = np.abs(y_pred - y_raw)
    print(f"\n  Distribución del error absoluto:", flush=True)
    for pct in [25, 50, 75, 90, 95]:
        print(f"    p{pct}: {np.percentile(errores, pct):.1f}h", flush=True)

    return metricas


def guardar_feature_importance(model, path):
    fi = pd.DataFrame({
        "feature":          ALL_FEATURES,
        "importance_gain":  model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(path, index=False)
    print(f"\nFeature importance guardada en: {path}", flush=True)
    print(fi.to_string(index=False), flush=True)
    return fi


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("ENTRENAMIENTO LGBM — REGRESIÓN hours_total", flush=True)
    print("="*60, flush=True)
    print(f"Features: {len(ALL_FEATURES)} | Target: hours_total (escala log)", flush=True)

    engine = get_engine()

    # ── Cargar datos ──────────────────────────────────────────
    print("\n[1/5] Cargando datos...", flush=True)
    X_train, y_train, y_train_raw, encoders = cargar_split(engine, "train", fit=True)
    X_val,   y_val,   y_val_raw,   _        = cargar_split(engine, "val",   encoders=encoders)
    X_test,  y_test,  y_test_raw,  _        = cargar_split(engine, "test",  encoders=encoders)

    print(f"\nTamaños: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}", flush=True)

    # ── Entrenar ──────────────────────────────────────────────
    print("\n[2/5] Entrenando...", flush=True)
    t_train = time.time()
    model = entrenar(X_train, y_train, X_val, y_val)
    print(f"Entrenamiento: {(time.time()-t_train)/60:.1f} minutos", flush=True)

    del X_train, y_train, y_train_raw
    gc.collect()

    # ── Evaluar ───────────────────────────────────────────────
    print("\n[3/5] Evaluando...", flush=True)
    metricas_val  = evaluar(model, X_val,  y_val,  y_val_raw,  "val")
    metricas_test = evaluar(model, X_test, y_test, y_test_raw, "test")

    del X_val, y_val, y_val_raw
    del X_test, y_test, y_test_raw
    gc.collect()

    # ── Guardar ───────────────────────────────────────────────
    print("\n[4/5] Guardando outputs...", flush=True)

    model_path = OUTPUT_DIR / "model_lgbm_reg_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":    model,
            "encoders": encoders,
            "features": ALL_FEATURES,
            "version":  "reg_v1",
            "target":   "hours_total (log1p)",
        }, f)
    print(f"  Modelo guardado: {model_path}", flush=True)

    results_path = OUTPUT_DIR / "results_reg_v1.json"
    with open(results_path, "w") as f:
        json.dump({"val": metricas_val, "test": metricas_test}, f, indent=2)
    print(f"  Métricas guardadas: {results_path}", flush=True)

    fi_path = OUTPUT_DIR / "feature_importance_reg_v1.csv"
    guardar_feature_importance(model, fi_path)

    # ── Resumen ───────────────────────────────────────────────
    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("RESUMEN FINAL — REGRESIÓN", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Tiempo total:       {total:.1f} minutos", flush=True)
    print(f"  MAE val:            {metricas_val['mae_horas']}h ({metricas_val['mae_dias']} días)", flush=True)
    print(f"  MAE test:           {metricas_test['mae_horas']}h ({metricas_test['mae_dias']} días)", flush=True)
    print(f"  RMSE val:           {metricas_val['rmse_horas']}h", flush=True)
    print(f"  RMSE test:          {metricas_test['rmse_horas']}h", flush=True)
    print(f"  R² val:             {metricas_val['r2']}", flush=True)
    print(f"  R² test:            {metricas_test['r2']}", flush=True)
    print(f"  Dentro ±24h val:    {metricas_val['dentro_24h']}%", flush=True)
    print(f"  Dentro ±24h test:   {metricas_test['dentro_24h']}%", flush=True)
    print(f"\n  Outputs en: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

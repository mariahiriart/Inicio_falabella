"""
Inferencia — Modelo LightGBM v1.
Lee ml_dataset_v1, aplica el modelo entrenado y escribe las predicciones
a staging_marts.ml_predictions en RDS.

Uso:
    python predict.py

Output en RDS:
    staging_marts.ml_predictions
    Columnas:
        logistic_order_id  — ID de la orden
        year_month         — mes de la orden
        split_set          — train / val / test
        prob_disrupcion    — probabilidad predicha (0.0 a 1.0)
        pred_disrupcion    — predicción binaria (0 o 1, umbral 0.5)
        target_real        — valor real (0 o 1)
        acierto            — 1 si la predicción fue correcta, 0 si no
"""

import os
import gc
import time
import pickle
import warnings
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

MODEL_PATH = Path("/home/ec2-user/Inicio_falabella/ml_outputs/model_lgbm_v1.pkl")
CHUNK_SIZE = 300_000

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
# CREAR TABLA DE PREDICCIONES
# ──────────────────────────────────────────────

def crear_tabla_predicciones(engine):
    """Crea (o recrea) la tabla de predicciones en RDS."""
    ddl = """
        DROP TABLE IF EXISTS staging_marts.ml_predictions;
        CREATE TABLE staging_marts.ml_predictions (
            logistic_order_id  TEXT,
            year_month         TEXT,
            split_set          TEXT,
            service_category   TEXT,
            prob_disrupcion    NUMERIC(6,4),
            pred_disrupcion    SMALLINT,
            target_real        SMALLINT,
            acierto            SMALLINT
        );
        CREATE INDEX idx_pred_order  ON staging_marts.ml_predictions (logistic_order_id);
        CREATE INDEX idx_pred_month  ON staging_marts.ml_predictions (year_month);
        CREATE INDEX idx_pred_split  ON staging_marts.ml_predictions (split_set);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("Tabla staging_marts.ml_predictions creada.", flush=True)


# ──────────────────────────────────────────────
# PREPROCESAMIENTO
# ──────────────────────────────────────────────

def preprocesar(df, encoders):
    """Aplica el mismo preprocesamiento que en el entrenamiento."""
    df = df.copy()
    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")
    for col in FEATURES_CATEGORICAS:
        le = encoders[col]
        df[col] = df[col].fillna("DESCONOCIDO").astype(str)
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "DESCONOCIDO")
        df[col] = le.transform(df[col]).astype("int32")
    return df[ALL_FEATURES]


# ──────────────────────────────────────────────
# INFERENCIA EN CHUNKS
# ──────────────────────────────────────────────

def predecir_y_guardar(engine, model, encoders, umbral=0.5):
    """
    Lee ml_dataset_v1 en chunks, predice y escribe a ml_predictions.
    Procesa todo el dataset (train + val + test).
    """
    cols_leer = (
        ["logistic_order_id", "year_month", "split_set", "service_category", "target_binario"]
        + ALL_FEATURES
    )
    cols_sql = ", ".join(cols_leer)
    query = text(f"SELECT {cols_sql} FROM staging_marts.ml_dataset_v1")

    total_procesado = 0
    total_correctos = 0
    t0 = time.time()

    with engine.connect() as conn_read:
        result = conn_read.execution_options(stream_results=True).execute(query)

        while True:
            rows = result.fetchmany(CHUNK_SIZE)
            if not rows:
                break

            df = pd.DataFrame(rows, columns=cols_leer)

            # Guardar columnas de contexto antes de preprocesar
            ids        = df["logistic_order_id"].values
            meses      = df["year_month"].values
            splits     = df["split_set"].values
            categorias = df["service_category"].values
            targets    = df["target_binario"].astype(int).values

            # Preprocesar y predecir
            X = preprocesar(df, encoders)
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= umbral).astype(int)
            aciertos = (preds == targets).astype(int)

            # Armar dataframe de resultados
            resultado = pd.DataFrame({
                "logistic_order_id": ids,
                "year_month":        meses,
                "split_set":         splits,
                "service_category":  categorias,
                "prob_disrupcion":   np.round(probs, 4),
                "pred_disrupcion":   preds,
                "target_real":       targets,
                "acierto":           aciertos,
            })

            # Escribir a RDS
            resultado.to_sql(
                "ml_predictions",
                engine,
                schema="staging_marts",
                if_exists="append",
                index=False,
                method="multi",
                chunksize=10_000,
            )

            total_procesado += len(df)
            total_correctos += aciertos.sum()
            acc_parcial = aciertos.mean()

            del df, X, resultado
            gc.collect()

            elapsed = time.time() - t0
            print(
                f"  {total_procesado:,} filas procesadas "
                f"({elapsed:.0f}s) — acc chunk: {acc_parcial:.3f}",
                flush=True,
            )

    acc_global = total_correctos / total_procesado
    print(f"\nTotal procesado: {total_procesado:,} filas", flush=True)
    print(f"Accuracy global: {acc_global:.4f}", flush=True)
    return total_procesado, acc_global


# ──────────────────────────────────────────────
# VALIDACIÓN FINAL
# ──────────────────────────────────────────────

def validar_predicciones(engine):
    """Muestra métricas de las predicciones guardadas por split y por categoría."""

    print("\n" + "="*60, flush=True)
    print("VALIDACIÓN DE PREDICCIONES EN RDS", flush=True)
    print("="*60, flush=True)

    # Por split
    q_split = text("""
        SELECT
            split_set,
            COUNT(*)                                         AS n,
            ROUND(AVG(prob_disrupcion::numeric), 4)          AS prob_media,
            ROUND(100.0 * SUM(pred_disrupcion) / COUNT(*), 2) AS pct_pred_disrupted,
            ROUND(100.0 * SUM(target_real)     / COUNT(*), 2) AS pct_real_disrupted,
            ROUND(100.0 * SUM(acierto)         / COUNT(*), 2) AS accuracy
        FROM staging_marts.ml_predictions
        GROUP BY split_set
        ORDER BY CASE split_set WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END
    """)
    with engine.connect() as conn:
        rows = conn.execute(q_split).fetchall()

    print("\nPor split:", flush=True)
    print(f"  {'split':<8} {'n':>12} {'prob_media':>11} {'pred_%':>8} {'real_%':>8} {'accuracy':>9}", flush=True)
    print("  " + "-"*58, flush=True)
    for r in rows:
        print(f"  {r[0]:<8} {r[1]:>12,} {r[2]:>11.4f} {r[3]:>8.2f}% {r[4]:>8.2f}% {r[5]:>8.2f}%", flush=True)

    # Por service_category
    q_cat = text("""
        SELECT
            service_category,
            COUNT(*)                                          AS n,
            ROUND(100.0 * SUM(target_real)     / COUNT(*), 2) AS pct_real_disrupted,
            ROUND(100.0 * SUM(pred_disrupcion) / COUNT(*), 2) AS pct_pred_disrupted,
            ROUND(100.0 * SUM(acierto)         / COUNT(*), 2) AS accuracy
        FROM staging_marts.ml_predictions
        GROUP BY service_category
        ORDER BY n DESC
    """)
    with engine.connect() as conn:
        rows_cat = conn.execute(q_cat).fetchall()

    print("\nPor service_category:", flush=True)
    print(f"  {'categoría':<15} {'n':>10} {'real_%':>8} {'pred_%':>8} {'accuracy':>9}", flush=True)
    print("  " + "-"*52, flush=True)
    for r in rows_cat:
        cat = str(r[0]) if r[0] else "(vacío)"
        print(f"  {cat:<15} {r[1]:>10,} {r[2]:>8.2f}% {r[3]:>8.2f}% {r[4]:>8.2f}%", flush=True)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("INFERENCIA — MODELO LGBM v1", flush=True)
    print("="*60, flush=True)

    # Cargar modelo
    print(f"\nCargando modelo desde {MODEL_PATH}...", flush=True)
    with open(MODEL_PATH, "rb") as f:
        artefacto = pickle.load(f)
    model    = artefacto["model"]
    encoders = artefacto["encoders"]
    print(f"  Features: {artefacto['features']}", flush=True)

    engine = get_engine()

    # Crear tabla destino
    print("\nCreando tabla de predicciones...", flush=True)
    crear_tabla_predicciones(engine)

    # Predecir y guardar
    print(f"\nProcesando dataset en chunks de {CHUNK_SIZE:,}...", flush=True)
    total, acc = predecir_y_guardar(engine, model, encoders, umbral=0.5)

    # Validar
    validar_predicciones(engine)

    total_min = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETADO en {total_min:.1f} minutos", flush=True)
    print(f"  {total:,} predicciones escritas en staging_marts.ml_predictions", flush=True)
    print(f"  Accuracy global: {acc:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

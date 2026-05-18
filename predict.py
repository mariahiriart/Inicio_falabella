"""
Inferencia — Modelo LightGBM v1.
Lee ml_dataset_v1, aplica el modelo y escribe predicciones
a staging_marts.ml_predictions en RDS.

Uso:
    python predict.py
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


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def crear_tabla_predicciones(engine):
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
        CREATE INDEX idx_pred_order ON staging_marts.ml_predictions (logistic_order_id);
        CREATE INDEX idx_pred_month ON staging_marts.ml_predictions (year_month);
        CREATE INDEX idx_pred_split ON staging_marts.ml_predictions (split_set);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("Tabla staging_marts.ml_predictions creada.", flush=True)


def encode_col(series, le):
    """Encodea una columna usando LabelEncoder de forma segura."""
    known = set(str(c) for c in le.classes_)
    values = series.fillna("DESCONOCIDO").astype(str).tolist()
    clean  = [v if v in known else "DESCONOCIDO" for v in values]
    return le.transform(clean)


def preprocesar(df, encoders):
    df = df.copy()
    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")
    for col in FEATURES_CATEGORICAS:
        df[col] = encode_col(df[col], encoders[col]).astype("int32")
    return df[ALL_FEATURES]


def predecir_y_guardar(engine, model, encoders, umbral=0.5):
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

            ids        = df["logistic_order_id"].values
            meses      = df["year_month"].values
            splits     = df["split_set"].values
            categorias = df["service_category"].values
            targets    = df["target_binario"].astype(int).values

            X        = preprocesar(df, encoders)
            probs    = model.predict_proba(X)[:, 1]
            preds    = (probs >= umbral).astype(int)
            aciertos = (preds == targets).astype(int)

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
            total_correctos += int(aciertos.sum())

            del df, X, resultado
            gc.collect()

            elapsed = time.time() - t0
            print(f"  {total_procesado:,} filas ({elapsed:.0f}s) — acc: {aciertos.mean():.4f}", flush=True)

    acc_global = total_correctos / total_procesado
    print(f"\nTotal: {total_procesado:,} | Accuracy global: {acc_global:.4f}", flush=True)
    return total_procesado, acc_global


def validar_predicciones(engine):
    print("\n" + "="*60, flush=True)
    print("VALIDACIÓN EN RDS", flush=True)
    print("="*60, flush=True)

    q = text("""
        SELECT split_set, COUNT(*) AS n,
            ROUND(AVG(prob_disrupcion::numeric), 4) AS prob_media,
            ROUND(100.0 * SUM(pred_disrupcion) / COUNT(*), 2) AS pct_pred,
            ROUND(100.0 * SUM(target_real)     / COUNT(*), 2) AS pct_real,
            ROUND(100.0 * SUM(acierto)         / COUNT(*), 2) AS accuracy
        FROM staging_marts.ml_predictions
        GROUP BY split_set
        ORDER BY CASE split_set WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END
    """)
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()

    print(f"\n{'split':<8} {'n':>12} {'prob':>8} {'pred_%':>8} {'real_%':>8} {'acc':>7}", flush=True)
    print("-"*56, flush=True)
    for r in rows:
        print(f"{r[0]:<8} {r[1]:>12,} {float(r[2]):>8.4f} {float(r[3]):>8.2f}% {float(r[4]):>8.2f}% {float(r[5]):>6.2f}%", flush=True)

    q2 = text("""
        SELECT COALESCE(service_category,'(vacío)') AS cat,
            COUNT(*) AS n,
            ROUND(100.0 * SUM(target_real) / COUNT(*), 2) AS pct_real,
            ROUND(100.0 * SUM(acierto)     / COUNT(*), 2) AS accuracy
        FROM staging_marts.ml_predictions
        GROUP BY service_category ORDER BY n DESC
    """)
    with engine.connect() as conn:
        rows2 = conn.execute(q2).fetchall()

    print(f"\n{'categoría':<15} {'n':>10} {'real_%':>8} {'acc':>7}", flush=True)
    print("-"*44, flush=True)
    for r in rows2:
        print(f"{str(r[0]):<15} {r[1]:>10,} {float(r[2]):>8.2f}% {float(r[3]):>6.2f}%", flush=True)


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("INFERENCIA — MODELO LGBM v1", flush=True)
    print("="*60, flush=True)

    print(f"\nCargando modelo desde {MODEL_PATH}...", flush=True)
    with open(MODEL_PATH, "rb") as f:
        art = pickle.load(f)
    model    = art["model"]
    encoders = art["encoders"]
    print(f"  Features: {len(art['features'])}", flush=True)

    engine = get_engine()

    print("\nCreando tabla...", flush=True)
    crear_tabla_predicciones(engine)

    print(f"\nProcesando en chunks de {CHUNK_SIZE:,}...", flush=True)
    total, acc = predecir_y_guardar(engine, model, encoders)

    validar_predicciones(engine)

    print(f"\nCOMPLETADO en {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"{total:,} predicciones en staging_marts.ml_predictions", flush=True)


if __name__ == "__main__":
    main()"""
Inferencia — Modelo LightGBM v1.
Lee ml_dataset_v1, aplica el modelo y escribe predicciones
a staging_marts.ml_predictions en RDS.

Uso:
    python predict.py
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


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def crear_tabla_predicciones(engine):
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
        CREATE INDEX idx_pred_order ON staging_marts.ml_predictions (logistic_order_id);
        CREATE INDEX idx_pred_month ON staging_marts.ml_predictions (year_month);
        CREATE INDEX idx_pred_split ON staging_marts.ml_predictions (split_set);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("Tabla staging_marts.ml_predictions creada.", flush=True)


def encode_col(series, le):
    """Encodea una columna usando LabelEncoder de forma segura."""
    known = set(str(c) for c in le.classes_)
    values = series.fillna("DESCONOCIDO").astype(str).tolist()
    clean  = [v if v in known else "DESCONOCIDO" for v in values]
    return le.transform(clean)


def preprocesar(df, encoders):
    df = df.copy()
    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")
    for col in FEATURES_CATEGORICAS:
        df[col] = encode_col(df[col], encoders[col]).astype("int32")
    return df[ALL_FEATURES]


def predecir_y_guardar(engine, model, encoders, umbral=0.5):
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

            ids        = df["logistic_order_id"].values
            meses      = df["year_month"].values
            splits     = df["split_set"].values
            categorias = df["service_category"].values
            targets    = df["target_binario"].astype(int).values

            X        = preprocesar(df, encoders)
            probs    = model.predict_proba(X)[:, 1]
            preds    = (probs >= umbral).astype(int)
            aciertos = (preds == targets).astype(int)

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
            total_correctos += int(aciertos.sum())

            del df, X, resultado
            gc.collect()

            elapsed = time.time() - t0
            print(f"  {total_procesado:,} filas ({elapsed:.0f}s) — acc: {aciertos.mean():.4f}", flush=True)

    acc_global = total_correctos / total_procesado
    print(f"\nTotal: {total_procesado:,} | Accuracy global: {acc_global:.4f}", flush=True)
    return total_procesado, acc_global


def validar_predicciones(engine):
    print("\n" + "="*60, flush=True)
    print("VALIDACIÓN EN RDS", flush=True)
    print("="*60, flush=True)

    q = text("""
        SELECT split_set, COUNT(*) AS n,
            ROUND(AVG(prob_disrupcion::numeric), 4) AS prob_media,
            ROUND(100.0 * SUM(pred_disrupcion) / COUNT(*), 2) AS pct_pred,
            ROUND(100.0 * SUM(target_real)     / COUNT(*), 2) AS pct_real,
            ROUND(100.0 * SUM(acierto)         / COUNT(*), 2) AS accuracy
        FROM staging_marts.ml_predictions
        GROUP BY split_set
        ORDER BY CASE split_set WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END
    """)
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()

    print(f"\n{'split':<8} {'n':>12} {'prob':>8} {'pred_%':>8} {'real_%':>8} {'acc':>7}", flush=True)
    print("-"*56, flush=True)
    for r in rows:
        print(f"{r[0]:<8} {r[1]:>12,} {float(r[2]):>8.4f} {float(r[3]):>8.2f}% {float(r[4]):>8.2f}% {float(r[5]):>6.2f}%", flush=True)

    q2 = text("""
        SELECT COALESCE(service_category,'(vacío)') AS cat,
            COUNT(*) AS n,
            ROUND(100.0 * SUM(target_real) / COUNT(*), 2) AS pct_real,
            ROUND(100.0 * SUM(acierto)     / COUNT(*), 2) AS accuracy
        FROM staging_marts.ml_predictions
        GROUP BY service_category ORDER BY n DESC
    """)
    with engine.connect() as conn:
        rows2 = conn.execute(q2).fetchall()

    print(f"\n{'categoría':<15} {'n':>10} {'real_%':>8} {'acc':>7}", flush=True)
    print("-"*44, flush=True)
    for r in rows2:
        print(f"{str(r[0]):<15} {r[1]:>10,} {float(r[2]):>8.2f}% {float(r[3]):>6.2f}%", flush=True)


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("INFERENCIA — MODELO LGBM v1", flush=True)
    print("="*60, flush=True)

    print(f"\nCargando modelo desde {MODEL_PATH}...", flush=True)
    with open(MODEL_PATH, "rb") as f:
        art = pickle.load(f)
    model    = art["model"]
    encoders = art["encoders"]
    print(f"  Features: {len(art['features'])}", flush=True)

    engine = get_engine()

    print("\nCreando tabla...", flush=True)
    crear_tabla_predicciones(engine)

    print(f"\nProcesando en chunks de {CHUNK_SIZE:,}...", flush=True)
    total, acc = predecir_y_guardar(engine, model, encoders)

    validar_predicciones(engine)

    print(f"\nCOMPLETADO en {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"{total:,} predicciones en staging_marts.ml_predictions", flush=True)


if __name__ == "__main__":
    main()
"""
Regresión v2 — 4 combinaciones para entender qué aprende el modelo.

Combinaciones:
  reg_v1  : target=hours_total,  con sla_horas_prometidas  (baseline ya entrenado)
  reg_v2a : target=desvio_sla,   con sla_horas_prometidas
  reg_v2b : target=desvio_sla,   sin sla_horas_prometidas
  reg_v2c : target=hours_total,  sin sla_horas_prometidas

Uso:
    python train_regression_v2.py

Outputs en ml_outputs/:
    model_lgbm_reg_v2a.pkl / v2b.pkl / v2c.pkl
    results_reg_v2_comparacion.json
    feature_importance_reg_v2a/b/c.csv
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

FEATURES_BASE = [
    "is_click_and_collect",
    "is_high_season",
    "has_insurance",
    "total_items",
    "distinct_skus",
    "dia_semana_creacion",
    "hora_creacion",
    "dia_mes_creacion",
]

FEATURES_CATEGORICAS = [
    "service_category",
    "source_order_type",
    "seller_id",
    "created_by",
]

# Con SLA = 13 features, sin SLA = 12 features
FEATURES_CON_SLA = FEATURES_BASE + ["sla_horas_prometidas"] + FEATURES_CATEGORICAS
FEATURES_SIN_SLA = FEATURES_BASE + FEATURES_CATEGORICAS

CHUNK_SIZE = 400_000

# Resultados de reg_v1 para comparación final
REG_V1 = {
    "nombre":    "reg_v1",
    "target":    "hours_total",
    "con_sla":   True,
    "mae_val":   None,   # se carga desde archivo si existe
    "rmse_val":  None,
    "r2_val":    None,
    "mae_test":  None,
    "rmse_test": None,
    "r2_test":   None,
}


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def encode_col(series, le):
    known = set(str(c) for c in le.classes_)
    clean = [str(v) if str(v) in known else "DESCONOCIDO" for v in list(series)]
    return le.transform(clean)


def cargar_split(engine, split_name, con_sla, encoders=None, fit=False):
    """
    Carga un split con ciclo completo.
    Calcula desvio_sla = hours_total - sla_horas_prometidas.
    Devuelve features según con_sla, y ambos targets.
    """
    features = FEATURES_CON_SLA if con_sla else FEATURES_SIN_SLA
    # Siempre leer sla_horas_prometidas para calcular desvio, aunque no sea feature
    cols_extra = [
        "logistic_order_id", "year_month", "split_set",
        "target_horas_totales", "sla_horas_prometidas"
    ]
    # Evitar duplicados si sla_horas_prometidas ya está en features
    cols_leer = cols_extra + [f for f in features if f != "sla_horas_prometidas"]

    cols_sql = ", ".join([
        "logistic_order_id", "year_month", "split_set",
        "target_horas_totales", "sla_horas_prometidas",
    ] + [f for f in features if f != "sla_horas_prometidas"])

    query = text(f"""
        SELECT {cols_sql}
        FROM staging_marts.ml_dataset_v1
        WHERE split_set = :split
          AND ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
    """)

    print(f"  Cargando '{split_name}' ({'con' if con_sla else 'sin'} SLA)...", flush=True)
    t0 = time.time()
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

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # Calcular desvío: positivo = tardó más de lo prometido (disrupción de tiempo)
    df["desvio_sla"] = df["target_horas_totales"] - df["sla_horas_prometidas"]

    # Targets en escala original
    y_hours_raw  = df["target_horas_totales"].astype("float32").values
    y_desvio_raw = df["desvio_sla"].astype("float32").values

    # Preprocesar numéricas
    num_features = [f for f in features if f not in FEATURES_CATEGORICAS]
    for col in num_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")

    # Preprocesar categóricas
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

    X = df[features].copy()
    elapsed = time.time() - t0
    print(f"    {len(X):,} filas en {elapsed:.1f}s", flush=True)

    del df
    gc.collect()
    return X, y_hours_raw, y_desvio_raw, encoders


def entrenar(X_train, y_train_log, X_val, y_val_log, nombre):
    """y_train_log y y_val_log ya están en escala log1p."""
    print(f"\n  Entrenando {nombre}...", flush=True)

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

    features = list(X_train.columns)
    cat_idx  = [features.index(c) for c in FEATURES_CATEGORICAS if c in features]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
        categorical_feature=cat_idx,
    )
    print(f"  Mejor iteración: {model.best_iteration_}", flush=True)
    return model


def evaluar(model, X, y_raw, split_name, es_desvio=False):
    """
    Predice y evalúa en escala original.
    Para desvío: el modelo predice log1p(desvío + offset) → volvemos a desvío en horas.
    Para hours_total: predice log1p(hours_total) → volvemos a horas.
    """
    y_pred_log = model.predict(X)
    y_pred     = np.expm1(y_pred_log)

    if es_desvio:
        # Para desvío, restamos el offset que usamos al entrenar
        y_pred = y_pred - 200.0
    else:
        y_pred = np.clip(y_pred, 0, None)

    mae  = mean_absolute_error(y_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_raw, y_pred))
    r2   = r2_score(y_raw, y_pred)

    dentro_24h = np.mean(np.abs(y_pred - y_raw) <= 24) * 100
    dentro_48h = np.mean(np.abs(y_pred - y_raw) <= 48) * 100

    print(f"    [{split_name}] MAE={mae:.2f}h ({mae/24:.2f}d)  RMSE={rmse:.2f}h  R²={r2:.4f}  ±24h={dentro_24h:.1f}%", flush=True)

    return {
        "split":       split_name,
        "n":           len(y_raw),
        "mae_horas":   round(float(mae),  2),
        "rmse_horas":  round(float(rmse), 2),
        "mae_dias":    round(float(mae/24), 2),
        "r2":          round(float(r2),   4),
        "dentro_24h":  round(float(dentro_24h), 2),
        "dentro_48h":  round(float(dentro_48h), 2),
    }


def guardar_fi(model, features, path):
    fi = pd.DataFrame({
        "feature":          features,
        "importance_gain":  model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(path, index=False)
    return fi


def imprimir_comparacion(resultados):
    print(f"\n{'='*70}", flush=True)
    print("COMPARACIÓN FINAL DE LOS 4 MODELOS", flush=True)
    print(f"{'='*70}", flush=True)
    header = f"{'Modelo':<12} {'Target':<15} {'SLA feat':>8} {'MAE val':>9} {'R² val':>8} {'MAE test':>9} {'R² test':>8}"
    print(header, flush=True)
    print("-"*70, flush=True)
    for r in resultados:
        sla_str = "Sí" if r["con_sla"] else "No"
        mae_v   = f"{r['mae_val']:.2f}h"   if r["mae_val"]  else "—"
        r2_v    = f"{r['r2_val']:.4f}"     if r["r2_val"]   else "—"
        mae_t   = f"{r['mae_test']:.2f}h"  if r["mae_test"] else "—"
        r2_t    = f"{r['r2_test']:.4f}"    if r["r2_test"]  else "—"
        print(f"  {r['nombre']:<10} {r['target']:<15} {sla_str:>8} {mae_v:>9} {r2_v:>8} {mae_t:>9} {r2_t:>8}", flush=True)

    print(f"\n  Interpretación:", flush=True)
    if len(resultados) >= 3:
        r2a = next((r for r in resultados if r["nombre"] == "reg_v2a"), None)
        r2b = next((r for r in resultados if r["nombre"] == "reg_v2b"), None)
        r2c = next((r for r in resultados if r["nombre"] == "reg_v2c"), None)
        rv1 = next((r for r in resultados if r["nombre"] == "reg_v1"), None)
        if r2a and r2b and r2a["mae_val"] and r2b["mae_val"]:
            diff = r2a["mae_val"] - r2b["mae_val"]
            print(f"  → v2a vs v2b (impacto de SLA como feature en desvío): {diff:+.2f}h", flush=True)
        if rv1 and r2c and rv1["mae_val"] and r2c["mae_val"]:
            diff = r2c["mae_val"] - rv1["mae_val"]
            print(f"  → v2c vs v1  (impacto de quitar SLA en hours_total):  {diff:+.2f}h", flush=True)
        if rv1 and r2a and rv1["mae_val"] and r2a["mae_val"]:
            diff = r2a["mae_val"] - rv1["mae_val"]
            print(f"  → v2a vs v1  (cambiar target a desvío, con SLA):       {diff:+.2f}h", flush=True)


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("REGRESIÓN v2 — 4 COMBINACIONES", flush=True)
    print("="*60, flush=True)

    engine = get_engine()
    resultados = []

    # Intentar cargar resultados de reg_v1 si existen
    reg_v1_path = OUTPUT_DIR / "results_reg_v1.json"
    if reg_v1_path.exists():
        with open(reg_v1_path) as f:
            rv1 = json.load(f)
        REG_V1.update({
            "mae_val":   rv1["val"]["mae_horas"],
            "rmse_val":  rv1["val"]["rmse_horas"],
            "r2_val":    rv1["val"]["r2"],
            "mae_test":  rv1["test"]["mae_horas"],
            "rmse_test": rv1["test"]["rmse_horas"],
            "r2_test":   rv1["test"]["r2"],
        })
        print(f"reg_v1 cargado: MAE val={REG_V1['mae_val']}h  R² val={REG_V1['r2_val']}", flush=True)
    resultados.append(REG_V1)

    # ── EXPERIMENTOS ─────────────────────────────────────────
    experimentos = [
        {"nombre": "reg_v2a", "target": "desvio_sla",   "con_sla": True},
        {"nombre": "reg_v2b", "target": "desvio_sla",   "con_sla": False},
        {"nombre": "reg_v2c", "target": "hours_total",  "con_sla": False},
    ]

    for exp in experimentos:
        nombre   = exp["nombre"]
        es_desvio = exp["target"] == "desvio_sla"
        con_sla  = exp["con_sla"]

        print(f"\n{'='*60}", flush=True)
        print(f"EXPERIMENTO: {nombre} | target={exp['target']} | SLA={'Sí' if con_sla else 'No'}", flush=True)
        print(f"{'='*60}", flush=True)

        # Cargar
        X_train, y_h_train, y_d_train, encoders = cargar_split(engine, "train", con_sla, fit=True)
        X_val,   y_h_val,   y_d_val,   _        = cargar_split(engine, "val",   con_sla, encoders=encoders)
        X_test,  y_h_test,  y_d_test,  _        = cargar_split(engine, "test",  con_sla, encoders=encoders)

        # Seleccionar target
        if es_desvio:
            # Offset +200 para que todos los valores sean positivos antes de log1p
            # (desvio puede ser muy negativo si la orden llegó antes de lo prometido)
            OFFSET = 200.0
            y_train_raw = y_d_train
            y_val_raw   = y_d_val
            y_test_raw  = y_d_test
            y_train_log = np.log1p(y_d_train + OFFSET)
            y_val_log   = np.log1p(y_d_val   + OFFSET)
        else:
            OFFSET = 0.0
            y_train_raw = y_h_train
            y_val_raw   = y_h_val
            y_test_raw  = y_h_test
            y_train_log = np.log1p(y_h_train)
            y_val_log   = np.log1p(y_h_val)

        # Entrenar
        t_exp = time.time()
        model = entrenar(X_train, y_train_log, X_val, y_val_log, nombre)
        print(f"  Tiempo entrenamiento: {(time.time()-t_exp)/60:.1f} min", flush=True)

        del X_train, y_h_train, y_d_train, y_train_log
        gc.collect()

        # Evaluar
        print(f"  Evaluando {nombre}...", flush=True)
        m_val  = evaluar(model, X_val,  y_val_raw,  "val",  es_desvio)
        m_test = evaluar(model, X_test, y_test_raw, "test", es_desvio)

        del X_val, y_h_val, y_d_val, y_val_log
        del X_test, y_h_test, y_d_test
        gc.collect()

        # Guardar modelo
        model_path = OUTPUT_DIR / f"model_lgbm_{nombre}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":    model,
                "encoders": encoders,
                "features": list(X_train.columns) if False else (FEATURES_CON_SLA if con_sla else FEATURES_SIN_SLA),
                "version":  nombre,
                "target":   exp["target"],
                "con_sla":  con_sla,
            }, f)

        # Guardar feature importance
        fi_path = OUTPUT_DIR / f"feature_importance_{nombre}.csv"
        features_used = FEATURES_CON_SLA if con_sla else FEATURES_SIN_SLA
        guardar_fi(model, features_used, fi_path)

        # Guardar métricas
        results_path = OUTPUT_DIR / f"results_{nombre}.json"
        with open(results_path, "w") as f:
            json.dump({"val": m_val, "test": m_test}, f, indent=2)

        resultados.append({
            "nombre":   nombre,
            "target":   exp["target"],
            "con_sla":  con_sla,
            "mae_val":  m_val["mae_horas"],
            "rmse_val": m_val["rmse_horas"],
            "r2_val":   m_val["r2"],
            "mae_test": m_test["mae_horas"],
            "rmse_test":m_test["rmse_horas"],
            "r2_test":  m_test["r2"],
        })

        imprimir_comparacion(resultados)

    # Guardar comparación final
    comp_path = OUTPUT_DIR / "results_reg_v2_comparacion.json"
    with open(comp_path, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\nComparación guardada en: {comp_path}", flush=True)

    total = (time.time() - t_inicio) / 60
    print(f"\nTIEMPO TOTAL: {total:.1f} minutos", flush=True)


if __name__ == "__main__":
    main()

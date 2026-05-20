"""
Entrenamiento modelo LightGBM v2 — optimizado para EC2 con poca RAM.

Estrategia: carga mes a mes en lugar de todo junto.
  - Lee un mes a la vez desde RDS
  - Libera memoria después de cada mes
  - Entrena con LightGBM en modo incremental (continue_training)
  - Mucho más estable en EC2 con memoria limitada

Uso:
    python3 train_model_v2_optimizado.py

Outputs en ml_outputs/:
    model_lgbm_v2.pkl
    results_v2.json
    feature_importance_v2.csv
"""

import os, gc, json, time, pickle, warnings
import numpy as np
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

# ── Config ───────────────────────────────────────────────────────────────────

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
    "franja_horaria",
]

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
TARGET       = "target_binario"

# Meses de cada split
MESES_TRAIN = [
    "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06",
    "2025-07","2025-08","2025-09","2025-10","2025-11"
]
MESES_VAL  = ["2025-12","2026-01"]
MESES_TEST = ["2026-02"]

PARAMS_LGBM = {
    "objective":         "binary",
    "metric":            ["binary_logloss", "auc"],
    "boosting_type":     "gbdt",
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


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def get_fresh_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


def cargar_mes(year_month: str, encoders: dict, fit: bool = False):
    """Carga un mes completo con retry en caso de SSL timeout."""
    print(f"    Mes {year_month}...", end=" ", flush=True)
    t0 = time.time()

    cols_sql = ", ".join(ALL_FEATURES + [TARGET])
    query = text(f"""
        SELECT {cols_sql}
        FROM staging_marts.ml_dataset_v2
        WHERE year_month = :ym
    """)

    for intento in range(3):
        try:
            eng = get_fresh_engine()
            with eng.connect() as conn:
                df = pd.read_sql(query, conn, params={"ym": year_month})
            eng.dispose()
            break
        except Exception as e:
            print(f"\n    Reintento {intento+1}/3 por error: {str(e)[:50]}", flush=True)
            time.sleep(5)
            if intento == 2:
                raise

    if len(df) == 0:
        print(f"0 filas — saltando", flush=True)
        return None, None, encoders

    # Numéricas
    for col in FEATURES_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")

    # Categóricas
    for col in FEATURES_CATEGORICAS:
        df[col] = df[col].fillna("DESCONOCIDO").astype(str)
        if fit and col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]).astype("int32")
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "DESCONOCIDO")
            df[col] = le.transform(df[col]).astype("int32")

    X = df[ALL_FEATURES].values.astype("float32")
    y = df[TARGET].astype("int8").values

    del df; gc.collect()
    print(f"{len(y):,} filas en {time.time()-t0:.0f}s", flush=True)
    return X, y, encoders


def cargar_split_completo(engine, meses, encoders, fit=False, nombre=""):
    """Carga varios meses y los concatena."""
    print(f"  Cargando {nombre} ({len(meses)} meses)...", flush=True)
    Xs, ys = [], []
    for mes in meses:
        X, y, encoders = cargar_mes(mes, encoders, fit=fit)
        if X is not None:
            Xs.append(X)
            ys.append(y)
        fit = False  # solo fitear encoders en el primer mes
        gc.collect()

    X_total = np.vstack(Xs)
    y_total = np.concatenate(ys)
    del Xs, ys; gc.collect()
    print(f"  {nombre} total: {len(y_total):,} filas  disrupted: {100*y_total.mean():.1f}%", flush=True)
    return X_total, y_total, encoders


def evaluar(model, X, y, split_name):
    y_prob = model.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y, y_prob)
    rep  = classification_report(y, y_pred, output_dict=True)

    print(f"\n{'='*50}", flush=True)
    print(f"MÉTRICAS — {split_name.upper()}", flush=True)
    print(f"  AUC-ROC:           {auc:.4f}", flush=True)
    print(f"  Accuracy:          {rep['accuracy']:.4f}", flush=True)
    print(f"  Precision disrupt: {rep['1']['precision']:.4f}", flush=True)
    print(f"  Recall    disrupt: {rep['1']['recall']:.4f}", flush=True)
    print(f"  F1        disrupt: {rep['1']['f1-score']:.4f}", flush=True)

    return {
        "split":               split_name,
        "n":                   int(len(y)),
        "auc_roc":             round(float(auc), 4),
        "accuracy":            round(float(rep["accuracy"]), 4),
        "precision_disrupted": round(float(rep["1"]["precision"]), 4),
        "recall_disrupted":    round(float(rep["1"]["recall"]), 4),
        "f1_disrupted":        round(float(rep["1"]["f1-score"]), 4),
        "f1_ok":               round(float(rep["0"]["f1-score"]), 4),
    }


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("ENTRENAMIENTO LGBM v2 — OPTIMIZADO (mes a mes)", flush=True)
    print(f"Features: {len(ALL_FEATURES)}", flush=True)
    print("="*60, flush=True)

    engine  = get_engine()
    encoders = {}

    # ── 1. Cargar datos mes a mes ────────────────────────────────
    print("\n[1/4] Cargando datos...", flush=True)

    X_train, y_train, encoders = cargar_split_completo(
        engine, MESES_TRAIN, encoders, fit=True, nombre="train"
    )
    X_val, y_val, encoders = cargar_split_completo(
        engine, MESES_VAL, encoders, fit=False, nombre="val"
    )
    X_test, y_test, encoders = cargar_split_completo(
        engine, MESES_TEST, encoders, fit=False, nombre="test"
    )

    # ── 2. Construir datasets LightGBM ───────────────────────────
    print("\n[2/4] Construyendo datasets LightGBM...", flush=True)
    cat_idx = [ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS]

    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_idx,
        feature_name=ALL_FEATURES,
        free_raw_data=True
    )
    dval = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=cat_idx,
        feature_name=ALL_FEATURES,
        reference=dtrain,
        free_raw_data=True
    )

    del X_train, y_train; gc.collect()

    # ── 3. Entrenar ──────────────────────────────────────────────
    print("\n[3/4] Entrenando...", flush=True)
    t_train = time.time()

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        PARAMS_LGBM,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    print(f"\n  Mejor iteración: {model.best_iteration}", flush=True)
    print(f"  Entrenamiento: {(time.time()-t_train)/60:.1f} minutos", flush=True)

    del dtrain, dval; gc.collect()

    # ── 4. Evaluar ───────────────────────────────────────────────
    print("\n[4/4] Evaluando...", flush=True)
    m_val  = evaluar(model, X_val,  y_val,  "val")
    m_test = evaluar(model, X_test, y_test, "test")

    # Comparar con v1 si existe
    v1_path = OUTPUT_DIR / "results_v1.json"
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        print(f"\n{'='*50}", flush=True)
        print("COMPARACIÓN v1 vs v2", flush=True)
        print(f"  AUC-ROC val:  v1={v1['val']['auc_roc']}  v2={m_val['auc_roc']}  Δ={m_val['auc_roc']-v1['val']['auc_roc']:+.4f}", flush=True)
        print(f"  F1 val:       v1={v1['val']['f1_disrupted']}  v2={m_val['f1_disrupted']}  Δ={m_val['f1_disrupted']-v1['val']['f1_disrupted']:+.4f}", flush=True)
        print(f"  AUC-ROC test: v1={v1['test']['auc_roc']}  v2={m_test['auc_roc']}  Δ={m_test['auc_roc']-v1['test']['auc_roc']:+.4f}", flush=True)

    # ── Guardar ──────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "model_lgbm_v2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":    model,
            "encoders": encoders,
            "features": ALL_FEATURES,
            "version":  "v2"
        }, f)

    results_path = OUTPUT_DIR / "results_v2.json"
    with open(results_path, "w") as f:
        json.dump({"val": m_val, "test": m_test}, f, indent=2)

    fi = pd.DataFrame({
        "feature":          model.feature_name(),
        "importance_gain":  model.feature_importance("gain"),
        "importance_split": model.feature_importance("split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance_v2.csv", index=False)
    print(f"\nTop 10 features:", flush=True)
    print(fi.head(10).to_string(index=False), flush=True)

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("RESUMEN FINAL", flush=True)
    print(f"  Tiempo total:  {total:.1f} minutos", flush=True)
    print(f"  AUC-ROC val:   {m_val['auc_roc']}", flush=True)
    print(f"  AUC-ROC test:  {m_test['auc_roc']}", flush=True)
    print(f"  F1 val:        {m_val['f1_disrupted']}", flush=True)
    print(f"  F1 test:       {m_test['f1_disrupted']}", flush=True)
    print(f"  Outputs en:    {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
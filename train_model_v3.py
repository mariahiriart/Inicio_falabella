"""
Entrenamiento modelo LightGBM v3 — con carrier y features de flujo.

Mejoras respecto al v2:
  - carrier_code: transportista asignado (BKSM, BLUEEXPRESS, IBIS, etc.)
  - n_eventos: cantidad de eventos del paquete
  - n_estados_distintos: cuántos estados atravesó
  - tuvo_delivery_attempted: intento fallido de entrega
  - tuvo_exception: excepción en el ciclo
  - tuvo_annulled: orden anulada
  - horas_max_gap: mayor tiempo sin actividad (impasse)
  - horas_ultimo_evento: tiempo total del ciclo de eventos

Dataset: staging_marts.ml_dataset_v3
Split:   temporal (train=2025-01/11, val=2025-12/2026-01, test=2026-02)

Uso:
    python3 train_model_v3.py

Outputs en ml_outputs/:
    model_lgbm_v3.pkl
    results_v3.json
    feature_importance_v3.csv
    comparacion_v1_v2_v3.json
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

# ── Features ─────────────────────────────────────────────────────────────────

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
    # features históricas v2
    "seller_n_ordenes",
    "seller_tasa_disrupcion",
    "categoria_tasa_disrupcion",
    "categoria_cac_tasa_disrupcion",
    "dia_semana_tasa_disrupcion",
    "franja_tasa_disrupcion",
    # features de flujo v3
    "n_eventos",
    "n_estados_distintos",
    "tuvo_delivery_attempted",
    "tuvo_exception",
    "tuvo_annulled",
    "horas_primer_evento",
    "horas_ultimo_evento",
    "horas_max_gap",
    "ciclo_completo",
]

FEATURES_CATEGORICAS = [
    "service_category",
    "franja_horaria",
    "carrier_code",       # nuevo v3
]

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
TARGET       = "target_binario"

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
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


def cargar_mes(year_month: str, encoders: dict, fit: bool = False):
    print(f"    {year_month}...", end=" ", flush=True)
    t0 = time.time()

    cols = ", ".join(ALL_FEATURES + [TARGET])
    query = text(f"""
        SELECT {cols}
        FROM staging_marts.ml_dataset_v3
        WHERE year_month = :ym
    """)

    for intento in range(3):
        try:
            eng = get_engine()
            with eng.connect() as conn:
                df = pd.read_sql(query, conn, params={"ym": year_month})
            eng.dispose()
            break
        except Exception as e:
            print(f"\n    Reintento {intento+1}: {str(e)[:50]}", flush=True)
            time.sleep(5)
            if intento == 2:
                return None, None, encoders

    if len(df) == 0:
        print("0 filas", flush=True)
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
            if col in encoders:
                le = encoders[col]
                known = set(le.classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known else "DESCONOCIDO"
                )
                df[col] = le.transform(df[col]).astype("int32")
            else:
                df[col] = 0

    X = df[ALL_FEATURES].values.astype("float32")
    y = df[TARGET].astype("int32").values

    del df; gc.collect()
    print(f"{len(y):,} filas en {time.time()-t0:.0f}s", flush=True)
    return X, y, encoders


def cargar_split(meses, encoders, fit=False, nombre=""):
    print(f"  Cargando {nombre} ({len(meses)} meses)...", flush=True)
    Xs, ys = [], []
    first = fit
    for mes in meses:
        X, y, encoders = cargar_mes(mes, encoders, fit=first)
        if X is not None:
            Xs.append(X)
            ys.append(y)
        first = False
        gc.collect()

    X_total = np.vstack(Xs)
    y_total = np.concatenate(ys)
    del Xs, ys; gc.collect()

    print(f"  {nombre}: {len(y_total):,} filas — disrupted: {100*y_total.mean():.1f}%", flush=True)
    return X_total, y_total, encoders


def evaluar(model, X, y, split_name):
    y_prob = model.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y, y_prob)
    rep    = classification_report(y, y_pred, output_dict=True)

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


def comparar_modelos(m_val, m_test):
    print(f"\n{'='*55}", flush=True)
    print("COMPARACIÓN v1 / v2 / v3", flush=True)
    print(f"{'='*55}", flush=True)

    resultados = {}

    for path, nombre in [
        (OUTPUT_DIR / "results_v1.json", "v1"),
        (OUTPUT_DIR / "results_v2.json", "v2"),
    ]:
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            resultados[nombre] = r

    header = f"{'Modelo':<8} {'AUC val':>10} {'F1 val':>10} {'AUC test':>10} {'F1 test':>10}"
    print(header, flush=True)
    print("-"*55, flush=True)

    for nombre, r in resultados.items():
        print(f"  {nombre:<6} {r['val']['auc_roc']:>10} {r['val']['f1_disrupted']:>10} "
              f"{r['test']['auc_roc']:>10} {r['test']['f1_disrupted']:>10}", flush=True)

    print(f"  {'v3':<6} {m_val['auc_roc']:>10} {m_val['f1_disrupted']:>10} "
          f"{m_test['auc_roc']:>10} {m_test['f1_disrupted']:>10}", flush=True)

    # Guardar comparación
    resultados["v3"] = {"val": m_val, "test": m_test}
    with open(OUTPUT_DIR / "comparacion_v1_v2_v3.json", "w") as f:
        json.dump(resultados, f, indent=2)


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("ENTRENAMIENTO LGBM v3 — con carrier + flujo", flush=True)
    print(f"Features: {len(ALL_FEATURES)} total", flush=True)
    print(f"  Nuevas vs v2: carrier_code, n_eventos, tuvo_exception,", flush=True)
    print(f"                tuvo_delivery_attempted, horas_max_gap", flush=True)
    print("="*60, flush=True)

    encoders = {}

    # ── 1. Cargar ────────────────────────────────────────────────
    print("\n[1/4] Cargando desde ml_dataset_v3...", flush=True)
    X_train, y_train, encoders = cargar_split(
        MESES_TRAIN, encoders, fit=True, nombre="train"
    )
    X_val,   y_val,   encoders = cargar_split(
        MESES_VAL, encoders, fit=False, nombre="val"
    )
    X_test,  y_test,  encoders = cargar_split(
        MESES_TEST, encoders, fit=False, nombre="test"
    )

    # ── 2. Datasets LightGBM ─────────────────────────────────────
    print("\n[2/4] Construyendo datasets...", flush=True)
    cat_idx = [ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS]

    dtrain = lgb.Dataset(X_train, label=y_train,
                         categorical_feature=cat_idx,
                         feature_name=ALL_FEATURES,
                         free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=y_val,
                         categorical_feature=cat_idx,
                         feature_name=ALL_FEATURES,
                         reference=dtrain, free_raw_data=True)
    del X_train, y_train; gc.collect()

    # ── 3. Entrenar ──────────────────────────────────────────────
    print("\n[3/4] Entrenando...", flush=True)
    t_train = time.time()

    model = lgb.train(
        PARAMS_LGBM,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    print(f"\n  Mejor iteración: {model.best_iteration}", flush=True)
    print(f"  Entrenamiento:   {(time.time()-t_train)/60:.1f} minutos", flush=True)
    del dtrain, dval; gc.collect()

    # ── 4. Evaluar ───────────────────────────────────────────────
    print("\n[4/4] Evaluando...", flush=True)
    m_val  = evaluar(model, X_val,  y_val,  "val")
    m_test = evaluar(model, X_test, y_test, "test")
    comparar_modelos(m_val, m_test)

    # ── Guardar ──────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "model_lgbm_v3.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":    model,
            "encoders": encoders,
            "features": ALL_FEATURES,
            "version":  "v3",
        }, f)

    with open(OUTPUT_DIR / "results_v3.json", "w") as f:
        json.dump({"val": m_val, "test": m_test}, f, indent=2)

    fi = pd.DataFrame({
        "feature":          model.feature_name(),
        "importance_gain":  model.feature_importance("gain"),
        "importance_split": model.feature_importance("split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance_v3.csv", index=False)

    print(f"\nTop 10 features:", flush=True)
    print(fi.head(10).to_string(index=False), flush=True)

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("RESUMEN FINAL — v3", flush=True)
    print(f"  Tiempo total:  {total:.1f} minutos", flush=True)
    print(f"  AUC-ROC val:   {m_val['auc_roc']}", flush=True)
    print(f"  AUC-ROC test:  {m_test['auc_roc']}", flush=True)
    print(f"  F1 val:        {m_val['f1_disrupted']}", flush=True)
    print(f"  F1 test:       {m_test['f1_disrupted']}", flush=True)
    print(f"  Outputs en:    {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()  
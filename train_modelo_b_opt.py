"""
Modelo B — Diagnóstico post-facto de disrupciones.
Versión optimizada: 3 clases, carga mes a mes, class weights.

Clases:
  0 — sin_disrupcion    (~89%)
  1 — ultima_milla      (~9.4%)
  2 — deposito_despacho (~1%)

Features: ex-ante + post-facto del ciclo (hours por tramo, deltas vs p95)

IMPORTANTE: usa info disponible SOLO después de completado el ciclo.
NO usar para predicción en tiempo real.

Uso:
    python3 train_modelo_b_opt.py

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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Features ─────────────────────────────────────────────────────────────────

FEATURES_EXANTE = [
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

FEATURES_POSTFACTO = [
    "hours_deposito",
    "hours_despacho",
    "hours_ultima_milla",
    "hours_total",
    "delta_deposito_vs_p95",
    "delta_despacho_vs_p95",
    "delta_ultima_milla_vs_p95",
]

FEATURES_CATEGORICAS = ["service_category"]

ALL_FEATURES = FEATURES_EXANTE + FEATURES_POSTFACTO + FEATURES_CATEGORICAS

# Meses por split
MESES_TRAIN = [
    "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06",
    "2025-07","2025-08","2025-09","2025-10","2025-11"
]
MESES_VAL  = ["2025-12","2026-01"]
MESES_TEST = ["2026-02"]

# Mapeo de target_tramo a 3 clases
def mapear_clase(tramo):
    if tramo is None or str(tramo).strip() in ("", "nan", "None"):
        return "sin_disrupcion"
    t = str(tramo).strip().lower()
    if "ultima_milla" in t or "milla" in t:
        return "ultima_milla"
    if t in ("sin_disrupcion", "sin disrupcion"):
        return "sin_disrupcion"
    return "deposito_despacho"


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


def cargar_mes(year_month: str, encoders: dict, fit: bool = False):
    """
    Carga un mes con JOIN ml_dataset_v1 + fct_orders.
    Reconecta para cada mes para evitar SSL timeout.
    """
    print(f"    Mes {year_month}...", end=" ", flush=True)
    t0 = time.time()

    exante_sql   = ", ".join([f"d.{c}" for c in FEATURES_EXANTE])
    postfacto_sql = ", ".join([f"f.{c}" for c in FEATURES_POSTFACTO])

    query = text(f"""
        SELECT
            d.logistic_order_id,
            d.target_tramo,
            {exante_sql},
            {postfacto_sql},
            d.service_category
        FROM staging_marts.ml_dataset_v1 d
        JOIN staging_marts.fct_orders f
          ON d.logistic_order_id = f.logistic_order_id
        WHERE d.year_month = :ym
          AND d.ciclo_completo = 1
          AND f.hours_total > 0
          AND d.target_tramo IS NOT NULL
    """)

    cols = (["logistic_order_id", "target_tramo"]
            + FEATURES_EXANTE + FEATURES_POSTFACTO + ["service_category"])

    for intento in range(3):
        try:
            eng = get_engine()
            with eng.connect() as conn:
                df = pd.read_sql(query, conn, params={"ym": year_month})
            eng.dispose()
            break
        except Exception as e:
            print(f"\n    Reintento {intento+1}/3: {str(e)[:60]}", flush=True)
            time.sleep(5)
            if intento == 2:
                raise

    if len(df) == 0:
        print("0 filas — saltando", flush=True)
        return None, None, encoders

    # Target: mapear a 3 clases
    df["clase"] = df["target_tramo"].apply(mapear_clase)

    # Numéricas
    for col in FEATURES_EXANTE + FEATURES_POSTFACTO:
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
    y = df["clase"].values

    del df; gc.collect()
    print(f"{len(y):,} filas en {time.time()-t0:.0f}s", flush=True)
    return X, y, encoders


def cargar_split(meses, encoders, fit=False, nombre=""):
    """Carga varios meses y concatena."""
    print(f"  Cargando {nombre} ({len(meses)} meses)...", flush=True)
    Xs, ys = [], []
    first_fit = fit
    for mes in meses:
        X, y, encoders = cargar_mes(mes, encoders, fit=first_fit)
        if X is not None:
            Xs.append(X)
            ys.append(y)
        first_fit = False
        gc.collect()

    X_total = np.vstack(Xs)
    y_total = np.concatenate(ys)
    del Xs, ys; gc.collect()

    # Distribución
    clases, counts = np.unique(y_total, return_counts=True)
    print(f"  {nombre}: {len(y_total):,} filas", flush=True)
    for c, n in zip(clases, counts):
        print(f"    {c:<25} {n:>8,} ({100*n/len(y_total):.1f}%)", flush=True)
    return X_total, y_total, encoders


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("MODELO B — DIAGNÓSTICO POST-FACTO (3 clases, mes a mes)", flush=True)
    print("Clases: sin_disrupcion / ultima_milla / deposito_despacho", flush=True)
    print(f"Features: {len(ALL_FEATURES)} ({len(FEATURES_EXANTE)} ex-ante, {len(FEATURES_POSTFACTO)} post-facto)", flush=True)
    print("="*60, flush=True)

    encoders = {}

    # ── 1. Cargar ────────────────────────────────────────────────
    print("\n[1/4] Cargando datos mes a mes...", flush=True)
    X_train, y_train, encoders = cargar_split(MESES_TRAIN, encoders, fit=True,  nombre="train")
    X_val,   y_val,   encoders = cargar_split(MESES_VAL,   encoders, fit=False, nombre="val")
    X_test,  y_test,  encoders = cargar_split(MESES_TEST,  encoders, fit=False, nombre="test")

    # ── 2. Encodear target ───────────────────────────────────────
    print("\n[2/4] Encoding target...", flush=True)
    le_target = LabelEncoder()
    le_target.fit(["sin_disrupcion", "ultima_milla", "deposito_despacho"])
    print(f"  Clases: {list(le_target.classes_)}", flush=True)

    y_train_enc = le_target.transform(y_train).astype("int32")
    y_val_enc   = le_target.transform(y_val).astype("int32")
    y_test_enc  = le_target.transform(y_test).astype("int32")
    del y_train, y_val, y_test; gc.collect()

    # Class weights — compensar desbalance
    n_total = len(y_train_enc)
    class_counts = np.bincount(y_train_enc)
    weights_map = {i: n_total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    sample_weights = np.array([weights_map[y] for y in y_train_enc], dtype="float32")
    print(f"  Class weights: {weights_map}", flush=True)

    # ── 3. Entrenar ──────────────────────────────────────────────
    print("\n[3/4] Entrenando LightGBM multiclase...", flush=True)
    t_train = time.time()

    params = {
        "objective":         "multiclass",
        "num_class":         3,
        "metric":            "multi_logloss",
        "boosting_type":     "gbdt",
        "learning_rate":     0.05,
        "num_leaves":        63,
        "max_depth":         -1,
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

    cat_idx = [ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS]

    dtrain = lgb.Dataset(
        X_train, label=y_train_enc,
        weight=sample_weights,
        categorical_feature=cat_idx,
        feature_name=ALL_FEATURES,
        free_raw_data=True,
    )
    dval = lgb.Dataset(
        X_val, label=y_val_enc,
        categorical_feature=cat_idx,
        feature_name=ALL_FEATURES,
        reference=dtrain,
        free_raw_data=True,
    )

    del X_train, y_train_enc, sample_weights; gc.collect()

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    print(f"\n  Mejor iteración: {model.best_iteration}", flush=True)
    print(f"  Entrenamiento: {(time.time()-t_train)/60:.1f} minutos", flush=True)
    del dtrain, dval; gc.collect()

    # ── 4. Evaluar ───────────────────────────────────────────────
    print("\n[4/4] Evaluando...", flush=True)

    def evaluar(X, y_enc, split_name):
        y_prob = model.predict(X)
        y_pred = np.argmax(y_prob, axis=1)

        acc    = (y_pred == y_enc).mean()
        f1_mac = f1_score(y_enc, y_pred, average="macro")
        f1_wt  = f1_score(y_enc, y_pred, average="weighted")
        report = classification_report(
            y_enc, y_pred,
            target_names=le_target.classes_,
            output_dict=True
        )
        cm = confusion_matrix(y_enc, y_pred).tolist()

        print(f"\n{'='*55}", flush=True)
        print(f"MÉTRICAS MODELO B — {split_name.upper()}", flush=True)
        print(f"  Accuracy:    {acc:.4f}", flush=True)
        print(f"  F1 macro:    {f1_mac:.4f}", flush=True)
        print(f"  F1 weighted: {f1_wt:.4f}", flush=True)
        print(f"\n  Por clase:", flush=True)
        for clase in le_target.classes_:
            r = report.get(clase, {})
            print(f"    {clase:<25} P={r.get('precision',0):.3f}  R={r.get('recall',0):.3f}  F1={r.get('f1-score',0):.3f}  n={int(r.get('support',0)):,}", flush=True)
        print(f"\n  Matriz de confusión:", flush=True)
        print(f"  Clases: {list(le_target.classes_)}", flush=True)
        for i, row in enumerate(cm):
            print(f"    {le_target.classes_[i]:<25} {row}", flush=True)

        return {
            "split": split_name, "n": int(len(y_enc)),
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1_mac), 4),
            "f1_weighted": round(float(f1_wt), 4),
            "por_clase": {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                          for k, v in report.items() if k in le_target.classes_},
            "confusion_matrix": cm,
            "clases": list(le_target.classes_),
        }

    m_val  = evaluar(X_val,  y_val_enc,  "val")
    m_test = evaluar(X_test, y_test_enc, "test")

    # ── Guardar ──────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "model_lgbm_b.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model, "encoders": encoders,
            "le_target": le_target, "features": ALL_FEATURES,
            "features_exante": FEATURES_EXANTE,
            "features_postfacto": FEATURES_POSTFACTO,
            "version": "modelo_b_v1",
            "clases": list(le_target.classes_),
        }, f)

    with open(OUTPUT_DIR / "results_b.json", "w") as f:
        json.dump({"val": m_val, "test": m_test}, f, indent=2)

    fi = pd.DataFrame({
        "feature":          model.feature_name(),
        "importance_gain":  model.feature_importance("gain"),
        "importance_split": model.feature_importance("split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance_b.csv", index=False)

    print(f"\nTop 10 features:", flush=True)
    print(fi.head(10).to_string(index=False), flush=True)

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("RESUMEN FINAL — MODELO B", flush=True)
    print(f"  Tiempo total:  {total:.1f} minutos", flush=True)
    print(f"  Accuracy val:  {m_val['accuracy']}", flush=True)
    print(f"  F1 macro val:  {m_val['f1_macro']}", flush=True)
    print(f"  Accuracy test: {m_test['accuracy']}", flush=True)
    print(f"  F1 macro test: {m_test['f1_macro']}", flush=True)
    print(f"  Outputs en:    {OUTPUT_DIR}", flush=True)
    print("\n  RECORDATORIO: modelo post-facto, NO usar para alertas.", flush=True)


if __name__ == "__main__":
    main()

"""
Modelo D — Predicción por flujo de la orden.

Usa la secuencia de eventos de stg_event_packages para predecir:
  1. Si la orden va a tener disrupción (binario)
  2. En qué tramo va a fallar (multiclase)

Features del flujo (calculadas desde los eventos):
  - horas_a_shipment_confirmed
  - horas_a_out_for_delivery
  - horas_entre_eventos_max
  - horas_entre_eventos_avg
  - n_eventos_total
  - n_estados_distintos
  - tuvo_delivery_attempted
  - tuvo_exception
  - tuvo_annulled
  - ultimo_estado

Más todas las features ex-ante del v2.

Uso:
    python3 train_modelo_d.py

Outputs en ml_outputs/:
    model_lgbm_d_binario.pkl
    model_lgbm_d_tramo.pkl
    results_d.json
    feature_importance_d_binario.csv
    feature_importance_d_tramo.csv
"""

import os, gc, json, time, pickle, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    roc_auc_score, classification_report,
    f1_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Features ─────────────────────────────────────────────────────────────────

# Features ex-ante del v2
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
    "seller_n_ordenes",
    "seller_tasa_disrupcion",
    "categoria_tasa_disrupcion",
    "categoria_cac_tasa_disrupcion",
    "dia_semana_tasa_disrupcion",
    "franja_tasa_disrupcion",
]

# Features del flujo (calculadas desde event_packages)
FEATURES_FLUJO = [
    "horas_a_shipment_confirmed",
    "horas_a_out_for_delivery",
    "horas_entre_eventos_max",
    "horas_entre_eventos_avg",
    "n_eventos_total",
    "n_estados_distintos",
    "tuvo_delivery_attempted",
    "tuvo_exception",
    "tuvo_annulled",
    "esta_en_transito",
    "esta_out_for_delivery",
    "esta_shipment_confirmed",
]

FEATURES_CATEGORICAS = [
    "service_category",
    "franja_horaria",
    "ultimo_estado",
]

ALL_FEATURES = FEATURES_EXANTE + FEATURES_FLUJO + FEATURES_CATEGORICAS

MESES_TRAIN = [
    "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06",
    "2025-07","2025-08","2025-09","2025-10","2025-11"
]
MESES_VAL  = ["2025-12","2026-01"]
MESES_TEST = ["2026-02"]

# Mapeo tramo a 3 clases
def mapear_tramo(tramo):
    if tramo is None or str(tramo).strip() in ("", "nan", "None", "sin_disrupcion"):
        return "sin_disrupcion"
    t = str(tramo).strip().lower()
    if "ultima_milla" in t or "milla" in t:
        return "ultima_milla"
    return "deposito_despacho"


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


def calcular_features_flujo(engine, year_month: str) -> pd.DataFrame:
    """
    Calcula features del flujo desde stg_event_packages para un mes.
    Devuelve un DataFrame con una fila por orden.
    """
    q = text("""
        SELECT
            logistic_order_id,
            event_dt,
            package_status
        FROM staging_marts.stg_event_packages
        WHERE year_month = :ym
          AND logistic_order_id IS NOT NULL
          AND event_dt IS NOT NULL
        ORDER BY logistic_order_id, event_dt
    """)

    try:
        eng = get_engine()
        with eng.connect() as conn:
            df = pd.read_sql(q, conn, params={"ym": year_month})
        eng.dispose()
    except Exception as e:
        print(f"    Error leyendo eventos: {e}", flush=True)
        return pd.DataFrame()

    if len(df) == 0:
        return pd.DataFrame()

    df["event_dt"] = pd.to_datetime(df["event_dt"], errors="coerce")
    df = df.dropna(subset=["event_dt"])

    # Calcular features por orden
    features = []
    for order_id, grp in df.groupby("logistic_order_id"):
        grp = grp.sort_values("event_dt")
        estados = grp["package_status"].tolist()
        tiempos = grp["event_dt"].tolist()

        # Tiempo al primer evento desde el inicio
        t_inicio = tiempos[0]

        # Horas a estados clave
        def horas_a_estado(estado):
            for t, s in zip(tiempos, estados):
                if s == estado:
                    return max(0, (t - t_inicio).total_seconds() / 3600)
            return -1  # no llegó a ese estado

        # Gaps entre eventos consecutivos
        gaps = []
        for i in range(1, len(tiempos)):
            gap = (tiempos[i] - tiempos[i-1]).total_seconds() / 3600
            if gap >= 0:
                gaps.append(gap)

        features.append({
            "logistic_order_id":        order_id,
            "horas_a_shipment_confirmed": horas_a_estado("SHIPMENT_CONFIRMED"),
            "horas_a_out_for_delivery":   horas_a_estado("OUT_FOR_DELIVERY"),
            "horas_entre_eventos_max":    max(gaps) if gaps else 0,
            "horas_entre_eventos_avg":    np.mean(gaps) if gaps else 0,
            "n_eventos_total":            len(grp),
            "n_estados_distintos":        grp["package_status"].nunique(),
            "tuvo_delivery_attempted":    int("DELIVERY_ATTEMPTED" in estados),
            "tuvo_exception":             int("EXCEPTION" in estados),
            "tuvo_annulled":              int("ANNULLED" in estados),
            "esta_en_transito":           int(estados[-1] == "IN_TRANSIT"),
            "esta_out_for_delivery":      int(estados[-1] == "OUT_FOR_DELIVERY"),
            "esta_shipment_confirmed":    int(estados[-1] == "SHIPMENT_CONFIRMED"),
            "ultimo_estado":              estados[-1],
        })

    del df; gc.collect()
    return pd.DataFrame(features)


def cargar_mes(year_month: str, encoders: dict,
               le_tramo=None, fit: bool = False):
    """
    Carga un mes: features ex-ante desde ml_dataset_v2 +
    features de flujo desde stg_event_packages.
    """
    print(f"    Mes {year_month}...", end=" ", flush=True)
    t0 = time.time()

    # Features ex-ante desde RDS
    cols = ", ".join(
        ["d.logistic_order_id", "d.target_binario", "d.target_tramo",
         "d.service_category", "d.franja_horaria"]
        + [f"d.{c}" for c in FEATURES_EXANTE]
    )

    q = text(f"""
        SELECT {cols}
        FROM staging_marts.ml_dataset_v2 d
        WHERE d.year_month = :ym
    """)

    for intento in range(3):
        try:
            eng = get_engine()
            with eng.connect() as conn:
                df_base = pd.read_sql(q, conn, params={"ym": year_month})
            eng.dispose()
            break
        except Exception as e:
            print(f"\n    Reintento {intento+1}: {str(e)[:50]}", flush=True)
            time.sleep(5)
            if intento == 2:
                print(f"    ERROR — saltando mes {year_month}", flush=True)
                return None, None, None, encoders, le_tramo

    if len(df_base) == 0:
        print("0 filas", flush=True)
        return None, None, None, encoders, le_tramo

    # Features de flujo desde stg_event_packages
    eng = get_engine()
    df_flujo = calcular_features_flujo(eng, year_month)
    eng.dispose()

    if len(df_flujo) == 0:
        # Si no hay eventos para este mes, usamos -1 como fallback
        for col in FEATURES_FLUJO:
            df_base[col] = -1
        df_base["ultimo_estado"] = "DESCONOCIDO"
    else:
        df_base = df_base.merge(df_flujo, on="logistic_order_id", how="left")
        for col in FEATURES_FLUJO:
            if col in df_base.columns:
                df_base[col] = df_base[col].fillna(-1)
        df_base["ultimo_estado"] = df_base["ultimo_estado"].fillna("DESCONOCIDO")

    # Numéricas
    for col in FEATURES_EXANTE + FEATURES_FLUJO:
        df_base[col] = pd.to_numeric(df_base[col], errors="coerce").fillna(-1).astype("float32")

    # Categóricas
    for col in FEATURES_CATEGORICAS:
        df_base[col] = df_base[col].fillna("DESCONOCIDO").astype(str)
        if fit and col not in encoders:
            le = LabelEncoder()
            df_base[col] = le.fit_transform(df_base[col]).astype("int32")
            encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                known = set(le.classes_)
                df_base[col] = df_base[col].apply(
                    lambda x: x if x in known else "DESCONOCIDO"
                )
                df_base[col] = le.transform(df_base[col]).astype("int32")
            else:
                df_base[col] = 0

    X = df_base[ALL_FEATURES].values.astype("float32")
    y_bin = df_base["target_binario"].astype("int32").values

    # Target tramo
    y_tramo_str = df_base["target_tramo"].apply(mapear_tramo).values
    if fit:
        le_tramo = LabelEncoder()
        le_tramo.fit(["sin_disrupcion", "ultima_milla", "deposito_despacho"])
    y_tramo = le_tramo.transform(y_tramo_str).astype("int32")

    n = len(X)
    elapsed = time.time() - t0
    print(f"{n:,} filas en {elapsed:.0f}s", flush=True)

    del df_base; gc.collect()
    return X, y_bin, y_tramo, encoders, le_tramo


def cargar_split(meses, encoders, le_tramo=None, fit=False, nombre=""):
    print(f"  Cargando {nombre} ({len(meses)} meses)...", flush=True)
    Xs, ys_bin, ys_tramo = [], [], []
    first = fit
    for mes in meses:
        X, y_bin, y_tramo, encoders, le_tramo = cargar_mes(
            mes, encoders, le_tramo, fit=first
        )
        if X is not None:
            Xs.append(X)
            ys_bin.append(y_bin)
            ys_tramo.append(y_tramo)
        first = False
        gc.collect()

    X_total     = np.vstack(Xs)
    y_bin_total = np.concatenate(ys_bin)
    y_tr_total  = np.concatenate(ys_tramo)
    del Xs, ys_bin, ys_tramo; gc.collect()

    print(f"  {nombre}: {len(X_total):,} filas — disrupted: {100*y_bin_total.mean():.1f}%", flush=True)
    return X_total, y_bin_total, y_tr_total, encoders, le_tramo


def entrenar_binario(X_train, y_train, X_val, y_val):
    print("\n  Entrenando modelo D — BINARIO...", flush=True)
    params = {
        "objective": "binary", "metric": ["binary_logloss","auc"],
        "boosting_type": "gbdt", "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 200,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "lambda_l1": 0.1, "lambda_l2": 0.1,
        "n_jobs": -1, "verbose": -1, "random_state": 42,
    }
    cat_idx = [ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS]
    dtrain = lgb.Dataset(X_train, label=y_train,
                         categorical_feature=cat_idx,
                         feature_name=ALL_FEATURES, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=y_val,
                         categorical_feature=cat_idx,
                         feature_name=ALL_FEATURES,
                         reference=dtrain, free_raw_data=True)
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[
                          lgb.early_stopping(50, verbose=True),
                          lgb.log_evaluation(50)
                      ])
    print(f"  Mejor iteración: {model.best_iteration}", flush=True)
    return model


def entrenar_tramo(X_train, y_train, X_val, y_val, le_tramo):
    print("\n  Entrenando modelo D — TRAMO...", flush=True)
    n_clases = len(le_tramo.classes_)

    # Class weights para compensar desbalance
    counts = np.bincount(y_train)
    weights = {i: len(y_train) / (n_clases * c) for i, c in enumerate(counts)}
    sample_w = np.array([weights[y] for y in y_train], dtype="float32")

    params = {
        "objective": "multiclass", "num_class": n_clases,
        "metric": "multi_logloss", "boosting_type": "gbdt",
        "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 100, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 0.1,
        "n_jobs": -1, "verbose": -1, "random_state": 42,
    }
    cat_idx = [ALL_FEATURES.index(c) for c in FEATURES_CATEGORICAS]
    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_w,
                         categorical_feature=cat_idx,
                         feature_name=ALL_FEATURES, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=y_val,
                         categorical_feature=cat_idx,
                         feature_name=ALL_FEATURES,
                         reference=dtrain, free_raw_data=True)
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[
                          lgb.early_stopping(50, verbose=True),
                          lgb.log_evaluation(50)
                      ])
    print(f"  Mejor iteración: {model.best_iteration}", flush=True)
    return model


def evaluar_binario(model, X, y, split_name):
    y_prob = model.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y, y_prob)
    rep = classification_report(y, y_pred, output_dict=True)
    print(f"\n  [{split_name}] AUC={auc:.4f}  Acc={rep['accuracy']:.4f}  "
          f"F1={rep['1']['f1-score']:.4f}  "
          f"Recall={rep['1']['recall']:.4f}", flush=True)
    return {
        "split": split_name, "n": len(y),
        "auc_roc": round(float(auc), 4),
        "accuracy": round(float(rep["accuracy"]), 4),
        "f1_disrupted": round(float(rep["1"]["f1-score"]), 4),
        "recall_disrupted": round(float(rep["1"]["recall"]), 4),
        "precision_disrupted": round(float(rep["1"]["precision"]), 4),
    }


def evaluar_tramo(model, X, y, split_name, le_tramo):
    y_prob = model.predict(X)
    y_pred = np.argmax(y_prob, axis=1)
    acc    = accuracy_score(y, y_pred)
    f1_mac = f1_score(y, y_pred, average="macro")
    rep    = classification_report(y, y_pred,
                                   target_names=le_tramo.classes_,
                                   output_dict=True)
    print(f"\n  [{split_name}] Acc={acc:.4f}  F1_macro={f1_mac:.4f}", flush=True)
    for clase in le_tramo.classes_:
        r = rep.get(clase, {})
        print(f"    {clase:<25} P={r.get('precision',0):.3f}  "
              f"R={r.get('recall',0):.3f}  F1={r.get('f1-score',0):.3f}", flush=True)
    return {
        "split": split_name, "n": len(y),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_mac), 4),
        "clases": list(le_tramo.classes_),
        "por_clase": {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                      for k, v in rep.items() if k in le_tramo.classes_},
    }


def guardar_fi(model, path, nombre):
    fi = pd.DataFrame({
        "feature":          model.feature_name(),
        "importance_gain":  model.feature_importance("gain"),
        "importance_split": model.feature_importance("split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(path, index=False)
    print(f"\n  Top 10 features — {nombre}:", flush=True)
    print(fi.head(10)[["feature","importance_gain"]].to_string(index=False), flush=True)
    return fi


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("MODELO D — PREDICCIÓN POR FLUJO DE LA ORDEN", flush=True)
    print(f"Features: {len(ALL_FEATURES)} ({len(FEATURES_EXANTE)} ex-ante, "
          f"{len(FEATURES_FLUJO)} flujo, {len(FEATURES_CATEGORICAS)} cat)", flush=True)
    print("="*60, flush=True)

    encoders, le_tramo = {}, None

    # ── 1. Cargar datos ──────────────────────────────────────────
    print("\n[1/4] Cargando datos...", flush=True)
    X_train, y_bin_train, y_tr_train, encoders, le_tramo = cargar_split(
        MESES_TRAIN, encoders, le_tramo, fit=True, nombre="train"
    )
    X_val, y_bin_val, y_tr_val, encoders, le_tramo = cargar_split(
        MESES_VAL, encoders, le_tramo, fit=False, nombre="val"
    )
    X_test, y_bin_test, y_tr_test, encoders, le_tramo = cargar_split(
        MESES_TEST, encoders, le_tramo, fit=False, nombre="test"
    )

    print(f"\n  Clases tramo: {list(le_tramo.classes_)}", flush=True)

    # ── 2. Entrenar ──────────────────────────────────────────────
    print("\n[2/4] Entrenando...", flush=True)
    t_train = time.time()

    model_bin   = entrenar_binario(X_train, y_bin_train, X_val, y_bin_val)
    model_tramo = entrenar_tramo(X_train, y_tr_train, X_val, y_tr_val, le_tramo)

    print(f"\n  Entrenamiento: {(time.time()-t_train)/60:.1f} minutos", flush=True)
    del X_train, y_bin_train, y_tr_train; gc.collect()

    # ── 3. Evaluar ───────────────────────────────────────────────
    print("\n[3/4] Evaluando...", flush=True)
    print("\n  === MODELO BINARIO ===", flush=True)
    m_bin_val  = evaluar_binario(model_bin, X_val,  y_bin_val,  "val")
    m_bin_test = evaluar_binario(model_bin, X_test, y_bin_test, "test")

    print("\n  === MODELO TRAMO ===", flush=True)
    m_tr_val  = evaluar_tramo(model_tramo, X_val,  y_tr_val,  "val",  le_tramo)
    m_tr_test = evaluar_tramo(model_tramo, X_test, y_tr_test, "test", le_tramo)

    # Comparar binario con v2
    v2_path = OUTPUT_DIR / "results_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            v2 = json.load(f)
        print(f"\n{'='*50}", flush=True)
        print("COMPARACIÓN v2 vs Modelo D (binario)", flush=True)
        print(f"  AUC-ROC val:  v2={v2['val']['auc_roc']}  D={m_bin_val['auc_roc']}  "
              f"Δ={m_bin_val['auc_roc']-v2['val']['auc_roc']:+.4f}", flush=True)
        print(f"  F1 val:       v2={v2['val']['f1_disrupted']}  D={m_bin_val['f1_disrupted']}  "
              f"Δ={m_bin_val['f1_disrupted']-v2['val']['f1_disrupted']:+.4f}", flush=True)
        print(f"  AUC-ROC test: v2={v2['test']['auc_roc']}  D={m_bin_test['auc_roc']}  "
              f"Δ={m_bin_test['auc_roc']-v2['test']['auc_roc']:+.4f}", flush=True)

    # ── 4. Guardar ───────────────────────────────────────────────
    print("\n[4/4] Guardando...", flush=True)

    with open(OUTPUT_DIR / "model_lgbm_d_binario.pkl", "wb") as f:
        pickle.dump({
            "model": model_bin, "encoders": encoders,
            "features": ALL_FEATURES, "version": "modelo_d_binario",
            "features_flujo": FEATURES_FLUJO,
        }, f)

    with open(OUTPUT_DIR / "model_lgbm_d_tramo.pkl", "wb") as f:
        pickle.dump({
            "model": model_tramo, "encoders": encoders,
            "le_tramo": le_tramo, "features": ALL_FEATURES,
            "version": "modelo_d_tramo",
            "features_flujo": FEATURES_FLUJO,
        }, f)

    with open(OUTPUT_DIR / "results_d.json", "w") as f:
        json.dump({
            "binario": {"val": m_bin_val, "test": m_bin_test},
            "tramo":   {"val": m_tr_val,  "test": m_tr_test},
        }, f, indent=2)

    guardar_fi(model_bin,   OUTPUT_DIR / "feature_importance_d_binario.csv", "binario")
    guardar_fi(model_tramo, OUTPUT_DIR / "feature_importance_d_tramo.csv",   "tramo")

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("RESUMEN FINAL — MODELO D", flush=True)
    print(f"  Tiempo total:        {total:.1f} minutos", flush=True)
    print(f"  Binario AUC-ROC val: {m_bin_val['auc_roc']}", flush=True)
    print(f"  Binario AUC-ROC test:{m_bin_test['auc_roc']}", flush=True)
    print(f"  Tramo F1 macro val:  {m_tr_val['f1_macro']}", flush=True)
    print(f"  Tramo F1 macro test: {m_tr_test['f1_macro']}", flush=True)
    print(f"  Outputs en: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

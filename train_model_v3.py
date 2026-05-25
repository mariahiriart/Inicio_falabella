"""
Entrenamiento modelo LightGBM v3 — carrier directo desde S3.

Lee el carrier desde los parquet crudos de S3 en lugar de RDS.
Filtra solo Chile (country=CL) y solo eventos STATE_CHANGED.
Une con ml_dataset_v2 de RDS en memoria.

Ventajas:
  - No necesita cargar stg_carrier a RDS
  - Filtro de país garantizado
  - Más rápido de implementar

Uso:
    python3 train_model_v3_s3.py

Outputs en ml_outputs/:
    model_lgbm_v3.pkl
    results_v3.json
    feature_importance_v3.csv
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
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

S3_BASE    = "s3://falabella-data"
EVENT_TYPE = "FULFILMENT_ORDER_ITEM_QUANTITY_STATE_CHANGED"

MESES_S3 = [
    "202501","202502","202503","202504","202505","202506",
    "202507","202508","202509","202510","202511","202512",
    "202601","202602"
]

MESES_TRAIN = [
    "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06",
    "2025-07","2025-08","2025-09","2025-10","2025-11"
]
MESES_VAL  = ["2025-12","2026-01"]
MESES_TEST = ["2026-02"]

# ── Features ─────────────────────────────────────────────────────────────────

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
    "carrier_code",
]

ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
TARGET       = "target_binario"

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


# ── Extracción de carrier desde S3 ───────────────────────────────────────────

def extraer_carrier_mes(mes_short: str) -> pd.DataFrame:
    """
    Lee un mes desde S3, filtra CL + STATE_CHANGED y extrae carrier.
    Devuelve DataFrame con logistic_order_id y carrier_code.
    """
    s3_path = f"{S3_BASE}/{mes_short}/"
    year_month = f"{mes_short[:4]}-{mes_short[4:]}"

    try:
        df = pd.read_parquet(s3_path, columns=["header", "data"])
    except Exception as e:
        print(f"    [{year_month}] ERROR S3: {e}", flush=True)
        return pd.DataFrame(columns=["logistic_order_id", "carrier_code"])

    if len(df) == 0:
        return pd.DataFrame(columns=["logistic_order_id", "carrier_code"])

    # Parsear header
    def parse_header(h):
        try:
            d = json.loads(h) if isinstance(h, str) else h
            return (
                d.get("eventType", ""),
                d.get("logisticOrderId", ""),
                d.get("country", "")
            )
        except:
            return ("", "", "")

    parsed = df["header"].apply(parse_header)
    df["event_type"]        = parsed.apply(lambda x: x[0])
    df["logistic_order_id"] = parsed.apply(lambda x: x[1])
    df["country"]           = parsed.apply(lambda x: x[2])

    # Filtrar solo CL + STATE_CHANGED
    df = df[
        (df["country"] == "CL") &
        (df["event_type"] == EVENT_TYPE)
    ].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=["logistic_order_id", "carrier_code"])

    # Extraer carrier del JSON data
    def get_carrier(data_str):
        try:
            data = json.loads(data_str) if isinstance(data_str, str) else data_str
            for item in data.get("orderItems", []):
                for state in item.get("states", []):
                    for pkg in state.get("packagesInfo", []):
                        td = pkg.get("trackingData", {})
                        code = td.get("carrierCode", "")
                        if code and code.strip():
                            return code.strip()
        except:
            pass
        return None

    df["carrier_code"] = df["data"].apply(get_carrier)

    # Una fila por orden — carrier más frecuente
    result = (
        df[df["carrier_code"].notna() & (df["logistic_order_id"] != "")]
        .groupby("logistic_order_id")["carrier_code"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )

    del df; gc.collect()
    print(f"    [{year_month}] {len(result):,} órdenes con carrier CL", flush=True)
    return result


def cargar_todos_los_carriers() -> pd.DataFrame:
    """
    Carga carriers de todos los meses desde S3 en paralelo (4 workers).
    Devuelve un DataFrame con logistic_order_id y carrier_code.
    """
    print("\n[PRE] Extrayendo carriers desde S3 (4 workers en paralelo)...", flush=True)
    t0 = time.time()

    frames = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extraer_carrier_mes, mes): mes for mes in MESES_S3}
        for future in as_completed(futures):
            try:
                df = future.result()
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)

    if not frames:
        print("  Sin carriers encontrados — usando DESCONOCIDO para todos", flush=True)
        return pd.DataFrame(columns=["logistic_order_id", "carrier_code"])

    carriers = pd.concat(frames, ignore_index=True)

    # Si una orden aparece en varios meses, tomar el más frecuente
    carriers = (
        carriers.groupby("logistic_order_id")["carrier_code"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )

    print(f"\n  Total órdenes con carrier: {len(carriers):,}", flush=True)
    print(f"  Top carriers:", flush=True)
    print(carriers["carrier_code"].value_counts().head(10).to_string(), flush=True)
    print(f"  Tiempo: {(time.time()-t0)/60:.1f} minutos", flush=True)

    del frames; gc.collect()
    return carriers


# ── Carga de datos desde RDS ──────────────────────────────────────────────────

def cargar_mes_rds(year_month: str, carriers_df: pd.DataFrame,
                   encoders: dict, fit: bool = False):
    """
    Carga un mes desde ml_dataset_v2 en RDS y une con carriers.
    """
    print(f"    {year_month}...", end=" ", flush=True)
    t0 = time.time()

    # Columnas del v2 (sin carrier_code)
    cols_v2 = FEATURES_NUMERICAS + ["service_category", "franja_horaria", TARGET]
    cols_sql = ", ".join(cols_v2)

    query = text(f"""
        SELECT logistic_order_id, {cols_sql}
        FROM staging_marts.ml_dataset_v2
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

    # JOIN con carriers en memoria
    df = df.merge(carriers_df, on="logistic_order_id", how="left")
    df["carrier_code"] = df["carrier_code"].fillna("DESCONOCIDO")

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

    # Info de cobertura de carrier
    n_con_carrier = (df["carrier_code"] != 0).sum() if "carrier_code" in df.columns else 0
    pct = 100 * (df["carrier_code"].astype(str) != str(encoders.get("carrier_code",
          type("", (), {"transform": lambda self, x: [0]})()).transform(["DESCONOCIDO"])[0]
          )).mean() if "carrier_code" in encoders else 0

    del df; gc.collect()
    print(f"{len(y):,} filas en {time.time()-t0:.0f}s", flush=True)
    return X, y, encoders


def cargar_split(meses, carriers_df, encoders, fit=False, nombre=""):
    print(f"  Cargando {nombre} ({len(meses)} meses)...", flush=True)
    Xs, ys = [], []
    first = fit
    for mes in meses:
        X, y, encoders = cargar_mes_rds(mes, carriers_df, encoders, fit=first)
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


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("ENTRENAMIENTO LGBM v3 — carrier directo desde S3", flush=True)
    print(f"Features: {len(ALL_FEATURES)} | Carrier filtrado: country=CL", flush=True)
    print("="*60, flush=True)

    # ── PRE: Extraer carriers desde S3 ──────────────────────────
    carriers_df = cargar_todos_los_carriers()

    encoders = {}

    # ── 1. Cargar datos desde RDS + JOIN carrier en memoria ──────
    print("\n[1/4] Cargando desde ml_dataset_v2 + carrier...", flush=True)
    X_train, y_train, encoders = cargar_split(
        MESES_TRAIN, carriers_df, encoders, fit=True, nombre="train"
    )
    X_val,   y_val,   encoders = cargar_split(
        MESES_VAL, carriers_df, encoders, fit=False, nombre="val"
    )
    X_test,  y_test,  encoders = cargar_split(
        MESES_TEST, carriers_df, encoders, fit=False, nombre="test"
    )

    del carriers_df; gc.collect()

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

    # Comparar con versiones anteriores
    print(f"\n{'='*55}", flush=True)
    print("COMPARACIÓN v1 / v2 / v3", flush=True)
    print(f"{'Modelo':<8} {'AUC val':>10} {'F1 val':>10} {'AUC test':>10}", flush=True)
    print("-"*45, flush=True)
    for nombre, path in [("v1", "results_v1.json"), ("v2", "results_v2.json")]:
        p = OUTPUT_DIR / path
        if p.exists():
            r = json.load(open(p))
            print(f"  {nombre:<6} {r['val']['auc_roc']:>10} "
                  f"{r['val']['f1_disrupted']:>10} {r['test']['auc_roc']:>10}", flush=True)
    print(f"  {'v3':<6} {m_val['auc_roc']:>10} "
          f"{m_val['f1_disrupted']:>10} {m_test['auc_roc']:>10}", flush=True)

    # ── Guardar ──────────────────────────────────────────────────
    with open(OUTPUT_DIR / "model_lgbm_v3.pkl", "wb") as f:
        pickle.dump({
            "model": model, "encoders": encoders,
            "features": ALL_FEATURES, "version": "v3",
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
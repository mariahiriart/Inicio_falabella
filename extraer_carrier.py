"""
Extractor de carrier desde S3 → RDS — versión optimizada.

Mejoras vs v1:
  - Lee solo columnas 'header' y 'data' desde S3 (más rápido)
  - Procesa 3 meses en paralelo con ThreadPoolExecutor
  - Parseo vectorizado con apply en lugar de loop

Uso:
    python3 extraer_carrier_opt.py
"""

import os, gc, json, time, warnings
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

MESES = [
    "202501","202502","202503","202504","202505","202506",
    "202507","202508","202509","202510","202511","202512",
    "202601","202602"
]

S3_BASE    = "s3://falabella-data"
EVENT_TYPE = "FULFILMENT_ORDER_ITEM_QUANTITY_STATE_CHANGED"


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


def crear_tabla():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS staging_marts.stg_carrier (
                logistic_order_id  TEXT PRIMARY KEY,
                carrier_code       TEXT,
                carrier_name       TEXT,
                year_month         TEXT
            );
        """))
        conn.commit()
    engine.dispose()
    print("Tabla stg_carrier lista", flush=True)


def extraer_carrier_json(data_str):
    """Extrae carrierCode y carrierName del JSON de data."""
    try:
        data = json.loads(data_str) if isinstance(data_str, str) else data_str
        for item in data.get("orderItems", []):
            for state in item.get("states", []):
                for pkg in state.get("packagesInfo", []):
                    td = pkg.get("trackingData", {})
                    code = td.get("carrierCode", "")
                    if code and code.strip():
                        return code.strip(), td.get("carrierName", code).strip()
    except Exception:
        pass
    return None, None


def procesar_mes(year_month_short: str) -> pd.DataFrame:
    """Lee un mes desde S3 y extrae carriers."""
    s3_path   = f"{S3_BASE}/{year_month_short}/"
    year_month = f"{year_month_short[:4]}-{year_month_short[4:]}"

    print(f"  {year_month} leyendo S3...", end=" ", flush=True)
    t0 = time.time()

    try:
        # Leer solo header y data — mucho más rápido
        df = pd.read_parquet(s3_path, columns=["header", "data"])
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        return pd.DataFrame()

    if len(df) == 0:
        print("0 filas", flush=True)
        return pd.DataFrame()

    # Extraer event_type y order_id del header en un solo pass
    def parse_header(h):
        try:
            d = json.loads(h) if isinstance(h, str) else h
            return d.get("eventType", ""), d.get("logisticOrderId", "")
        except:
            return "", ""

    parsed = df["header"].apply(parse_header)
    df["event_type"]         = parsed.apply(lambda x: x[0])
    df["logistic_order_id"]  = parsed.apply(lambda x: x[1])

    # Filtrar solo el event type con carrier
    df = df[df["event_type"] == EVENT_TYPE].copy()

    if len(df) == 0:
        print("0 eventos con carrier", flush=True)
        return pd.DataFrame()

    print(f"{len(df):,} eventos → extrayendo carrier...", end=" ", flush=True)

    # Extraer carrier vectorizado
    carrier_parsed = df["data"].apply(extraer_carrier_json)
    df["carrier_code"] = carrier_parsed.apply(lambda x: x[0])
    df["carrier_name"] = carrier_parsed.apply(lambda x: x[1])
    df["year_month"]   = year_month

    # Una fila por orden con el carrier más frecuente
    df_valid = df[df["carrier_code"].notna() & (df["logistic_order_id"] != "")]
    if len(df_valid) == 0:
        print("0 con carrier", flush=True)
        return pd.DataFrame()

    result = (
        df_valid
        .groupby("logistic_order_id")
        .agg(
            carrier_code=("carrier_code", lambda x: x.value_counts().index[0]),
            carrier_name=("carrier_name", "first"),
            year_month=("year_month",   "first"),
        )
        .reset_index()
    )

    elapsed = time.time() - t0
    print(f"{len(result):,} ordenes en {elapsed:.0f}s", flush=True)

    del df, df_valid; gc.collect()
    return result


def insertar_en_rds(df: pd.DataFrame):
    """Inserta un DataFrame en stg_carrier ignorando duplicados."""
    if len(df) == 0:
        return 0
    try:
        eng = get_engine()
        # Usar INSERT ON CONFLICT DO NOTHING via SQL para no duplicar
        records = df.to_dict("records")
        with eng.connect() as conn:
            for i in range(0, len(records), 5000):
                batch = records[i:i+5000]
                conn.execute(text("""
                    INSERT INTO staging_marts.stg_carrier
                        (logistic_order_id, carrier_code, carrier_name, year_month)
                    VALUES (:logistic_order_id, :carrier_code, :carrier_name, :year_month)
                    ON CONFLICT (logistic_order_id) DO NOTHING
                """), batch)
            conn.commit()
        eng.dispose()
        return len(df)
    except Exception as e:
        print(f"    Error RDS: {e}", flush=True)
        return 0


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("EXTRACTOR CARRIER S3 → RDS (optimizado)", flush=True)
    print(f"Meses: {len(MESES)} | Paralelo: 3 simultáneos", flush=True)
    print("="*60, flush=True)

    crear_tabla()

    # Verificar si hay datos
    engine = get_engine()
    with engine.connect() as conn:
        n_exist = conn.execute(text(
            "SELECT COUNT(*) FROM staging_marts.stg_carrier"
        )).scalar()
    engine.dispose()

    if n_exist > 0:
        print(f"\nYa hay {n_exist:,} filas en stg_carrier.", flush=True)
        resp = input("¿Truncar y recargar? (s/n): ").strip().lower()
        if resp == "s":
            eng = get_engine()
            with eng.connect() as conn:
                conn.execute(text("TRUNCATE staging_marts.stg_carrier"))
                conn.commit()
            eng.dispose()
        else:
            print("Carga cancelada.", flush=True)
            return

    # Procesar en paralelo — 3 meses a la vez
    print(f"\nProcesando {len(MESES)} meses en paralelo...", flush=True)
    total = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(procesar_mes, mes): mes for mes in MESES}
        for future in as_completed(futures):
            mes = futures[future]
            try:
                df_carrier = future.result()
                if len(df_carrier) > 0:
                    n = insertar_en_rds(df_carrier)
                    total += n
                    print(f"  [{mes}] insertadas {n:,} en RDS", flush=True)
            except Exception as e:
                print(f"  [{mes}] ERROR: {e}", flush=True)

    # Resumen final
    print(f"\n{'='*60}", flush=True)
    engine = get_engine()
    with engine.connect() as conn:
        df_check = pd.read_sql(text("""
            SELECT carrier_code, carrier_name, COUNT(*) as n
            FROM staging_marts.stg_carrier
            GROUP BY carrier_code, carrier_name
            ORDER BY n DESC LIMIT 10
        """), conn)
        total_rds = conn.execute(text(
            "SELECT COUNT(*) FROM staging_marts.stg_carrier"
        )).scalar()
    engine.dispose()

    print("Top carriers encontrados:", flush=True)
    print(df_check.to_string(index=False), flush=True)
    print(f"\nTotal órdenes con carrier: {total_rds:,}", flush=True)
    print(f"Tiempo total: {(time.time()-t_inicio)/60:.1f} minutos", flush=True)


if __name__ == "__main__":
    main() 
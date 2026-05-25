"""
Extractor de carrier desde S3 → RDS — versión v2 (bajo consumo de memoria).
Lee cada archivo parquet individualmente y libera memoria entre cada uno.
"""

import os, gc, json, time, warnings
import boto3
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

MESES = [
    "202501","202502","202503","202504","202505","202506",
    "202507","202508","202509","202510","202511","202512",
    "202601","202602"
]

S3_BUCKET  = "falabella-data"
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


def listar_archivos_s3(mes: str):
    """Lista todos los archivos parquet de un mes en S3."""
    s3 = boto3.client("s3")
    prefix = f"{mes}/"
    paginator = s3.get_paginator("list_objects_v2")
    archivos = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                archivos.append(key)
    return archivos


def procesar_archivo(s3_key: str, year_month: str) -> pd.DataFrame:
    """Lee un solo archivo parquet desde S3 y extrae carriers."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        buf = BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf, columns=["header", "data"])
        buf.close()
    except Exception as e:
        print(f"    ERROR leyendo {s3_key}: {e}", flush=True)
        return pd.DataFrame()

    if len(df) == 0:
        return pd.DataFrame()

    # Parsear header
    def parse_header(h):
        try:
            d = json.loads(h) if isinstance(h, str) else h
            return d.get("eventType", ""), d.get("logisticOrderId", "")
        except:
            return "", ""

    parsed = df["header"].apply(parse_header)
    df["event_type"]        = parsed.apply(lambda x: x[0])
    df["logistic_order_id"] = parsed.apply(lambda x: x[1])

    # Filtrar event type
    df = df[df["event_type"] == EVENT_TYPE].copy()
    if len(df) == 0:
        del df; gc.collect()
        return pd.DataFrame()

    # Extraer carrier
    carrier_parsed = df["data"].apply(extraer_carrier_json)
    df["carrier_code"] = carrier_parsed.apply(lambda x: x[0])
    df["carrier_name"] = carrier_parsed.apply(lambda x: x[1])
    df["year_month"]   = year_month

    df_valid = df[df["carrier_code"].notna() & (df["logistic_order_id"] != "")].copy()
    del df; gc.collect()

    if len(df_valid) == 0:
        return pd.DataFrame()

    result = (
        df_valid
        .groupby("logistic_order_id")
        .agg(
            carrier_code=("carrier_code", lambda x: x.value_counts().index[0]),
            carrier_name=("carrier_name", "first"),
            year_month  =("year_month",   "first"),
        )
        .reset_index()
    )
    del df_valid; gc.collect()
    return result


def insertar_en_rds(df: pd.DataFrame):
    if len(df) == 0:
        return 0
    try:
        eng = get_engine()
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


def procesar_mes(mes: str) -> int:
    year_month = f"{mes[:4]}-{mes[4:]}"
    print(f"\n  [{mes}] Listando archivos S3...", flush=True)

    archivos = listar_archivos_s3(mes)
    if not archivos:
        print(f"  [{mes}] Sin archivos", flush=True)
        return 0

    print(f"  [{mes}] {len(archivos)} archivos — procesando uno por uno...", flush=True)
    total_mes = 0
    dfs = []

    for i, key in enumerate(archivos, 1):
        print(f"    [{mes}] archivo {i}/{len(archivos)}...", end=" ", flush=True)
        t0 = time.time()
        df = procesar_archivo(key, year_month)
        elapsed = time.time() - t0
        if len(df) > 0:
            dfs.append(df)
            print(f"{len(df):,} ordenes en {elapsed:.0f}s", flush=True)
        else:
            print(f"0 ordenes en {elapsed:.0f}s", flush=True)

    if dfs:
        df_mes = pd.concat(dfs, ignore_index=True)
        del dfs; gc.collect()

        # Deduplicar dentro del mes
        df_mes = (
            df_mes.groupby("logistic_order_id")
            .agg(
                carrier_code=("carrier_code", lambda x: x.value_counts().index[0]),
                carrier_name=("carrier_name", "first"),
                year_month  =("year_month",   "first"),
            )
            .reset_index()
        )
        n = insertar_en_rds(df_mes)
        total_mes = n
        print(f"  [{mes}] ✓ {n:,} órdenes insertadas en RDS", flush=True)
        del df_mes; gc.collect()

    return total_mes


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("EXTRACTOR CARRIER S3 → RDS (v2 - archivo por archivo)", flush=True)
    print(f"Meses: {len(MESES)} | Secuencial (bajo uso de memoria)", flush=True)
    print("="*60, flush=True)

    crear_tabla()

    engine = get_engine()
    with engine.connect() as conn:
        n_exist = conn.execute(text(
            "SELECT COUNT(*) FROM staging_marts.stg_carrier"
        )).scalar()
    engine.dispose()

    if n_exist > 0:
        print(f"\nYa hay {n_exist:,} filas en stg_carrier.", flush=True)
        print("Truncando y recargando...", flush=True)
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("TRUNCATE staging_marts.stg_carrier"))
            conn.commit()
        eng.dispose()

    total = 0
    for mes in MESES:
        total += procesar_mes(mes)

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
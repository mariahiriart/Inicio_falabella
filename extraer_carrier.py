"""
Extractor de carrier desde S3 → RDS.

Lee los parquet crudos de S3 (202501/ a 202602/),
filtra eventos FULFILMENT_ORDER_ITEM_QUANTITY_STATE_CHANGED,
parsea el JSON de 'data' y extrae carrierCode/carrierName.

Guarda en staging_marts.stg_carrier una fila por orden
con el carrier más frecuente.

Uso:
    python3 extraer_carrier.py

Tiempo estimado: 20-30 minutos
"""

import os, gc, json, time, warnings
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

MESES = [
    "202501","202502","202503","202504","202505","202506",
    "202507","202508","202509","202510","202511","202512",
    "202601","202602"
]

S3_BASE = "s3://falabella-data"
EVENT_TYPE = "FULFILMENT_ORDER_ITEM_QUANTITY_STATE_CHANGED"


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


def crear_tabla(engine):
    print("Creando tabla stg_carrier...", flush=True)
    q = text("""
        CREATE TABLE IF NOT EXISTS staging_marts.stg_carrier (
            logistic_order_id  TEXT PRIMARY KEY,
            carrier_code       TEXT,
            carrier_name       TEXT,
            year_month         TEXT
        );
    """)
    with engine.connect() as conn:
        conn.execute(q)
        conn.commit()
    print("  Tabla creada OK", flush=True)


def extraer_carrier_de_data(data_str: str) -> dict:
    """
    Parsea el JSON del campo 'data' y extrae carrierCode/carrierName.
    Busca en orderItems[].states[].packagesInfo[].trackingData
    """
    try:
        data = json.loads(data_str) if isinstance(data_str, str) else data_str
        carriers = []

        items = data.get("orderItems", [])
        for item in items:
            states = item.get("states", [])
            for state in states:
                packages = state.get("packagesInfo", [])
                for pkg in packages:
                    td = pkg.get("trackingData", {})
                    code = td.get("carrierCode", "")
                    name = td.get("carrierName", "")
                    if code and code.strip():
                        carriers.append({
                            "carrier_code": code.strip(),
                            "carrier_name": name.strip() if name else code.strip()
                        })

        if carriers:
            # Tomar el más frecuente
            df = pd.DataFrame(carriers)
            top = df["carrier_code"].value_counts().index[0]
            row = df[df["carrier_code"] == top].iloc[0]
            return {
                "carrier_code": row["carrier_code"],
                "carrier_name": row["carrier_name"]
            }
    except Exception:
        pass
    return {"carrier_code": None, "carrier_name": None}


def procesar_mes(year_month_short: str) -> pd.DataFrame:
    """
    Lee todos los parquet de un mes desde S3,
    filtra por event_type y extrae carrier.
    """
    s3_path = f"{S3_BASE}/{year_month_short}/"
    year_month = f"{year_month_short[:4]}-{year_month_short[4:]}"

    print(f"  {year_month}...", end=" ", flush=True)
    t0 = time.time()

    try:
        df = pd.read_parquet(s3_path)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        return pd.DataFrame()

    if len(df) == 0:
        print("0 filas", flush=True)
        return pd.DataFrame()

    # Filtrar solo el event type que tiene carrier
    df["event_type"] = df["header"].apply(
        lambda x: json.loads(x).get("eventType", "") if isinstance(x, str) else ""
    )
    df_filtered = df[df["event_type"] == EVENT_TYPE].copy()

    if len(df_filtered) == 0:
        print("0 eventos con carrier", flush=True)
        return pd.DataFrame()

    # Extraer order_id del header
    df_filtered["logistic_order_id"] = df_filtered["header"].apply(
        lambda x: json.loads(x).get("logisticOrderId", "") if isinstance(x, str) else ""
    )

    # Extraer carrier del data
    print(f"extrayendo carrier de {len(df_filtered):,} eventos...", end=" ", flush=True)
    carriers = df_filtered["data"].apply(extraer_carrier_de_data)
    df_filtered["carrier_code"] = carriers.apply(lambda x: x["carrier_code"])
    df_filtered["carrier_name"] = carriers.apply(lambda x: x["carrier_name"])
    df_filtered["year_month"]   = year_month

    # Una fila por orden — tomar el carrier más frecuente
    result = (
        df_filtered[df_filtered["carrier_code"].notna()]
        .groupby("logistic_order_id")
        .agg(
            carrier_code=("carrier_code", lambda x: x.value_counts().index[0]),
            carrier_name=("carrier_name", "first"),
            year_month=("year_month", "first")
        )
        .reset_index()
    )

    del df, df_filtered; gc.collect()

    elapsed = time.time() - t0
    print(f"{len(result):,} órdenes con carrier en {elapsed:.0f}s", flush=True)
    return result


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("EXTRACTOR DE CARRIER S3 → RDS", flush=True)
    print(f"Meses: {len(MESES)} | Evento: {EVENT_TYPE}", flush=True)
    print("="*60, flush=True)

    engine = get_engine()
    crear_tabla(engine)

    # Verificar si ya hay datos
    with engine.connect() as conn:
        n_exist = conn.execute(text(
            "SELECT COUNT(*) FROM staging_marts.stg_carrier"
        )).scalar()

    if n_exist > 0:
        print(f"\nYa hay {n_exist:,} filas en stg_carrier.", flush=True)
        resp = input("¿Truncar y recargar? (s/n): ").strip().lower()
        if resp == "s":
            with engine.connect() as conn:
                conn.execute(text("TRUNCATE staging_marts.stg_carrier"))
                conn.commit()
        else:
            print("Carga cancelada.", flush=True)
            return

    # Procesar mes a mes
    print(f"\nProcesando {len(MESES)} meses...", flush=True)
    total_ordenes = 0

    for mes in MESES:
        df_carrier = procesar_mes(mes)
        if len(df_carrier) == 0:
            continue

        # Insertar en RDS con ON CONFLICT para no duplicar
        try:
            eng = get_engine()
            # Insertar chunk por chunk
            for i in range(0, len(df_carrier), 10000):
                chunk = df_carrier.iloc[i:i+10000]
                chunk.to_sql(
                    "stg_carrier", eng,
                    schema="staging_marts",
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=2000,
                )
            eng.dispose()
            total_ordenes += len(df_carrier)
        except Exception as e:
            print(f"    Error insertando: {e}", flush=True)
            try:
                eng.dispose()
            except:
                pass

        del df_carrier; gc.collect()

    # Verificar resultado
    print(f"\n{'='*60}", flush=True)
    print("VERIFICACIÓN", flush=True)
    with engine.connect() as conn:
        df_check = pd.read_sql(text("""
            SELECT
                carrier_code,
                carrier_name,
                COUNT(*) as n_ordenes
            FROM staging_marts.stg_carrier
            GROUP BY carrier_code, carrier_name
            ORDER BY n_ordenes DESC
            LIMIT 15
        """), conn)
    print(df_check.to_string(index=False), flush=True)

    with engine.connect() as conn:
        total = conn.execute(text(
            "SELECT COUNT(*) FROM staging_marts.stg_carrier"
        )).scalar()

    total_min = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print("COMPLETADO", flush=True)
    print(f"  Total órdenes con carrier: {total:,}", flush=True)
    print(f"  Tiempo total:              {total_min:.1f} minutos", flush=True)
    print(f"  Tabla:                     staging_marts.stg_carrier", flush=True)


if __name__ == "__main__":
    main()
"""
Carga event_packages desde S3 a RDS PostgreSQL.

Lee los archivos Parquet de S3 mes a mes y los inserta en
staging_marts.stg_event_packages en partes (row groups) para no saturar la RAM.

Uso:
    python3 cargar_event_packages.py
"""

import os, gc, time, warnings, io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

MESES = [
    "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06",
    "2025-07","2025-08","2025-09","2025-10","2025-11","2025-12",
    "2026-01","2026-02","2026-03"
]

S3_BASE = "s3://falabella-data/processed/event_packages/country=CL"
COLS = [
    "logistic_order_id", "event_type", "event_dt", "country",
    "shipment_id", "package_id", "promised_date", "carrier",
    "tracking_number", "package_status", "year_month",
]


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


def crear_tabla(engine):
    """Crea la tabla si no existe."""
    print("Creando tabla stg_event_packages...", flush=True)
    q = text("""
        CREATE TABLE IF NOT EXISTS staging_marts.stg_event_packages (
            logistic_order_id  TEXT,
            event_type         TEXT,
            event_dt           TIMESTAMP,
            country            TEXT,
            shipment_id        TEXT,
            package_id         TEXT,
            promised_date      TEXT,
            carrier            FLOAT,
            tracking_number    FLOAT,
            package_status     TEXT,
            year_month         TEXT
        );
    """)
    with engine.connect() as conn:
        conn.execute(q)
        conn.commit()
    print("  Tabla creada OK", flush=True)


def crear_indices(engine):
    """Crea índices para JOINs rápidos."""
    print("\nCreando índices...", flush=True)
    indices = [
        "CREATE INDEX IF NOT EXISTS idx_ep_order_id ON staging_marts.stg_event_packages (logistic_order_id);",
        "CREATE INDEX IF NOT EXISTS idx_ep_year_month ON staging_marts.stg_event_packages (year_month);",
        "CREATE INDEX IF NOT EXISTS idx_ep_status ON staging_marts.stg_event_packages (package_status);",
        "CREATE INDEX IF NOT EXISTS idx_ep_event_dt ON staging_marts.stg_event_packages (event_dt);",
    ]
    with engine.connect() as conn:
        for idx in indices:
            conn.execute(text(idx))
        conn.commit()
    print("  Índices creados OK", flush=True)


def stream_parquet_safe(fpath: str, fs) -> pd.DataFrame:
    """
    Generador que lee un archivo parquet por partes (Row Groups) desde S3.
    Esto evita cargar archivos gigantescos enteros a la memoria RAM.
    """
    pf = pq.ParquetFile(fpath, filesystem=fs)
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i)
        # Normalización de compatibilidad de esquemas
        for name in t.schema.names:
            col = t.column(name)
            if pa.types.is_dictionary(col.type):
                t = t.set_column(t.schema.get_field_index(name), name, col.cast(pa.string()))
        
        df = t.to_pandas()
        yield df
        
        # Forzar liberación de memoria del bloque procesado
        del t, df
        gc.collect()


def copy_df_to_rds(df, engine):
    """Inserta un DataFrame en RDS vía COPY utilizando el engine existente."""
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="\\N")
        buf.seek(0)
        cursor.copy_expert(
            f"COPY staging_marts.stg_event_packages ({', '.join(COLS)}) FROM STDIN WITH CSV NULL '\\N'",
            buf,
        )
        raw_conn.commit()
        cursor.close()
    finally:
        raw_conn.close()


def cargar_mes(year_month: str, engine, fs) -> int:
    """Lee un mes desde S3 procesando los archivos en partes (row groups)."""
    s3_path = f"{S3_BASE}/year_month={year_month}/"
    t0 = time.time()

    bucket_prefix = s3_path.replace("s3://", "")
    try:
        archivos = sorted([f"s3://{f}" for f in fs.glob(f"{bucket_prefix}*.parquet")])
    except Exception as e:
        print(f"  {year_month} ERROR listando S3: {e}", flush=True)
        return 0

    if not archivos:
        print(f"  {year_month} 0 archivos", flush=True)
        return 0

    print(f"  {year_month} — {len(archivos)} archivos (procesando por partes)", flush=True)
    total = 0

    for i, fpath in enumerate(archivos, 1):
        try:
            # En lugar de leer todo el archivo, iteramos sobre sus partes (row groups)
            for df_chunk in stream_parquet_safe(fpath, fs):
                df_chunk["year_month"] = year_month
                if "event_dt" in df_chunk.columns:
                    df_chunk["event_dt"] = pd.to_datetime(df_chunk["event_dt"], errors="coerce")
                df_chunk = df_chunk.reindex(columns=COLS)

                copy_df_to_rds(df_chunk, engine)
                total += len(df_chunk)
                
                del df_chunk
                gc.collect()
                
        except Exception as e:
            print(f"    [{i}/{len(archivos)}] ERROR: {e}", flush=True)
            continue

        print(f"    [{i}/{len(archivos)}] {total:,} filas procesadas  {time.time()-t0:.0f}s", flush=True)

    print(f"  {year_month} LISTO: {total:,} filas en {time.time()-t0:.0f}s", flush=True)
    return total


def verificar_carga(engine):
    """Muestra resumen de lo cargado."""
    print("\n=== VERIFICACIÓN ===", flush=True)
    q = text("""
        SELECT
            year_month,
            COUNT(*) AS n,
            COUNT(DISTINCT logistic_order_id) AS ordenes,
            COUNT(DISTINCT package_status) AS estados
        FROM staging_marts.stg_event_packages
        GROUP BY year_month
        ORDER BY year_month
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    print(df.to_string(index=False), flush=True)


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("CARGA OPTIMIZADA event_packages S3 → RDS (Bajo Consumo de RAM)", flush=True)
    print(f"Meses: {len(MESES)} | Tamaño estimado: 7.2 GB", flush=True)
    print("="*60, flush=True)

    engine = get_engine()
    fs = s3fs.S3FileSystem()

    # Crear tabla
    crear_tabla(engine)

    # Verificar si ya hay datos
    with engine.connect() as conn:
        n_exist = conn.execute(text(
            "SELECT COUNT(*) FROM staging_marts.stg_event_packages"
        )).scalar()

    # Detectar meses ya cargados para no reprocesarlos
    meses_ya_cargados = set()
    if n_exist > 0:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT year_month FROM staging_marts.stg_event_packages"
            )).fetchall()
        meses_ya_cargados = {r[0] for r in rows}
        print(f"\nYa cargados: {sorted(meses_ya_cargados)}", flush=True)

    meses_pendientes = [m for m in MESES if m not in meses_ya_cargados]
    if not meses_pendientes:
        print("Todos los meses ya están cargados.", flush=True)
        verificar_carga(engine)
        return

    print(f"\nCargando {len(meses_pendientes)} meses pendientes desde S3...", flush=True)
    total_filas = 0
    for mes in meses_pendientes:
        n = cargar_mes(mes, engine, fs)
        total_filas += n
        gc.collect()

    # Crear índices
    crear_indices(engine)

    # Verificar
    verificar_carga(engine)

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"CARGA COMPLETADA", flush=True)
    print(f"  Total filas:   {total_filas:,}", flush=True)
    print(f"  Tiempo total:  {total:.1f} minutos", flush=True)


if __name__ == "__main__":
    main()
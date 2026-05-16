"""
Carga un mes de parquets de S3 a tablas raw en RDS PostgreSQL.

Uso:
    python load_month_to_rds.py 2025-01
    python load_month_to_rds.py 2025-01 2025-02
    python load_month_to_rds.py --all
"""
import os
import sys
import time
import boto3
import duckdb
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

load_dotenv("/home/ec2-user/Inicio_falabella/.env")

BUCKET = "falabella-data"
REGION = "us-east-1"
COUNTRY = "CL"

TABLAS = {
    "events":         f"s3://{BUCKET}/processed/events/country={COUNTRY}/year_month={{mes}}/data.parquet",
    "event_items":    f"s3://{BUCKET}/processed/event_items/country={COUNTRY}/year_month={{mes}}/data.parquet",
    "event_packages": f"s3://{BUCKET}/processed/event_packages/country={COUNTRY}/year_month={{mes}}/data.parquet",
}

MESES_TODOS = [
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03",
]


def get_duckdb_con():
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_region='{REGION}';")
    con.execute(f"SET s3_access_key_id='{creds.access_key}';")
    con.execute(f"SET s3_secret_access_key='{creds.secret_key}';")
    if creds.token:
        con.execute(f"SET s3_session_token='{creds.token}';")
    con.execute("SET memory_limit='6GB';")
    return con


def get_pg_con():
    return psycopg2.connect(
        host=os.environ["DBT_DB_HOST"],
        port=5432,
        dbname=os.environ["DBT_DB_NAME"],
        user=os.environ["DBT_DB_USER"],
        password=os.environ["DBT_DB_PASSWORD"],
        sslmode="require",
    )


def ensure_raw_schema(pg_con):
    with pg_con.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    pg_con.commit()


def duck_schema_to_pg(duck_con, parquet_path):
    cols_info = duck_con.sql(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").fetchall()
    pg_cols = []
    for row in cols_info:
        name = row[0]
        duck_type = row[1].upper()
        if any(t in duck_type for t in ["MAP", "STRUCT", "LIST", "[]"]):
            pg_type = "JSONB"
        elif "VARCHAR" in duck_type or "STRING" in duck_type:
            pg_type = "TEXT"
        elif "BIGINT" in duck_type or "INT64" in duck_type:
            pg_type = "BIGINT"
        elif "INTEGER" in duck_type or "INT" in duck_type:
            pg_type = "INTEGER"
        elif "DOUBLE" in duck_type or "FLOAT" in duck_type or "DECIMAL" in duck_type:
            pg_type = "DOUBLE PRECISION"
        elif "BOOLEAN" in duck_type:
            pg_type = "BOOLEAN"
        elif "TIMESTAMP" in duck_type:
            pg_type = "TIMESTAMP"
        elif "DATE" in duck_type:
            pg_type = "DATE"
        else:
            pg_type = "TEXT"
        pg_cols.append(f'"{name}" {pg_type}')
    return pg_cols


def ensure_table(pg_con, duck_con, table_name, parquet_path):
    pg_cols = duck_schema_to_pg(duck_con, parquet_path)
    ddl = f'CREATE TABLE IF NOT EXISTS raw.{table_name} (\n  ' + ",\n  ".join(pg_cols) + "\n);"
    with pg_con.cursor() as cur:
        cur.execute(ddl)
    pg_con.commit()


def delete_month(pg_con, table_name, mes):
    with pg_con.cursor() as cur:
        cur.execute(f'DELETE FROM raw.{table_name} WHERE year_month = %s;', (mes,))
        deleted = cur.rowcount
    pg_con.commit()
    return deleted


def load_month_table(pg_con, duck_con, table_name, mes):
    parquet_path = TABLAS[table_name].format(mes=mes)
    t0 = time.time()

    ensure_table(pg_con, duck_con, table_name, parquet_path)
    n_parquet = duck_con.sql(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()[0]
    deleted = delete_month(pg_con, table_name, mes)
    if deleted > 0:
        print(f"    [{table_name}] Borradas {deleted:,} filas previas del mes {mes}", flush=True)

    cols_info = duck_con.sql(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").fetchall()
    col_names = [c[0] for c in cols_info]
    col_types = {c[0]: c[1].upper() for c in cols_info}

    select_exprs = []
    for cname in col_names:
        ctype = col_types[cname]
        if any(t in ctype for t in ["MAP", "STRUCT", "LIST", "[]"]):
            select_exprs.append(f'TO_JSON("{cname}")::VARCHAR AS "{cname}"')
        else:
            select_exprs.append(f'"{cname}"')
    select_sql = ", ".join(select_exprs)

    csv_path = f"/tmp/{table_name}_{mes}.csv"
    duck_con.sql(f"""
        COPY (SELECT {select_sql} FROM read_parquet('{parquet_path}'))
        TO '{csv_path}' (HEADER FALSE, DELIMITER E'\t', QUOTE '"', ESCAPE '"', NULL '\\N');
    """)

    cols_quoted = ", ".join([f'"{c}"' for c in col_names])
    with pg_con.cursor() as cur, open(csv_path, "r") as f:
        cur.copy_expert(
            f"COPY raw.{table_name} ({cols_quoted}) FROM STDIN "
            f"WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N', QUOTE '\"', ESCAPE '\"')",
            f,
        )
    pg_con.commit()

    with pg_con.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM raw.{table_name} WHERE year_month = %s;', (mes,))
        n_pg = cur.fetchone()[0]

    Path(csv_path).unlink(missing_ok=True)
    elapsed = time.time() - t0
    match = "OK" if n_pg == n_parquet else "MISMATCH"
    print(f"    [{table_name}] {n_parquet:,} en parquet, {n_pg:,} en Postgres ({elapsed:.1f}s) [{match}]", flush=True)
    return n_pg == n_parquet


def load_month(mes):
    print(f"\n=== Cargando {mes} ===", flush=True)
    t0 = time.time()
    duck_con = get_duckdb_con()
    pg_con = get_pg_con()
    ensure_raw_schema(pg_con)

    all_ok = True
    for table_name in TABLAS.keys():
        try:
            ok = load_month_table(pg_con, duck_con, table_name, mes)
            all_ok = all_ok and ok
        except Exception as e:
            print(f"    [{table_name}] ERROR: {e}", flush=True)
            all_ok = False

    duck_con.close()
    pg_con.close()
    elapsed = time.time() - t0
    status = "OK" if all_ok else "CON ERRORES"
    print(f"=== {mes} terminado en {elapsed:.1f}s [{status}] ===", flush=True)
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--all":
        meses = MESES_TODOS
    else:
        meses = sys.argv[1:]

    print(f"Voy a cargar {len(meses)} mes(es): {meses}", flush=True)
    inicio = time.time()
    resultados = {}
    for mes in meses:
        resultados[mes] = load_month(mes)

    print("\n" + "=" * 50, flush=True)
    print("RESUMEN FINAL", flush=True)
    print("=" * 50, flush=True)
    for mes, ok in resultados.items():
        print(f"  {mes}: {'OK' if ok else 'FALLO'}", flush=True)
    print(f"\nTiempo total: {(time.time() - inicio) / 60:.1f} minutos", flush=True)
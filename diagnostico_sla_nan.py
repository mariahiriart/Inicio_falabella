"""
Diagnóstico de NaN en sla_horas_prometidas.

Objetivo: entender de dónde vienen los 1.2M de filas descartadas en val
al calcular desvio_sla = hours_total - sla_horas_prometidas.

Preguntas a responder:
  1. ¿Cuántas órdenes tienen sla_horas_prometidas NULL, cero o negativa?
  2. ¿Están concentradas en alguna service_category o split?
  3. ¿Son las mismas órdenes sin service_category del análisis de negocio?
  4. ¿Cuántas órdenes tendrían desvio_sla válido si filtramos correctamente?

Uso:
    python diagnostico_sla_nan.py
"""

import os
import json
import time
import warnings
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True)


def q1_resumen_global(engine):
    """
    Cuenta cuántas órdenes del dataset ML (con ciclo completo)
    tienen sla_horas_prometidas en cada estado: NULL, <= 0, o válida.
    """
    print("\n" + "="*60, flush=True)
    print("Q1 — Estado de sla_horas_prometidas en el dataset ML", flush=True)
    print("="*60, flush=True)

    query = text("""
        SELECT
            split_set,
            COUNT(*)                                                        AS total,
            COUNT(*) FILTER (WHERE sla_horas_prometidas IS NULL)           AS sla_null,
            COUNT(*) FILTER (WHERE sla_horas_prometidas IS NOT NULL
                               AND sla_horas_prometidas <= 0)              AS sla_zero_neg,
            COUNT(*) FILTER (WHERE sla_horas_prometidas > 0)               AS sla_valida,
            ROUND(100.0 * COUNT(*) FILTER (WHERE sla_horas_prometidas IS NULL)
                  / COUNT(*), 2)                                            AS pct_null,
            ROUND(AVG(sla_horas_prometidas) FILTER
                  (WHERE sla_horas_prometidas > 0), 1)                     AS media_sla_valida,
            ROUND(MIN(sla_horas_prometidas), 1)                            AS min_sla,
            ROUND(MAX(sla_horas_prometidas), 1)                            AS max_sla
        FROM staging_marts.ml_dataset_v1
        WHERE ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
        GROUP BY split_set
        ORDER BY split_set
    """)

    t0 = time.time()
    df = pd.read_sql(query, engine)
    print(f"  Query en {time.time()-t0:.1f}s\n", flush=True)
    print(df.to_string(index=False), flush=True)
    return df


def q2_nan_por_categoria(engine):
    """
    Cruce service_category × split_set para ver si los NaN
    están concentrados en categorías específicas.
    """
    print("\n" + "="*60, flush=True)
    print("Q2 — NaN en SLA por service_category × split", flush=True)
    print("="*60, flush=True)

    query = text("""
        SELECT
            COALESCE(service_category, '(NULL)')    AS service_category,
            split_set,
            COUNT(*)                                AS total,
            COUNT(*) FILTER (WHERE sla_horas_prometidas IS NULL
                              OR sla_horas_prometidas <= 0) AS sla_invalida,
            ROUND(100.0 * COUNT(*) FILTER (WHERE sla_horas_prometidas IS NULL
                              OR sla_horas_prometidas <= 0) / COUNT(*), 1) AS pct_invalida
        FROM staging_marts.ml_dataset_v1
        WHERE ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
        GROUP BY service_category, split_set
        ORDER BY service_category, split_set
    """)

    t0 = time.time()
    df = pd.read_sql(query, engine)
    print(f"  Query en {time.time()-t0:.1f}s\n", flush=True)
    print(df.to_string(index=False), flush=True)
    return df


def q3_nan_por_seller(engine):
    """
    Top sellers con más NaN en SLA — para ver si el problema
    está concentrado en vendedores específicos.
    """
    print("\n" + "="*60, flush=True)
    print("Q3 — Top 15 sellers con más SLA inválido", flush=True)
    print("="*60, flush=True)

    query = text("""
        SELECT
            seller_id,
            COUNT(*)                                                        AS total,
            COUNT(*) FILTER (WHERE sla_horas_prometidas IS NULL
                              OR sla_horas_prometidas <= 0)                AS sla_invalida,
            ROUND(100.0 * COUNT(*) FILTER (WHERE sla_horas_prometidas IS NULL
                              OR sla_horas_prometidas <= 0) / COUNT(*), 1) AS pct_invalida,
            COALESCE(service_category, '(NULL)')                           AS categoria_mas_frecuente
        FROM staging_marts.ml_dataset_v1
        WHERE ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
          AND (sla_horas_prometidas IS NULL OR sla_horas_prometidas <= 0)
        GROUP BY seller_id, service_category
        ORDER BY sla_invalida DESC
        LIMIT 15
    """)

    t0 = time.time()
    df = pd.read_sql(query, engine)
    print(f"  Query en {time.time()-t0:.1f}s\n", flush=True)
    print(df.to_string(index=False), flush=True)
    return df


def q4_impacto_filtro(engine):
    """
    Simula cuántas filas quedarían si filtramos
    sla_horas_prometidas > 0 en el entrenamiento del modelo de desvío.
    Muestra el trade-off: perdemos filas pero ganamos un target limpio.
    """
    print("\n" + "="*60, flush=True)
    print("Q4 — Impacto de filtrar sla > 0 en el dataset ML", flush=True)
    print("="*60, flush=True)

    query = text("""
        SELECT
            split_set,
            COUNT(*)                                                   AS filas_actuales,
            COUNT(*) FILTER (WHERE sla_horas_prometidas > 0)           AS filas_con_sla_valido,
            ROUND(100.0 * COUNT(*) FILTER (WHERE sla_horas_prometidas > 0)
                  / COUNT(*), 1)                                        AS pct_retencion,
            -- Distribución del desvío para las filas con SLA válido
            ROUND(AVG(target_horas_totales - sla_horas_prometidas)
                  FILTER (WHERE sla_horas_prometidas > 0), 1)          AS desvio_medio_h,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY target_horas_totales - sla_horas_prometidas)
                  FILTER (WHERE sla_horas_prometidas > 0), 1)          AS desvio_mediana_h,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
                  (ORDER BY target_horas_totales - sla_horas_prometidas)
                  FILTER (WHERE sla_horas_prometidas > 0), 1)          AS desvio_p90_h
        FROM staging_marts.ml_dataset_v1
        WHERE ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
        GROUP BY split_set
        ORDER BY split_set
    """)

    t0 = time.time()
    df = pd.read_sql(query, engine)
    print(f"  Query en {time.time()-t0:.1f}s\n", flush=True)
    print(df.to_string(index=False), flush=True)

    # Resumen interpretativo
    print("\n  Interpretación del desvío (filas con SLA válido):", flush=True)
    print("  - Desvío positivo = orden tardó más de lo prometido (late)", flush=True)
    print("  - Desvío negativo = orden llegó antes de lo prometido (early)", flush=True)
    print("  - Desvío = 0 = llegó exactamente en el SLA prometido", flush=True)
    return df


def q5_distribucion_desvio(engine):
    """
    Distribución del desvío para las filas válidas — ver si tiene
    sentido como target de regresión o si necesita transformación.
    """
    print("\n" + "="*60, flush=True)
    print("Q5 — Distribución del desvío (filas con SLA > 0)", flush=True)
    print("="*60, flush=True)

    query = text("""
        SELECT
            CASE
                WHEN (target_horas_totales - sla_horas_prometidas) < -24  THEN 'muy_early (< -24h)'
                WHEN (target_horas_totales - sla_horas_prometidas) < 0    THEN 'early (0 a -24h)'
                WHEN (target_horas_totales - sla_horas_prometidas) = 0    THEN 'exacto'
                WHEN (target_horas_totales - sla_horas_prometidas) <= 24  THEN 'leve_tarde (0-24h)'
                WHEN (target_horas_totales - sla_horas_prometidas) <= 72  THEN 'tarde (24-72h)'
                ELSE 'muy_tarde (> 72h)'
            END                                                            AS categoria_desvio,
            COUNT(*)                                                       AS n,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)            AS pct
        FROM staging_marts.ml_dataset_v1
        WHERE ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
          AND sla_horas_prometidas > 0
        GROUP BY 1
        ORDER BY
            CASE categoria_desvio
                WHEN 'muy_early (< -24h)' THEN 1
                WHEN 'early (0 a -24h)'   THEN 2
                WHEN 'exacto'             THEN 3
                WHEN 'leve_tarde (0-24h)' THEN 4
                WHEN 'tarde (24-72h)'     THEN 5
                ELSE 6
            END
    """)

    t0 = time.time()
    df = pd.read_sql(query, engine)
    print(f"  Query en {time.time()-t0:.1f}s\n", flush=True)
    print(df.to_string(index=False), flush=True)
    return df


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("DIAGNÓSTICO — NaN en sla_horas_prometidas", flush=True)
    print("="*60, flush=True)

    engine = get_engine()

    df_q1 = q1_resumen_global(engine)
    df_q2 = q2_nan_por_categoria(engine)
    df_q3 = q3_nan_por_seller(engine)
    df_q4 = q4_impacto_filtro(engine)
    df_q5 = q5_distribucion_desvio(engine)

    # Guardar todo en JSON
    out = {
        "resumen_global":       df_q1.to_dict(orient="records"),
        "nan_por_categoria":    df_q2.to_dict(orient="records"),
        "top_sellers_nan":      df_q3.to_dict(orient="records"),
        "impacto_filtro":       df_q4.to_dict(orient="records"),
        "distribucion_desvio":  df_q5.to_dict(orient="records"),
    }
    out_path = OUTPUT_DIR / "diagnostico_sla_nan.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResultados guardados en: {out_path}", flush=True)

    total = (time.time() - t_inicio) / 60
    print(f"\nDiagnóstico completado en {total:.1f} minutos", flush=True)
    print("\nQué mirar:", flush=True)
    print("  Q1: ¿qué % del val tiene SLA NULL? Si es ~43% explica los 1.2M", flush=True)
    print("  Q2: ¿los NULL están concentrados en (NULL) service_category?", flush=True)
    print("  Q3: ¿hay algún seller que explique la mayoría de los NULL?", flush=True)
    print("  Q4: ¿cuántas filas quedan tras filtrar sla > 0? ¿es suficiente?", flush=True)
    print("  Q5: ¿el desvío tiene una forma razonable para regresión?", flush=True)


if __name__ == "__main__":
    main()

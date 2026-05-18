"""
Diagnóstico paso 1 — Distribución de hours_total por clase.

Objetivo: confirmar que la población OK y la población DISRUPTED tienen
distribuciones de hours_total tan distintas que un solo regresor no puede
capturar ambas, lo que explica el R² bajo del modelo v1.

Salidas:
  - Tabla en stdout con percentiles por clase
  - Tabla por service_category × is_disrupted
  - diagnostico_distribucion.json con los números crudos
  - histograma_hours_total.png con los dos histogramas superpuestos

Uso:
    python diagnostico_distribucion.py
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text

import matplotlib
matplotlib.use("Agg")  # sin display en EC2
import matplotlib.pyplot as plt

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


def percentiles_por_clase(engine):
    """
    Percentiles de hours_total separados por is_disrupted, usando
    SOLO órdenes del dataset ML con ciclo completo (las mismas que entrarían
    a la regresión).
    """
    print("\n" + "="*60, flush=True)
    print("PASO 1.A — Percentiles de hours_total por clase", flush=True)
    print("="*60, flush=True)

    query = text("""
        SELECT
            target_binario                                                    AS is_disrupted,
            COUNT(*)                                                          AS n,
            AVG(target_horas_totales)                                         AS media,
            MIN(target_horas_totales)                                         AS min_h,
            percentile_cont(0.05) WITHIN GROUP (ORDER BY target_horas_totales) AS p05,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY target_horas_totales) AS p25,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY target_horas_totales) AS p50,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY target_horas_totales) AS p75,
            percentile_cont(0.90) WITHIN GROUP (ORDER BY target_horas_totales) AS p90,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY target_horas_totales) AS p95,
            percentile_cont(0.99) WITHIN GROUP (ORDER BY target_horas_totales) AS p99,
            MAX(target_horas_totales)                                         AS max_h,
            STDDEV(target_horas_totales)                                      AS std_h
        FROM staging_marts.ml_dataset_v1
        WHERE ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
        GROUP BY target_binario
        ORDER BY target_binario
    """)

    t0 = time.time()
    df = pd.read_sql(query, engine)
    print(f"  Query ejecutada en {time.time()-t0:.1f}s", flush=True)

    # Renombrar para legibilidad
    df["clase"] = df["is_disrupted"].map({0: "OK", 1: "DISRUPTED"})

    print("\n  Distribución de hours_total por clase:", flush=True)
    print("-" * 60, flush=True)
    cols_mostrar = ["clase", "n", "media", "p05", "p25", "p50", "p75", "p90", "p95", "p99", "max_h", "std_h"]
    print(df[cols_mostrar].to_string(index=False, float_format=lambda x: f"{x:,.1f}"), flush=True)

    return df


def percentiles_por_categoria_y_clase(engine):
    """
    Cruce service_category × is_disrupted para ver si la diferencia
    se concentra en ciertas categorías o es generalizada.
    """
    print("\n" + "="*60, flush=True)
    print("PASO 1.B — hours_total por service_category × clase", flush=True)
    print("="*60, flush=True)

    query = text("""
        SELECT
            COALESCE(service_category, '(NULL)')                              AS service_category,
            target_binario                                                    AS is_disrupted,
            COUNT(*)                                                          AS n,
            AVG(target_horas_totales)                                         AS media,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY target_horas_totales) AS p50,
            percentile_cont(0.90) WITHIN GROUP (ORDER BY target_horas_totales) AS p90,
            percentile_cont(0.99) WITHIN GROUP (ORDER BY target_horas_totales) AS p99
        FROM staging_marts.ml_dataset_v1
        WHERE ciclo_completo = 1
          AND target_horas_totales IS NOT NULL
          AND target_horas_totales > 0
        GROUP BY service_category, target_binario
        ORDER BY service_category, target_binario
    """)

    t0 = time.time()
    df = pd.read_sql(query, engine)
    print(f"  Query ejecutada en {time.time()-t0:.1f}s", flush=True)

    df["clase"] = df["is_disrupted"].map({0: "OK", 1: "DISRUPTED"})

    print("\n  Media y percentiles por categoría:", flush=True)
    print("-" * 80, flush=True)
    print(df[["service_category", "clase", "n", "media", "p50", "p90", "p99"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.1f}"), flush=True)

    # Calcular ratio media DISRUPTED / OK por categoría
    print("\n  Ratio media DISRUPTED / OK por categoría (cuántas veces más tarda una orden rota):", flush=True)
    print("-" * 80, flush=True)
    pivot = df.pivot_table(index="service_category", columns="clase",
                           values="media", aggfunc="first")
    if "OK" in pivot.columns and "DISRUPTED" in pivot.columns:
        pivot["ratio_disrupted_ok"] = pivot["DISRUPTED"] / pivot["OK"]
        print(pivot.to_string(float_format=lambda x: f"{x:,.2f}"), flush=True)

    return df


def histograma(engine):
    """
    Histograma de hours_total para las dos clases superpuestas.
    Usa una muestra aleatoria para que no exploten los plots.
    """
    print("\n" + "="*60, flush=True)
    print("PASO 1.C — Histograma de hours_total (muestra de 500k por clase)", flush=True)
    print("="*60, flush=True)

    query = text("""
        WITH muestra AS (
            SELECT
                target_horas_totales AS hours_total,
                target_binario       AS is_disrupted,
                ROW_NUMBER() OVER (PARTITION BY target_binario ORDER BY RANDOM()) AS rn
            FROM staging_marts.ml_dataset_v1
            WHERE ciclo_completo = 1
              AND target_horas_totales IS NOT NULL
              AND target_horas_totales > 0
        )
        SELECT hours_total, is_disrupted
        FROM muestra
        WHERE rn <= 500000
    """)

    t0 = time.time()
    df = pd.read_sql(query, engine)
    print(f"  Muestra cargada en {time.time()-t0:.1f}s — {len(df):,} filas", flush=True)

    ok        = df.loc[df["is_disrupted"] == 0, "hours_total"].values
    disrupted = df.loc[df["is_disrupted"] == 1, "hours_total"].values

    # Para el plot recortamos al p99 global para que no se vea aplastado
    p99_global = np.percentile(df["hours_total"].values, 99)
    print(f"  p99 global = {p99_global:.1f}h — recorto plot a este valor", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: escala lineal recortada a p99
    axes[0].hist(np.clip(ok, 0, p99_global), bins=80, alpha=0.6,
                 label=f"OK (n={len(ok):,})", color="#2ecc71")
    axes[0].hist(np.clip(disrupted, 0, p99_global), bins=80, alpha=0.6,
                 label=f"DISRUPTED (n={len(disrupted):,})", color="#e74c3c")
    axes[0].set_xlabel("hours_total (recortado a p99 global)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución de hours_total — escala lineal")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: escala log para ver mejor las dos distribuciones
    axes[1].hist(np.log1p(ok), bins=80, alpha=0.6,
                 label=f"OK (n={len(ok):,})", color="#2ecc71")
    axes[1].hist(np.log1p(disrupted), bins=80, alpha=0.6,
                 label=f"DISRUPTED (n={len(disrupted):,})", color="#e74c3c")
    axes[1].set_xlabel("log1p(hours_total)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución en escala log — donde entrena el modelo")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "histograma_hours_total.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Histograma guardado en: {out_path}", flush=True)

    # Resumen numérico de la muestra
    print("\n  Resumen numérico de la muestra:", flush=True)
    print(f"    OK        — media: {ok.mean():.1f}h  mediana: {np.median(ok):.1f}h  std: {ok.std():.1f}h", flush=True)
    print(f"    DISRUPTED — media: {disrupted.mean():.1f}h  mediana: {np.median(disrupted):.1f}h  std: {disrupted.std():.1f}h", flush=True)
    print(f"    Ratio medias DISRUPTED/OK: {disrupted.mean()/ok.mean():.2f}x", flush=True)


def main():
    t_inicio = time.time()
    print("="*60, flush=True)
    print("DIAGNÓSTICO — Distribución de hours_total por clase", flush=True)
    print("="*60, flush=True)
    print("Hipótesis a verificar: las poblaciones OK y DISRUPTED tienen", flush=True)
    print("distribuciones tan distintas que un solo regresor no puede capturarlas.", flush=True)

    engine = get_engine()

    df_clase = percentiles_por_clase(engine)
    df_cat   = percentiles_por_categoria_y_clase(engine)
    histograma(engine)

    # Guardar números crudos
    out_json = OUTPUT_DIR / "diagnostico_distribucion.json"
    payload = {
        "percentiles_por_clase":          df_clase.to_dict(orient="records"),
        "percentiles_por_cat_y_clase":    df_cat.to_dict(orient="records"),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Resultados crudos guardados en: {out_json}", flush=True)

    total = (time.time() - t_inicio) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"Diagnóstico completado en {total:.1f} minutos", flush=True)
    print(f"{'='*60}", flush=True)
    print("\nQué mirar en el output:", flush=True)
    print("  1. ¿La mediana (p50) de DISRUPTED es mucho mayor que la de OK?", flush=True)
    print("  2. ¿La cola alta (p95, p99) está concentrada en DISRUPTED?", flush=True)
    print("  3. En el histograma log, ¿se ven dos modas claramente separadas?", flush=True)
    print("\nSi las respuestas son SÍ, confirmamos la hipótesis y pasamos al paso 2.", flush=True)


if __name__ == "__main__":
    main()

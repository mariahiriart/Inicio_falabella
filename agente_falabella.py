"""
Agente de Telegram — Predictor de disrupciones Falabella Chile.
Cerebro: Kimi (Moonshot AI) via OpenAI SDK
Herramientas: modelo ML v2 + consultas RDS

El agente entiende lenguaje natural, extrae los datos de la orden,
busca el histórico en RDS y predice la disrupción con explicación.
También puede consultar el recorrido completo de una orden específica
usando su logistic_order_id, mostrando todos los eventos y estados
por los que pasó la orden.

Uso:
    pip3 install python-telegram-bot openai
    python3 agente_falabella_kimi.py

Requiere:
    - model_lgbm_v2.pkl en ml_outputs/
    - .env con credenciales RDS y KIMI_API_KEY
"""

import os, json, pickle, logging, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

# ── Config ───────────────────────────────────────────────────────────────────

TOKEN      = "8836242266:AAHDQGPqlsOFGJFXQu1M_yPt-jz8Wv7PfQs"
OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
MODEL_PATH = OUTPUT_DIR / "model_lgbm_v2.pkl"
KIMI_KEY   = os.environ.get("KIMI_API_KEY", "")

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

DIAS = ["Domingo","Lunes","Martes","Miércoles","Jueves","Viernes","Sábado"]

# ── Cargar modelo ML ──────────────────────────────────────────────────────────

print("Cargando modelo ML v2...", flush=True)
with open(MODEL_PATH, "rb") as f:
    artefacto = pickle.load(f)
MODEL    = artefacto["model"]
ENCODERS = artefacto["encoders"]
FEATURES = artefacto["features"]
print(f"Modelo cargado. Features: {len(FEATURES)}", flush=True)

# ── RDS ──────────────────────────────────────────────────────────────────────

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


# ── Herramientas del agente ───────────────────────────────────────────────────

def buscar_historico_seller(seller_id: str, service_category: str) -> dict:
    """
    Busca tasas históricas de disrupción para un seller y categoría.
    Devuelve promedios globales si no encuentra el seller específico.
    """
    try:
        engine = get_engine()

        # Buscar por seller específico
        q = text("""
            SELECT
                COUNT(*)                          AS n_ordenes,
                AVG(h.seller_tasa_disrupcion)     AS seller_tasa,
                AVG(h.categoria_tasa_disrupcion)  AS cat_tasa,
                AVG(h.categoria_cac_tasa_disrupcion) AS cat_cac_tasa,
                AVG(h.dia_semana_tasa_disrupcion) AS dia_tasa,
                AVG(h.franja_tasa_disrupcion)     AS franja_tasa,
                MAX(h.seller_n_ordenes)           AS seller_n_ordenes
            FROM staging_marts.ml_historical_features h
            JOIN staging_marts.ml_dataset_v1 d
              ON h.logistic_order_id = d.logistic_order_id
            WHERE d.seller_id = :seller
              AND d.service_category = :cat
        """)
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={
                "seller": seller_id, "cat": service_category
            })
        engine.dispose()

        if len(df) > 0 and df.iloc[0]["n_ordenes"] and df.iloc[0]["n_ordenes"] > 0:
            row = df.iloc[0]
            return {
                "encontrado":     True,
                "fuente":         f"seller {seller_id} + categoría {service_category}",
                "n_ordenes":      int(row["n_ordenes"]),
                "seller_tasa":    float(row["seller_tasa"] or 0.4),
                "cat_tasa":       float(row["cat_tasa"] or 0.4),
                "cat_cac_tasa":   float(row["cat_cac_tasa"] or 0.4),
                "dia_tasa":       float(row["dia_tasa"] or 0.4),
                "franja_tasa":    float(row["franja_tasa"] or 0.4),
                "seller_n_ordenes": float(row["seller_n_ordenes"] or 0),
            }

        # Fallback: solo por categoría
        q2 = text("""
            SELECT
                COUNT(*)                          AS n_ordenes,
                AVG(h.categoria_tasa_disrupcion)  AS cat_tasa,
                AVG(h.categoria_cac_tasa_disrupcion) AS cat_cac_tasa
            FROM staging_marts.ml_historical_features h
            JOIN staging_marts.ml_dataset_v1 d
              ON h.logistic_order_id = d.logistic_order_id
            WHERE d.service_category = :cat
        """)
        with engine.connect() as conn:
            df2 = pd.read_sql(q2, conn, params={"cat": service_category})
        engine.dispose()

        if len(df2) > 0:
            row2 = df2.iloc[0]
            return {
                "encontrado":     False,
                "fuente":         f"promedio categoría {service_category} (seller nuevo)",
                "n_ordenes":      int(row2["n_ordenes"] or 0),
                "seller_tasa":    0.40,
                "cat_tasa":       float(row2["cat_tasa"] or 0.4),
                "cat_cac_tasa":   float(row2["cat_cac_tasa"] or 0.4),
                "dia_tasa":       0.42,
                "franja_tasa":    0.40,
                "seller_n_ordenes": 0,
            }

    except Exception as e:
        logger.error(f"Error buscando histórico: {e}")

    # Fallback global
    return {
        "encontrado":     False,
        "fuente":         "promedios globales (seller y categoría nuevos)",
        "n_ordenes":      0,
        "seller_tasa":    0.39,
        "cat_tasa":       0.39,
        "cat_cac_tasa":   0.39,
        "dia_tasa":       0.42,
        "franja_tasa":    0.40,
        "seller_n_ordenes": 0,
    }


def buscar_tramo_historico(seller_id: str, service_category: str) -> dict:
    """
    Busca en qué tramo históricamente fallan las órdenes de este seller+categoría.
    """
    try:
        engine = get_engine()
        q = text("""
            SELECT
                f.tramo_disruptivo,
                f.actor_responsable,
                COUNT(*) AS n
            FROM staging_marts.fct_order_disruptions f
            JOIN staging_marts.fct_orders o
              ON f.logistic_order_id = o.logistic_order_id
            WHERE o.seller_id = :seller
              AND o.service_category = :cat
              AND f.tramo_disruptivo != 'sin_disrupcion'
            GROUP BY f.tramo_disruptivo, f.actor_responsable
            ORDER BY n DESC
            LIMIT 5
        """)
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={
                "seller": seller_id, "cat": service_category
            })
        engine.dispose()

        if len(df) == 0:
            # Fallback por categoría
            q2 = text("""
                SELECT
                    f.tramo_disruptivo,
                    f.actor_responsable,
                    COUNT(*) AS n
                FROM staging_marts.fct_order_disruptions f
                JOIN staging_marts.fct_orders o
                  ON f.logistic_order_id = o.logistic_order_id
                WHERE o.service_category = :cat
                  AND f.tramo_disruptivo != 'sin_disrupcion'
                GROUP BY f.tramo_disruptivo, f.actor_responsable
                ORDER BY n DESC
                LIMIT 5
            """)
            with engine.connect() as conn:
                df = pd.read_sql(q2, conn, params={"cat": service_category})
            engine.dispose()

        if len(df) > 0:
            total = df["n"].sum()
            tramos = []
            for _, row in df.iterrows():
                tramos.append({
                    "tramo":  row["tramo_disruptivo"],
                    "actor":  row["actor_responsable"],
                    "n":      int(row["n"]),
                    "pct":    round(100 * int(row["n"]) / total, 1)
                })
            return {"tramos": tramos, "total_disrupciones": int(total)}

    except Exception as e:
        logger.error(f"Error buscando tramo: {e}")

    return {"tramos": [], "total_disrupciones": 0}


def consultar_recorrido_orden(logistic_order_id: str) -> dict:
    """Consulta el recorrido completo de una orden agrupado por paquete."""
    try:
        engine = get_engine()
        q = text("""
            SELECT
                package_id,
                shipment_id,
                promised_date,
                package_status,
                event_type,
                event_dt,
                EXTRACT(EPOCH FROM (event_dt - LAG(event_dt) OVER (
                    PARTITION BY logistic_order_id, package_id ORDER BY event_dt
                ))) / 3600 AS horas_desde_anterior
            FROM staging_marts.stg_event_packages
            WHERE logistic_order_id = :oid
            ORDER BY package_id, event_dt
        """)
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"oid": logistic_order_id})
        engine.dispose()

        if len(df) == 0:
            return {"encontrado": False, "mensaje": f"No se encontraron eventos para {logistic_order_id}"}

        paquetes = {}
        for _, row in df.iterrows():
            pkg_id = row["package_id"] or "SIN_PACKAGE_ID"
            if pkg_id not in paquetes:
                paquetes[pkg_id] = {
                    "package_id":    pkg_id,
                    "shipment_id":   row["shipment_id"],
                    "promised_date": str(row["promised_date"]) if row["promised_date"] else None,
                    "eventos":       [],
                    "ultimo_estado": None,
                    "horas_max_gap": 0,
                }
            gap = float(row["horas_desde_anterior"]) if row["horas_desde_anterior"] and not pd.isna(row["horas_desde_anterior"]) else None
            paquetes[pkg_id]["eventos"].append({
                "fecha":                str(row["event_dt"]),
                "event_type":           row["event_type"],
                "estado":               row["package_status"],
                "horas_desde_anterior": round(gap, 1) if gap else None,
            })
            paquetes[pkg_id]["ultimo_estado"] = row["package_status"]
            if gap and gap > paquetes[pkg_id]["horas_max_gap"]:
                paquetes[pkg_id]["horas_max_gap"] = round(gap, 1)

        return {
            "encontrado":        True,
            "logistic_order_id": logistic_order_id,
            "n_paquetes":        len(paquetes),
            "paquetes":          list(paquetes.values()),
        }
    except Exception as e:
        return {"encontrado": False, "error": str(e)}


def buscar_orden_por_id(logistic_order_id: str) -> dict:
    """Busca una orden específica por ID en RDS y devuelve todos sus datos."""
    try:
        engine = get_engine()
        q = text("""
            WITH raw_eventos AS (
                SELECT
                    logistic_order_id,
                    package_status,
                    event_dt,
                    LAG(event_dt) OVER (
                        PARTITION BY logistic_order_id ORDER BY event_dt
                    ) AS prev_dt,
                    LAST_VALUE(package_status) OVER (
                        PARTITION BY logistic_order_id ORDER BY event_dt
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS ultimo_estado
                FROM staging_marts.stg_event_packages
                WHERE logistic_order_id = :oid
            ),
            ep AS (
                SELECT
                    logistic_order_id,
                    COUNT(*)                                                                AS n_eventos,
                    COUNT(DISTINCT package_status)                                          AS n_estados_distintos,
                    MAX(CASE WHEN package_status = 'DELIVERY_ATTEMPTED' THEN 1 ELSE 0 END) AS tuvo_delivery_attempted,
                    MAX(CASE WHEN package_status = 'EXCEPTION'          THEN 1 ELSE 0 END) AS tuvo_exception,
                    MAX(CASE WHEN package_status = 'ANNULLED'           THEN 1 ELSE 0 END) AS tuvo_annulled,
                    MAX(EXTRACT(EPOCH FROM (event_dt - prev_dt)) / 3600)                   AS horas_max_gap,
                    MAX(ultimo_estado)                                                      AS ultimo_estado
                FROM raw_eventos
                GROUP BY logistic_order_id
            )
            SELECT
                d.logistic_order_id,
                d.service_category,
                d.is_click_and_collect,
                d.is_high_season,
                d.has_insurance,
                d.total_items,
                d.distinct_skus,
                d.dia_semana_creacion,
                d.hora_creacion,
                d.dia_mes_creacion,
                d.sla_horas_prometidas,
                d.target_binario,
                d.year_month,
                d.split_set,
                h.seller_tasa_disrupcion,
                h.categoria_tasa_disrupcion,
                h.franja_horaria,
                f.tramo_disruptivo,
                f.actor_responsable,
                ep.n_eventos,
                ep.n_estados_distintos,
                ep.tuvo_delivery_attempted,
                ep.tuvo_exception,
                ep.tuvo_annulled,
                ep.horas_max_gap,
                ep.ultimo_estado
            FROM staging_marts.ml_dataset_v2 d
            LEFT JOIN staging_marts.ml_historical_features h
              ON d.logistic_order_id = h.logistic_order_id
            LEFT JOIN staging_marts.fct_order_disruptions f
              ON d.logistic_order_id = f.logistic_order_id
            LEFT JOIN ep
              ON d.logistic_order_id = ep.logistic_order_id
            WHERE d.logistic_order_id = :oid
            LIMIT 1
        """)
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"oid": logistic_order_id})
        engine.dispose()

        if len(df) == 0:
            return {"encontrado": False, "mensaje": f"Orden {logistic_order_id} no encontrada en el historial"}

        row = df.iloc[0].to_dict()
        row["encontrado"] = True
        for k, v in row.items():
            if pd.isna(v) if not isinstance(v, str) else False:
                row[k] = None
        return row

    except Exception as e:
        return {"encontrado": False, "error": str(e)}


def consultar_estadisticas_generales(consulta: str) -> dict:
    """Consulta estadísticas agregadas del dataset."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df_dia = pd.read_sql(text("""
                SELECT
                    dia_semana_creacion,
                    ROUND(AVG(target_binario::numeric)*100, 2) as pct_disrupted,
                    COUNT(*) as n_ordenes
                FROM staging_marts.ml_dataset_v2
                WHERE service_category = 'REGULAR'
                GROUP BY dia_semana_creacion
                ORDER BY pct_disrupted DESC
            """), conn)

            df_franja = pd.read_sql(text("""
                SELECT
                    franja_horaria,
                    ROUND(AVG(d.target_binario::numeric)*100, 2) as pct_disrupted,
                    COUNT(*) as n_ordenes
                FROM staging_marts.ml_dataset_v2 d
                WHERE service_category = 'REGULAR'
                GROUP BY franja_horaria
                ORDER BY pct_disrupted DESC
            """), conn)

        engine.dispose()

        dias = {0: "Domingo", 1: "Lunes", 2: "Martes", 3: "Miércoles",
                4: "Jueves",  5: "Viernes", 6: "Sábado"}

        return {
            "por_dia": [
                {
                    "dia": dias.get(int(row["dia_semana_creacion"]), str(row["dia_semana_creacion"])),
                    "pct_disrupted": float(row["pct_disrupted"]),
                    "n_ordenes": int(row["n_ordenes"])
                }
                for _, row in df_dia.iterrows()
            ],
            "por_franja": [
                {
                    "franja": row["franja_horaria"],
                    "pct_disrupted": float(row["pct_disrupted"]),
                    "n_ordenes": int(row["n_ordenes"])
                }
                for _, row in df_franja.iterrows()
            ]
        }
    except Exception as e:
        return {"error": str(e)}


def predecir_con_itinerario(
    service_category: str,
    seller_id: str,
    sla_horas: float,
    created_at: str,
    eventos: list
) -> dict:
    """
    Predice el riesgo calculando features de flujo desde eventos reales.
    eventos = [{"timestamp": "2025-06-05T20:08:00", "estado": "DELIVERY_ATTEMPTED"}, ...]
    """
    try:
        tiempos = [datetime.fromisoformat(e["timestamp"].replace("Z", "")) for e in eventos]
        estados = [e["estado"] for e in eventos]

        gaps = []
        for i in range(1, len(tiempos)):
            gap = (tiempos[i] - tiempos[i-1]).total_seconds() / 3600
            if gap >= 0:
                gaps.append(gap)

        horas_max_gap  = max(gaps) if gaps else 0
        horas_avg_gap  = sum(gaps) / len(gaps) if gaps else 0
        n_eventos      = len(eventos)
        n_estados_dist = len(set(estados))

        t0 = tiempos[0]
        horas_a_ship = next(
            ((tiempos[i] - t0).total_seconds() / 3600
             for i, s in enumerate(estados) if s == "SHIPMENT_CONFIRMED"), -1
        )
        horas_a_otd = next(
            ((tiempos[i] - t0).total_seconds() / 3600
             for i, s in enumerate(estados) if s == "OUT_FOR_DELIVERY"), -1
        )

        tuvo_da  = int("DELIVERY_ATTEMPTED" in estados)
        tuvo_exc = int("EXCEPTION" in estados)
        tuvo_ann = int("ANNULLED" in estados)
        ultimo   = estados[-1]

        engine = get_engine()
        q = text("""
            SELECT AVG(h.seller_tasa_disrupcion)        AS seller_tasa,
                   AVG(h.categoria_tasa_disrupcion)     AS cat_tasa,
                   AVG(h.categoria_cac_tasa_disrupcion) AS cat_cac_tasa,
                   AVG(h.dia_semana_tasa_disrupcion)    AS dia_tasa,
                   AVG(h.franja_tasa_disrupcion)        AS franja_tasa,
                   MAX(h.seller_n_ordenes)              AS seller_n_ordenes
            FROM staging_marts.ml_historical_features h
            JOIN staging_marts.ml_dataset_v1 d ON h.logistic_order_id = d.logistic_order_id
            WHERE d.service_category = :cat
            LIMIT 1
        """)
        with engine.connect() as conn:
            hist = pd.read_sql(q, conn, params={"cat": service_category})
        engine.dispose()
        row = hist.iloc[0] if len(hist) > 0 else pd.Series()

        dt = datetime.fromisoformat(created_at.replace("Z", "")) if created_at else datetime.now()
        dia_semana = (dt.weekday() + 1) % 7
        hora = dt.hour
        franja = "manana" if 6 <= hora < 12 else "tarde" if 12 <= hora < 20 else "madrugada"

        feature_map = {
            "is_click_and_collect":          0.0,
            "is_high_season":                1.0,
            "has_insurance":                 1.0,
            "total_items":                   1.0,
            "distinct_skus":                 1.0,
            "dia_semana_creacion":           float(dia_semana),
            "hora_creacion":                 float(hora),
            "dia_mes_creacion":              float(dt.day),
            "sla_horas_prometidas":          float(sla_horas),
            "seller_n_ordenes":              float(row.get("seller_n_ordenes", 0) or 0),
            "seller_tasa_disrupcion":        float(row.get("seller_tasa", 0.4) or 0.4),
            "categoria_tasa_disrupcion":     float(row.get("cat_tasa", 0.4) or 0.4),
            "categoria_cac_tasa_disrupcion": float(row.get("cat_cac_tasa", 0.4) or 0.4),
            "dia_semana_tasa_disrupcion":    float(row.get("dia_tasa", 0.4) or 0.4),
            "franja_tasa_disrupcion":        float(row.get("franja_tasa", 0.4) or 0.4),
            "horas_a_shipment_confirmed":    float(horas_a_ship),
            "horas_a_out_for_delivery":      float(horas_a_otd),
            "horas_entre_eventos_max":       float(horas_max_gap),
            "horas_entre_eventos_avg":       float(horas_avg_gap),
            "n_eventos_total":               float(n_eventos),
            "n_estados_distintos":           float(n_estados_dist),
            "tuvo_delivery_attempted":       float(tuvo_da),
            "tuvo_exception":                float(tuvo_exc),
            "tuvo_annulled":                 float(tuvo_ann),
            "esta_en_transito":              float(1 if ultimo == "IN_TRANSIT" else 0),
            "esta_out_for_delivery":         float(1 if ultimo == "OUT_FOR_DELIVERY" else 0),
            "esta_shipment_confirmed":       float(1 if ultimo == "SHIPMENT_CONFIRMED" else 0),
            "service_category":              service_category,
            "franja_horaria":                franja,
            "ultimo_estado":                 ultimo,
        }

        for col in ["service_category", "franja_horaria", "ultimo_estado"]:
            if col in ENCODERS:
                le  = ENCODERS[col]
                val = feature_map[col]
                if val not in le.classes_:
                    val = "DESCONOCIDO"
                feature_map[col] = int(le.transform([val])[0])
            else:
                feature_map[col] = 0

        X    = np.array([[feature_map[f] for f in FEATURES]], dtype="float32")
        prob = float(MODEL.predict(X)[0])

        nivel = "CRÍTICO" if prob >= 0.8 else "ALTO" if prob >= 0.6 else "MODERADO" if prob >= 0.4 else "BAJO"

        return {
            "prob_disrupcion":         round(prob, 4),
            "porcentaje":              round(prob * 100, 1),
            "nivel_riesgo":            nivel,
            "n_eventos_analizados":    n_eventos,
            "horas_max_gap":           round(horas_max_gap, 1),
            "tuvo_delivery_attempted": bool(tuvo_da),
            "ultimo_estado":           ultimo,
            "modelo":                  "Modelo D — AUC 0.9831",
        }

    except Exception as e:
        return {"error": str(e)}


def predecir_orden(
    service_category: str,
    seller_id: str,
    sla_horas: float,
    created_at: str,
    total_items: int = 1,
    distinct_skus: int = 1,
    is_click_and_collect: int = 0,
    is_high_season: int = 0,
    has_insurance: int = 0,
) -> dict:
    """
    Corre el modelo ML v2 y devuelve la predicción con todos los detalles.
    """
    # Parsear fecha
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", ""))
    except:
        dt = datetime.now()

    dia_semana = (dt.weekday() + 1) % 7
    hora       = dt.hour
    dia_mes    = dt.day

    # Franja horaria
    if 6 <= hora < 12:
        franja = "manana"
    elif 12 <= hora < 20:
        franja = "tarde"
    else:
        franja = "madrugada"

    # Buscar histórico
    hist = buscar_historico_seller(seller_id, service_category)

    # Construir vector
    row = {
        "is_click_and_collect":          float(is_click_and_collect),
        "is_high_season":                float(is_high_season),
        "has_insurance":                 float(has_insurance),
        "total_items":                   float(total_items),
        "distinct_skus":                 float(distinct_skus),
        "dia_semana_creacion":           float(dia_semana),
        "hora_creacion":                 float(hora),
        "dia_mes_creacion":              float(dia_mes),
        "sla_horas_prometidas":          float(sla_horas),
        "seller_n_ordenes":              float(hist["seller_n_ordenes"]),
        "seller_tasa_disrupcion":        float(hist["seller_tasa"]),
        "categoria_tasa_disrupcion":     float(hist["cat_tasa"]),
        "categoria_cac_tasa_disrupcion": float(hist["cat_cac_tasa"]),
        "dia_semana_tasa_disrupcion":    float(hist["dia_tasa"]),
        "franja_tasa_disrupcion":        float(hist["franja_tasa"]),
        "service_category":              service_category,
        "franja_horaria":                franja,
    }

    # Encodear categóricas
    for col in ["service_category", "franja_horaria"]:
        if col in ENCODERS:
            le  = ENCODERS[col]
            val = row[col]
            if val not in le.classes_:
                val = "DESCONOCIDO"
            row[col] = int(le.transform([val])[0])
        else:
            row[col] = 0

    X    = np.array([[row[f] for f in FEATURES]], dtype="float32")
    prob = float(MODEL.predict(X)[0])

    return {
        "prob_disrupcion":  round(prob, 4),
        "pred_disrupcion":  1 if prob >= 0.5 else 0,
        "porcentaje":       round(prob * 100, 1),
        "dia_semana":       DIAS[dia_semana],
        "hora":             hora,
        "franja":           franja,
        "historico":        hist,
        "service_category": service_category,
        "seller_id":        seller_id,
        "sla_horas":        sla_horas,
    }


# ── Definición de herramientas para Kimi (formato OpenAI) ────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predecir_disrupcion",
            "description": (
                "Predice si una orden logística va a tener una disrupción "
                "usando el modelo ML entrenado con 15M de órdenes de Falabella Chile. "
                "Llamar cuando el usuario proporcione datos de una orden nueva."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_category": {
                        "type": "string",
                        "description": "Categoría del servicio: REGULAR, MESON, DATE_RANGE, SAME_DAY, EXPRESS, TO_CAR",
                    },
                    "seller_id": {
                        "type": "string",
                        "description": "ID del seller/vendedor",
                    },
                    "sla_horas": {
                        "type": "number",
                        "description": "Horas entre creación y entrega prometida al cliente",
                    },
                    "created_at": {
                        "type": "string",
                        "description": "Fecha y hora de creación en formato ISO. Si no se especifica usar ahora.",
                    },
                    "total_items": {
                        "type": "integer",
                        "description": "Cantidad total de items en la orden",
                    },
                    "distinct_skus": {
                        "type": "integer",
                        "description": "Cantidad de SKUs distintos",
                    },
                    "is_click_and_collect": {
                        "type": "integer",
                        "description": "1 si es retiro en tienda, 0 si es delivery",
                    },
                    "is_high_season": {
                        "type": "integer",
                        "description": "1 si es temporada alta, 0 si no",
                    },
                    "has_insurance": {
                        "type": "integer",
                        "description": "1 si tiene seguro, 0 si no",
                    },
                },
                "required": ["service_category", "seller_id", "sla_horas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_tramo_historico",
            "description": (
                "Consulta en qué tramo logístico históricamente fallan las órdenes "
                "de un seller y categoría. Llamar cuando el usuario quiera saber "
                "dónde suelen ocurrir las disrupciones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_id": {"type": "string", "description": "ID del seller"},
                    "service_category": {"type": "string", "description": "Categoría del servicio"},
                },
                "required": ["seller_id", "service_category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_historico_seller",
            "description": (
                "Obtiene las tasas históricas de disrupción de un seller y categoría. "
                "Llamar cuando el usuario quiera saber el historial de un seller específico."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_id": {"type": "string", "description": "ID del seller"},
                    "service_category": {"type": "string", "description": "Categoría del servicio"},
                },
                "required": ["seller_id", "service_category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_recorrido_orden",
            "description": (
                "Consulta el recorrido completo de una orden específica: todos los estados y eventos "
                "por los que pasó, con las fechas y tiempos entre cada estado. "
                "Usar cuando el usuario pregunte por el tracking, recorrido, estados o eventos de una orden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "logistic_order_id": {
                        "type": "string",
                        "description": "ID de la orden logística",
                    },
                },
                "required": ["logistic_order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_estadisticas_generales",
            "description": (
                "Consulta estadísticas agregadas del dataset — peor día de la semana, "
                "peor franja horaria, patrones por categoría. "
                "Usar cuando el usuario pregunte por patrones generales, "
                "el peor día, la peor hora, o tendencias por categoría."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "Descripción de qué estadística se quiere consultar",
                    },
                },
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predecir_con_itinerario",
            "description": (
                "Predice el riesgo de disrupción usando el itinerario real de eventos de una orden. "
                "Llamar cuando el usuario quiera saber el riesgo en un momento específico del recorrido "
                "o cuando proporcione una secuencia de estados con timestamps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_category": {"type": "string"},
                    "seller_id":        {"type": "string"},
                    "sla_horas":        {"type": "number"},
                    "created_at":       {"type": "string"},
                    "eventos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {"type": "string"},
                                "estado":    {"type": "string"},
                            }
                        }
                    }
                },
                "required": ["service_category", "sla_horas", "eventos"],
            },
        },
    },
]


# ── Motor del agente con Claude ───────────────────────────────────────────────

SYSTEM_PROMPT = """Sos un agente de inteligencia artificial especializado en logística
de Falabella Chile. Tu función es predecir disrupciones en órdenes de entrega
usando modelos de Machine Learning entrenados con 15 millones de órdenes históricas.

Cuando el usuario te mande datos de una orden (puede ser en cualquier formato,
lenguaje natural, JSON, o una descripción), vos:
1. Extraés los datos relevantes: categoría, seller, SLA, fecha/hora
2. Llamás a la herramienta predecir_disrupcion
3. Si la probabilidad es alta (>60%), consultás también el tramo histórico
4. Respondés de forma clara y accionable

REGLA CRÍTICA — PREDICCIÓN CON ITINERARIO:
Si el usuario menciona un logistic_order_id Y pide predecir el riesgo o analizar
la orden, SIEMPRE seguís este orden obligatorio:
1. Primero llamás consultar_recorrido_orden con ese ID para obtener todos los
   eventos reales de la orden desde la base de datos
2. Con esos eventos reales llamás predecir_con_itinerario pasando TODOS los
   eventos encontrados — nunca uses solo los eventos que el usuario menciona
   en el texto, siempre usá los eventos reales de la base de datos
3. Combinás ambos resultados para dar una predicción precisa

Si el usuario pregunta sobre el historial de un seller, usás consultar_historico_seller.
Si pregunta dónde suelen fallar las órdenes, usás consultar_tramo_historico.
Si el usuario pregunta por el recorrido, tracking o estados de una orden,
usás consultar_recorrido_orden.
Si el usuario pregunta por datos de una orden específica por ID,
usás buscar_orden_por_id.
Si el usuario pregunta por patrones generales o estadísticas,
usás consultar_estadisticas_generales.

REGLA — ANÁLISIS PROGRESIVO:
Cuando el usuario pida analizar el riesgo de una orden histórica por ID,
SIEMPRE mostrás la evolución del riesgo en cada cambio de estado — no solo
el riesgo final. El formato es:

Evento 1 — 03/06 18:01 | SHIPMENT_CONFIRMED → 45%
Evento 2 — 04/06 01:28 | IN_TRANSIT (+7hs) → 52%
Evento 3 — 05/06 16:13 | OUT_FOR_DELIVERY (+35hs) → 61%
Evento 4 — 05/06 20:08 | DELIVERY_ATTEMPTED (+4hs) → 78% 🔔 ALERTA
...

Así el usuario ve exactamente en qué momento el modelo hubiera alertado.
Solo mostrás los cambios de estado, no los eventos duplicados.

Respondés siempre en español, de forma concisa y con emojis para facilitar la lectura.
Sos honesto sobre la confianza de la predicción cuando el seller es nuevo o desconocido.

Datos sobre las categorías de servicio:
- REGULAR: 79.7% disrupción histórica — riesgo alto
- SAME_DAY: 93.4% disrupción — riesgo crítico
- EXPRESS: 96% disrupción — riesgo crítico
- MESON: 13.8% disrupción — riesgo bajo
- DATE_RANGE: 16.6% disrupción — riesgo bajo
- TO_CAR: 22.2% disrupción — riesgo moderado

El modelo tiene AUC-ROC de 0.9831 y detecta 9 de cada 10 disrupciones."""


async def procesar_con_claude(mensaje: str, historial: list) -> str:
    """
    Manda el mensaje a Kimi (via OpenAI SDK) con las herramientas disponibles.
    Kimi decide qué herramientas usar y genera la respuesta final.
    """
    client = OpenAI(
        api_key=KIMI_KEY,
        base_url="https://api.moonshot.ai/v1"
    )

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + historial
        + [{"role": "user", "content": mensaje}]
    )

    # Loop del agente — Kimi puede usar múltiples herramientas
    for _ in range(5):
        response = client.chat.completions.create(
            model="kimi-k2.5",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1000,
            temperature=1,
        )

        msg = response.choices[0].message

        # Si Kimi terminó — no hay más herramientas que usar
        if not msg.tool_calls:
            return msg.content or "No pude generar una respuesta."

        # Kimi quiere usar herramientas — agregarlas al historial
        messages.append(msg)

        # Ejecutar cada herramienta
        for tool_call in msg.tool_calls:
            tool_name  = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            logger.info(f"Herramienta: {tool_name} — input: {tool_input}")

            if tool_name == "predecir_disrupcion":
                resultado = predecir_orden(
                    service_category     = tool_input.get("service_category", "REGULAR"),
                    seller_id            = tool_input.get("seller_id", "DESCONOCIDO"),
                    sla_horas            = float(tool_input.get("sla_horas", 48)),
                    created_at           = tool_input.get("created_at", datetime.now().isoformat()),
                    total_items          = int(tool_input.get("total_items", 1)),
                    distinct_skus        = int(tool_input.get("distinct_skus", 1)),
                    is_click_and_collect = int(tool_input.get("is_click_and_collect", 0)),
                    is_high_season       = int(tool_input.get("is_high_season", 0)),
                    has_insurance        = int(tool_input.get("has_insurance", 0)),
                )
            elif tool_name == "consultar_tramo_historico":
                resultado = buscar_tramo_historico(
                    seller_id        = tool_input.get("seller_id", ""),
                    service_category = tool_input.get("service_category", "REGULAR"),
                )
            elif tool_name == "consultar_historico_seller":
                resultado = buscar_historico_seller(
                    seller_id        = tool_input.get("seller_id", ""),
                    service_category = tool_input.get("service_category", "REGULAR"),
                )
            elif tool_name == "consultar_recorrido_orden":
                resultado = consultar_recorrido_orden(
                    logistic_order_id = tool_input.get("logistic_order_id", ""),
                )
            elif tool_name == "buscar_orden_por_id":
                resultado = buscar_orden_por_id(
                    logistic_order_id = tool_input.get("logistic_order_id", ""),
                )
            elif tool_name == "consultar_estadisticas_generales":
                resultado = consultar_estadisticas_generales(
                    consulta = tool_input.get("consulta", ""),
                )
            elif tool_name == "predecir_con_itinerario":
                resultado = predecir_con_itinerario(
                    service_category = tool_input.get("service_category", ""),
                    seller_id        = tool_input.get("seller_id", ""),
                    sla_horas        = tool_input.get("sla_horas", 48),
                    created_at       = tool_input.get("created_at", ""),
                    eventos          = tool_input.get("eventos", []),
                )
            else:
                resultado = {"error": f"Herramienta {tool_name} no encontrada"}

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      json.dumps(resultado, ensure_ascii=False, default=str),
            })

    return "Lo siento, no pude procesar la solicitud. Intentá de nuevo."


# ── Manejo de conversaciones por usuario ──────────────────────────────────────

# Historial de conversación por usuario (en memoria)
historiales = {}

def get_historial(user_id: int) -> list:
    return historiales.get(user_id, [])

def actualizar_historial(user_id: int, mensaje: str, respuesta: str):
    if user_id not in historiales:
        historiales[user_id] = []
    historiales[user_id].append({"role": "user",      "content": mensaje})
    historiales[user_id].append({"role": "assistant",  "content": respuesta})
    # Mantener solo los últimos 10 intercambios para no explotar el contexto
    if len(historiales[user_id]) > 20:
        historiales[user_id] = historiales[user_id][-20:]


# ── Handlers de Telegram ──────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    historiales[user_id] = []  # resetear historial

    msg = (
        "👋 *Hola, soy el Agente de Disrupciones Falabella*\n\n"
        "Estoy entrenado con *15 millones de órdenes* de Falabella Chile "
        "y puedo predecir si una orden va a tener problemas de entrega "
        "antes de que ocurran.\n\n"
        "*¿Qué podés preguntarme?*\n\n"
        "📦 *Predecir una orden nueva:*\n"
        "\"Tengo una orden REGULAR del seller SCE3622, "
        "SLA 48 horas, 3 items, creada hoy a las 14hs\"\n\n"
        "📊 *Consultar historial de un seller:*\n"
        "\"¿Cómo viene el seller FALABELLA\\_CHILE en REGULAR?\"\n\n"
        "🔍 *Dónde suelen fallar las órdenes:*\n"
        "\"¿Dónde falla normalmente SCE3622?\"\n\n"
        "Escribime en lenguaje natural, no necesitás saber ningún formato especial 🚀"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    historiales[user_id] = []
    await update.message.reply_text(
        "🔄 Conversación reiniciada. ¿En qué te puedo ayudar?",
        parse_mode="HTML"
    )


async def cmd_ejemplo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📄 *Ejemplos de cómo mandar una orden:*\n\n"
        "*Lenguaje natural:*\n"
        "\"Tengo una orden REGULAR del seller SCE3622, "
        "promete entrega en 2 días, son 3 productos\"\n\n"
        "*Más detallado:*\n"
        "\"Orden de hoy 15hs, categoría SAME\\_DAY, "
        "seller TOTTUS\\_CHILE, SLA 4 horas, 1 item\"\n\n"
        "*También acepta JSON:*\n"
        "```\n"
        "{\n"
        "  \"service_category\": \"REGULAR\",\n"
        "  \"seller_id\": \"SCE3622\",\n"
        "  \"sla_horas\": 48,\n"
        "  \"total_items\": 3\n"
        "}\n"
        "```\n\n"
        "Usá /reset para empezar una conversación nueva."
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def handle_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler principal — procesa cualquier mensaje con Claude."""
    user_id  = update.effective_user.id
    mensaje  = update.message.text.strip()
    username = update.effective_user.first_name or "Operador"

    logger.info(f"Mensaje de {username} ({user_id}): {mensaje[:100]}")

    # Si el mensaje parece una nueva orden, limpiar historial automáticamente
    keywords_nueva_orden = ["tengo una orden", "nueva orden", "orden nueva",
                            "tengo orden", "ingreso orden", "creada hoy",
                            "creada hace", "sla ", "items, creada"]
    if any(kw in mensaje.lower() for kw in keywords_nueva_orden):
        historiales[user_id] = []
        logger.info(f"Historial limpiado automáticamente para nueva orden")

    # Indicador de escritura
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    historial = get_historial(user_id)

    try:
        respuesta = await procesar_con_claude(mensaje, historial)
        actualizar_historial(user_id, mensaje, respuesta)

        # Telegram tiene límite de 4096 chars por mensaje
        if len(respuesta) > 4000:
            partes = [respuesta[i:i+4000] for i in range(0, len(respuesta), 4000)]
            for parte in partes:
                await update.message.reply_text(parte, parse_mode="HTML")
        else:
            await update.message.reply_text(respuesta, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error procesando mensaje: {e}")
        await update.message.reply_text(
            "❌ Hubo un error procesando tu mensaje. "
            "Intentá de nuevo o usá /reset para reiniciar.",
            parse_mode="HTML"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not KIMI_KEY:
        print("ERROR: KIMI_API_KEY no encontrada en .env", flush=True)
        return

    print("Iniciando Agente Falabella en Telegram...", flush=True)
    print(f"Token Telegram: {TOKEN[:20]}...", flush=True)

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("reset",   cmd_reset))
    app.add_handler(CommandHandler("ejemplo", cmd_ejemplo))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_mensaje
    ))

    print("Agente corriendo. Escribile a tu bot en Telegram.", flush=True)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
"""
Demo Progresiva — Validación post-mortem de órdenes.

Toma una orden histórica y simula el flujo evento por evento,
mostrando cómo el Modelo D actualiza el riesgo en cada estado.

Uso:
    python3 demo_progresiva.py --orden FOFCL000025516262
    python3 demo_progresiva.py --orden FOFCL000025516262 --hasta "2025-06-05T20:08:00"

Outputs:
    - Timeline con riesgo en cada evento
    - Momento exacto en que el modelo hubiera alertado
    - Comparación predicción vs resultado real
"""

import os, gc, json, pickle, warnings, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

OUTPUT_DIR = Path("/home/ec2-user/Inicio_falabella/ml_outputs")
MODEL_PATH = OUTPUT_DIR / "model_lgbm_d_binario.pkl"

# Cargar modelo
with open(MODEL_PATH, "rb") as f:
    art = pickle.load(f)
MODEL    = art["model"]
ENCODERS = art["encoders"]
FEATURES = art["features"]


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DBT_DB_USER']}:{os.environ['DBT_DB_PASSWORD']}"
        f"@{os.environ['DBT_DB_HOST']}:5432/{os.environ['DBT_DB_NAME']}?sslmode=require"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=60)


def cargar_orden(logistic_order_id: str, hasta: str = None) -> dict:
    """Carga todos los datos de una orden desde RDS."""
    engine = get_engine()

    # Datos base de la orden
    q_base = text("""
        SELECT
            d.logistic_order_id,
            d.service_category,
            d.sla_horas_prometidas,
            d.target_binario,
            d.is_click_and_collect,
            d.is_high_season,
            d.has_insurance,
            d.total_items,
            d.distinct_skus,
            d.dia_semana_creacion,
            d.hora_creacion,
            d.dia_mes_creacion,
            d.year_month,
            h.seller_tasa_disrupcion,
            h.categoria_tasa_disrupcion,
            h.categoria_cac_tasa_disrupcion,
            h.dia_semana_tasa_disrupcion,
            h.franja_horaria,
            h.franja_tasa_disrupcion,
            h.seller_n_ordenes,
            f.tramo_disruptivo,
            f.actor_responsable
        FROM staging_marts.ml_dataset_v2 d
        LEFT JOIN staging_marts.ml_historical_features h
          ON d.logistic_order_id = h.logistic_order_id
        LEFT JOIN staging_marts.fct_order_disruptions f
          ON d.logistic_order_id = f.logistic_order_id
        WHERE d.logistic_order_id = :oid
        LIMIT 1
    """)

    with engine.connect() as conn:
        df_base = pd.read_sql(q_base, conn, params={"oid": logistic_order_id})

    if len(df_base) == 0:
        print(f"Orden {logistic_order_id} no encontrada en ml_dataset_v2")
        return None

    # Eventos de la orden
    hasta_clause = "AND event_dt <= :hasta" if hasta else ""
    q_eventos = text(f"""
        SELECT
            package_id,
            package_status,
            event_type,
            event_dt,
            promised_date
        FROM staging_marts.stg_event_packages
        WHERE logistic_order_id = :oid
        {hasta_clause}
        ORDER BY event_dt
    """)

    params = {"oid": logistic_order_id}
    if hasta:
        params["hasta"] = hasta

    with engine.connect() as conn:
        df_eventos = pd.read_sql(q_eventos, conn, params=params)

    engine.dispose()

    return {
        "base":    df_base.iloc[0].to_dict(),
        "eventos": df_eventos,
    }


def predecir_en_momento(base: dict, eventos_hasta: pd.DataFrame) -> float:
    """Predice el riesgo con los eventos disponibles hasta un momento dado."""
    if len(eventos_hasta) == 0:
        horas_max_gap = -1
        horas_avg_gap = -1
        n_eventos     = 0
        n_estados     = 1
        tuvo_da = tuvo_exc = tuvo_ann = 0
        horas_a_ship = horas_a_otd = -1
        ultimo_estado = "SHIPMENT_CONFIRMED"
        en_transito = en_otd = en_ship = 0
    else:
        eventos_hasta["event_dt"] = pd.to_datetime(eventos_hasta["event_dt"])
        estados = eventos_hasta["package_status"].tolist()
        tiempos = eventos_hasta["event_dt"].tolist()

        gaps = []
        for i in range(1, len(tiempos)):
            gap = (tiempos[i] - tiempos[i-1]).total_seconds() / 3600
            if gap >= 0:
                gaps.append(gap)

        horas_max_gap = max(gaps) if gaps else 0
        horas_avg_gap = sum(gaps)/len(gaps) if gaps else 0
        n_eventos     = len(eventos_hasta)
        n_estados     = eventos_hasta["package_status"].nunique()
        tuvo_da  = int("DELIVERY_ATTEMPTED" in estados)
        tuvo_exc = int("EXCEPTION" in estados)
        tuvo_ann = int("ANNULLED" in estados)
        ultimo_estado = estados[-1]
        en_transito = int(ultimo_estado == "IN_TRANSIT")
        en_otd      = int(ultimo_estado == "OUT_FOR_DELIVERY")
        en_ship     = int(ultimo_estado == "SHIPMENT_CONFIRMED")

        t0 = tiempos[0]
        horas_a_ship = next(
            ((tiempos[i]-t0).total_seconds()/3600
             for i, s in enumerate(estados) if s == "SHIPMENT_CONFIRMED"), -1
        )
        horas_a_otd = next(
            ((tiempos[i]-t0).total_seconds()/3600
             for i, s in enumerate(estados) if s == "OUT_FOR_DELIVERY"), -1
        )

    feature_map = {
        "is_click_and_collect":          float(base.get("is_click_and_collect", 0) or 0),
        "is_high_season":                float(base.get("is_high_season", 0) or 0),
        "has_insurance":                 float(base.get("has_insurance", 0) or 0),
        "total_items":                   float(base.get("total_items", 1) or 1),
        "distinct_skus":                 float(base.get("distinct_skus", 1) or 1),
        "dia_semana_creacion":           float(base.get("dia_semana_creacion", 0) or 0),
        "hora_creacion":                 float(base.get("hora_creacion", 12) or 12),
        "dia_mes_creacion":              float(base.get("dia_mes_creacion", 1) or 1),
        "sla_horas_prometidas":          float(base.get("sla_horas_prometidas", 48) or 48),
        "seller_n_ordenes":              float(base.get("seller_n_ordenes", 0) or 0),
        "seller_tasa_disrupcion":        float(base.get("seller_tasa_disrupcion", 0.4) or 0.4),
        "categoria_tasa_disrupcion":     float(base.get("categoria_tasa_disrupcion", 0.4) or 0.4),
        "categoria_cac_tasa_disrupcion": float(base.get("categoria_cac_tasa_disrupcion", 0.4) or 0.4),
        "dia_semana_tasa_disrupcion":    float(base.get("dia_semana_tasa_disrupcion", 0.4) or 0.4),
        "franja_tasa_disrupcion":        float(base.get("franja_tasa_disrupcion", 0.4) or 0.4),
        "horas_a_shipment_confirmed":    float(horas_a_ship),
        "horas_a_out_for_delivery":      float(horas_a_otd),
        "horas_entre_eventos_max":       float(horas_max_gap),
        "horas_entre_eventos_avg":       float(horas_avg_gap),
        "n_eventos_total":               float(n_eventos),
        "n_estados_distintos":           float(n_estados),
        "tuvo_delivery_attempted":       float(tuvo_da),
        "tuvo_exception":                float(tuvo_exc),
        "tuvo_annulled":                 float(tuvo_ann),
        "esta_en_transito":              float(en_transito),
        "esta_out_for_delivery":         float(en_otd),
        "esta_shipment_confirmed":       float(en_ship),
        "service_category":              str(base.get("service_category", "REGULAR")),
        "franja_horaria":                str(base.get("franja_horaria", "tarde")),
        "ultimo_estado":                 str(ultimo_estado),
    }

    # Encodear categóricas
    for col in ["service_category", "franja_horaria", "ultimo_estado"]:
        if col in ENCODERS:
            le  = ENCODERS[col]
            val = feature_map[col]
            if val not in le.classes_:
                val = "DESCONOCIDO"
            feature_map[col] = int(le.transform([val])[0])
        else:
            feature_map[col] = 0

    X = np.array([[feature_map[f] for f in FEATURES]], dtype="float32")
    return float(MODEL.predict(X)[0])


def barra(prob_pct):
    filled = int(prob_pct / 10)
    return "█" * filled + "░" * (10 - filled)


def emoji_riesgo(prob_pct):
    if prob_pct >= 80: return "🔴"
    if prob_pct >= 60: return "🟠"
    if prob_pct >= 40: return "🟡"
    return "🟢"


def main():
    parser = argparse.ArgumentParser(description="Demo progresiva de predicción")
    parser.add_argument("--orden", required=True, help="logistic_order_id")
    parser.add_argument("--hasta", default=None, help="Simular hasta este timestamp ISO")
    args = parser.parse_args()

    print("=" * 65)
    print(f"DEMO PROGRESIVA — {args.orden}")
    print(f"Modelo D — AUC 0.9831")
    print("=" * 65)

    datos = cargar_orden(args.orden, args.hasta)
    if not datos:
        return

    base    = datos["base"]
    eventos = datos["eventos"]

    print(f"\nOrden:     {args.orden}")
    print(f"Categoría: {base.get('service_category')}")
    print(f"SLA:       {base.get('sla_horas_prometidas')} horas")
    print(f"Disrumpió: {'SÍ ✗' if base.get('target_binario') == 1 else 'NO ✓'}")
    if base.get("tramo_disruptivo"):
        print(f"Tramo:     {base.get('tramo_disruptivo')} — {base.get('actor_responsable')}")
    print(f"Total eventos disponibles: {len(eventos)}")
    print()

    # Predicción inicial — sin eventos
    prob_inicial = predecir_en_momento(base, pd.DataFrame()) * 100
    print(f"{'='*65}")
    print(f"PREDICCIÓN INICIAL (solo features de creación):")
    print(f"  {emoji_riesgo(prob_inicial)} {barra(prob_inicial)} {prob_inicial:.1f}%")
    print()

    # Simular evento por evento
    primera_alerta = None
    print(f"{'='*65}")
    print("SIMULACIÓN PROGRESIVA — evento por evento:")
    print(f"{'='*65}")

    # Agrupar eventos únicos por timestamp + estado
    eventos["event_dt"] = pd.to_datetime(eventos["event_dt"])
    eventos_unicos = eventos.drop_duplicates(
        subset=["event_dt", "package_status"]
    ).sort_values("event_dt").reset_index(drop=True)

    # Tomar hitos clave — no mostrar todos los duplicados
    hitos = []
    ultimo_estado = None
    for _, ev in eventos_unicos.iterrows():
        if ev["package_status"] != ultimo_estado:
            hitos.append(ev)
            ultimo_estado = ev["package_status"]

    for i, hito in enumerate(hitos):
        eventos_hasta = eventos[eventos["event_dt"] <= hito["event_dt"]]
        prob = predecir_en_momento(base, eventos_hasta) * 100

        estado = hito["package_status"]
        fecha  = hito["event_dt"].strftime("%d/%m %H:%M")

        # Detectar alertas
        alerta = ""
        if estado == "DELIVERY_ATTEMPTED":
            alerta = " ⚠️ INTENTO FALLIDO"
        elif estado == "ANNULLED":
            alerta = " 💀 ANULADA"
        elif estado == "EXCEPTION":
            alerta = " 🚨 EXCEPCIÓN"

        # Gap desde evento anterior
        gap_txt = ""
        if i > 0:
            gap = (hito["event_dt"] - hitos[i-1]["event_dt"]).total_seconds() / 3600
            if gap > 24:
                gap_txt = f" (+{gap:.0f}hs ⚠️)"
            elif gap > 0:
                gap_txt = f" (+{gap:.1f}hs)"

        print(f"\nEvento {i+1:02d} — {fecha}{gap_txt}")
        print(f"  Estado: {estado}{alerta}")
        print(f"  {emoji_riesgo(prob)} {barra(prob)} {prob:.1f}%")

        # Primera vez que supera el umbral de alerta
        if prob >= 70 and primera_alerta is None:
            primera_alerta = {
                "evento": i+1,
                "fecha":  fecha,
                "estado": estado,
                "prob":   prob,
            }
            print(f"  🔔 PRIMERA ALERTA AQUÍ — umbral 70% superado")

    # Resumen final
    print(f"\n{'='*65}")
    print("RESUMEN FINAL")
    print(f"{'='*65}")

    prob_final = predecir_en_momento(base, eventos) * 100
    resultado  = "DISRUMPIÓ ✗" if base.get("target_binario") == 1 else "ENTREGADA ✓"

    print(f"  Resultado real:     {resultado}")
    print(f"  Prob final modelo:  {prob_final:.1f}%")
    print(f"  Total eventos:      {len(eventos)}")

    if primera_alerta:
        print(f"\n  🔔 El modelo hubiera alertado en:")
        print(f"     Evento {primera_alerta['evento']} — {primera_alerta['fecha']}")
        print(f"     Estado: {primera_alerta['estado']}")
        print(f"     Probabilidad: {primera_alerta['prob']:.1f}%")
    else:
        print(f"\n  El modelo nunca superó el umbral de 70%")

    if base.get("tramo_disruptivo"):
        print(f"\n  Diagnóstico real:")
        print(f"     Tramo: {base.get('tramo_disruptivo')}")
        print(f"     Actor: {base.get('actor_responsable')}")


if __name__ == "__main__":
    main()

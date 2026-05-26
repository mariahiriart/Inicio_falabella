import re

with open('agente_falabella.py', 'r') as f:
    content = f.read()

# ── 1. Nueva función buscar_orden_por_id ─────────────────────────────────────
nueva_funcion = '''

def buscar_orden_por_id(logistic_order_id: str) -> dict:
    """Busca una orden específica por ID en RDS y devuelve todos sus datos."""
    try:
        engine = get_engine()
        q = text("""
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
            LEFT JOIN (
                SELECT
                    logistic_order_id,
                    COUNT(*) as n_eventos,
                    COUNT(DISTINCT package_status) as n_estados_distintos,
                    MAX(CASE WHEN package_status = 'DELIVERY_ATTEMPTED' THEN 1 ELSE 0 END) as tuvo_delivery_attempted,
                    MAX(CASE WHEN package_status = 'EXCEPTION' THEN 1 ELSE 0 END) as tuvo_exception,
                    MAX(CASE WHEN package_status = 'ANNULLED' THEN 1 ELSE 0 END) as tuvo_annulled,
                    MAX(EXTRACT(EPOCH FROM (event_dt - LAG(event_dt) OVER (
                        PARTITION BY logistic_order_id ORDER BY event_dt
                    ))) / 3600) as horas_max_gap,
                    LAST_VALUE(package_status) OVER (
                        PARTITION BY logistic_order_id ORDER BY event_dt
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) as ultimo_estado
                FROM staging_marts.stg_event_packages
                WHERE logistic_order_id = :oid
                GROUP BY logistic_order_id, package_status, event_dt
                LIMIT 1
            ) ep ON d.logistic_order_id = ep.logistic_order_id
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
'''

# Insertar antes de def predecir_orden(
if 'def predecir_orden(' in content:
    content = content.replace('def predecir_orden(', nueva_funcion + '\ndef predecir_orden(', 1)
    print("✓ Función buscar_orden_por_id insertada")
else:
    print("✗ No se encontró 'def predecir_orden(' — revisá el archivo")

# ── 2. Nueva tool buscar_orden_por_id ────────────────────────────────────────
nueva_tool = '''    ,
    {
        "type": "function",
        "function": {
            "name": "buscar_orden_por_id",
            "description": (
                "Busca una orden específica por su logistic_order_id en la base de datos histórica. "
                "Devuelve todos los datos de la orden: categoría, seller, SLA, eventos, tramo de disrupción. "
                "Usar cuando el usuario mencione un ID específico de orden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "logistic_order_id": {
                        "type": "string",
                        "description": "ID de la orden logística, ejemplo: FOPCL000089234712",
                    },
                },
                "required": ["logistic_order_id"],
            },
        },
    }
'''

# Buscar el cierre de TOOLS = [...] con regex flexible
pattern = r'(\s*\},?\s*\]\s*\n)(# -- Motor)'
match = re.search(pattern, content)

if match:
    insert_pos = match.start(2)
    closing = match.group(1)
    # Reemplazar el cierre ] por cierre con la nueva tool
    new_closing = closing.rstrip()
    # Quitar el ] del cierre actual y agregar la tool + ]
    new_closing = re.sub(r'\](\s*)$', nueva_tool + ']\n', new_closing)
    content = content[:match.start(1)] + new_closing + '\n' + content[insert_pos:]
    print("✓ Tool buscar_orden_por_id agregada a TOOLS")
else:
    print("✗ No se encontró el cierre de TOOLS — buscá manualmente '# -- Motor' en el archivo")

with open('agente_falabella.py', 'w') as f:
    f.write(content)

print("✓ Archivo guardado")

# Verificar sintaxis
import ast
try:
    ast.parse(content)
    print("✓ Sintaxis Python OK")
except SyntaxError as e:
    print(f"✗ SyntaxError en línea {e.lineno}: {e.msg}")

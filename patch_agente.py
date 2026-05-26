import ast

content = open('agente_falabella.py').read()

# ── 1. Nueva función buscar_orden_por_id ─────────────────────────────────────
nueva_funcion = '''
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
'''

if 'def predecir_orden(' not in content:
    print("✗ No se encontró 'def predecir_orden(' — revisá el archivo")
    exit(1)

content = content.replace(
    'def predecir_orden(',
    nueva_funcion + '\ndef predecir_orden(',
    1
)
print("✓ Función buscar_orden_por_id insertada")

# ── 2. Nueva tool buscar_orden_por_id ────────────────────────────────────────
# La búsqueda incluye el cierre del último tool existente ("    },")
# para restaurarlo correctamente en el reemplazo.
nueva_tool = '''    },
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
]'''

if '    },\n]\n\n' not in content:
    print("✗ No se encontró el cierre de TOOLS — revisá el archivo")
    exit(1)

content = content.replace(
    '    },\n]\n\n',
    nueva_tool + '\n\n',
    1
)
print("✓ Tool buscar_orden_por_id agregada a TOOLS")

# ── Verificar sintaxis antes de guardar ──────────────────────────────────────
try:
    ast.parse(content)
    print("✓ Sintaxis Python OK")
except SyntaxError as e:
    print(f"✗ SyntaxError en línea {e.lineno}: {e.msg}")
    exit(1)

open('agente_falabella.py', 'w').write(content)
print("✓ Archivo guardado")

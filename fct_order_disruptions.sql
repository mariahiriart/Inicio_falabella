-- models/marts/fct_order_disruptions.sql
-- Solo las ordenes con disrupcion detectada.
-- Tabla principal para el dashboard de operaciones y para ML.
-- Incluye atribucion de responsabilidad por tramo y actor.

{{
    config(
        materialized='table',
        schema='marts',
        indexes=[
            {'columns': ['logistic_order_id'], 'unique': true},
            {'columns': ['tramo_disruptivo']},
            {'columns': ['actor_responsable']},
            {'columns': ['year_month']},
        ]
    )
}}

with ordenes as (
    select * from {{ ref('fct_orders') }}
    where is_disrupted = true
),

final as (
    select
        logistic_order_id,
        year_month,
        source_order_type,
        service_category,
        is_high_season,

        -- Atribucion de la disrupcion
        tramo_disruptivo,
        actor_responsable,

        -- Tipo de disrupcion
        case
            when is_stuck           then 'orden_atascada'
            when is_sla_breach      then 'sla_breach'
            when tramo_disruptivo = 'deposito'      then 'demora_deposito'
            when tramo_disruptivo = 'despacho'      then 'demora_despacho'
            when tramo_disruptivo = 'ultima_milla'  then 'demora_ultima_milla'
            else 'otro'
        end                                             as tipo_disrupcion,

        -- Severidad basada en el delta respecto al p95
        case
            when is_stuck then 'critica'
            when is_sla_breach and hours_sla_delta > 48 then 'alta'
            when is_sla_breach then 'media'
            when greatest(
                coalesce(delta_deposito_vs_p95, 0),
                coalesce(delta_despacho_vs_p95, 0),
                coalesce(delta_ultima_milla_vs_p95, 0)
            ) > 48 then 'alta'
            else 'media'
        end                                             as severidad,

        -- Tiempos
        hours_deposito,
        hours_despacho,
        hours_ultima_milla,
        hours_total,
        hours_sla_delta,

        -- Deltas vs baseline
        delta_deposito_vs_p95,
        delta_despacho_vs_p95,
        delta_ultima_milla_vs_p95,

        -- Timestamps clave
        created_at,
        dispatched_at,
        completed_at,
        promised_date_to,

        -- Contexto
        seller_id,
        main_sku,
        total_items,

        -- Flags
        is_completed,
        is_cancelled,
        is_stuck,
        is_sla_breach

    from ordenes
)

select * from final
order by severidad desc, hours_total desc nulls last

-- models/intermediate/int_stage_baselines.sql
-- Calcula los percentiles de tiempo de cada tramo del flujo
-- por mes y por actor. Estos son los baselines contra los que
-- se compara cada orden para detectar disrupciones.
-- Solo usa ordenes completadas sin alta estacionalidad.

{{
    config(
        materialized='table',
        schema='intermediate'
    )
}}

with timeline as (
    select * from {{ ref('int_order_timeline') }}
    where
        is_completed = true
        and is_high_season = false
        and hours_total_cycle between 0 and 720  -- excluir outliers extremos
),

baselines as (
    select
        year_month,

        -- Tramo 1: Deposito (CREATED -> PACKED)
        percentile_cont(0.50) within group
            (order by hours_created_to_packed)          as p50_deposito,
        percentile_cont(0.75) within group
            (order by hours_created_to_packed)          as p75_deposito,
        percentile_cont(0.90) within group
            (order by hours_created_to_packed)          as p90_deposito,
        percentile_cont(0.95) within group
            (order by hours_created_to_packed)          as p95_deposito,

        -- Tramo 2: Despacho (PACKED -> DISPATCHED)
        percentile_cont(0.50) within group
            (order by hours_packed_to_dispatched)       as p50_despacho,
        percentile_cont(0.75) within group
            (order by hours_packed_to_dispatched)       as p75_despacho,
        percentile_cont(0.90) within group
            (order by hours_packed_to_dispatched)       as p90_despacho,
        percentile_cont(0.95) within group
            (order by hours_packed_to_dispatched)       as p95_despacho,

        -- Tramo 3: Ultima milla (DISPATCHED -> COMPLETED)
        percentile_cont(0.50) within group
            (order by hours_dispatched_to_completed)    as p50_ultima_milla,
        percentile_cont(0.75) within group
            (order by hours_dispatched_to_completed)    as p75_ultima_milla,
        percentile_cont(0.90) within group
            (order by hours_dispatched_to_completed)    as p90_ultima_milla,
        percentile_cont(0.95) within group
            (order by hours_dispatched_to_completed)    as p95_ultima_milla,

        -- Ciclo total
        percentile_cont(0.50) within group
            (order by hours_total_cycle)                as p50_total,
        percentile_cont(0.90) within group
            (order by hours_total_cycle)                as p90_total,
        percentile_cont(0.95) within group
            (order by hours_total_cycle)                as p95_total,

        -- Tasa de SLA breach del mes
        round(100.0 * sum(case when is_sla_breach then 1 else 0 end)
            / nullif(count(*), 0), 2)                   as pct_sla_breach,

        count(*)                                        as ordenes_base

    from timeline
    where hours_created_to_packed is not null
    group by year_month
)

select * from baselines
order by year_month

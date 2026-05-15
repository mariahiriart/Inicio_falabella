-- models/marts/fct_orders.sql
-- Una fila por orden. Tabla principal para análisis y ML.
-- Incluye tiempos de cada tramo, comparación contra baselines
-- y flags de calidad/anomalia.

{{
    config(
        materialized='table',
        schema='marts',
        indexes=[
            {'columns': ['logistic_order_id'], 'unique': true},
            {'columns': ['year_month']},
            {'columns': ['is_sla_breach']},
            {'columns': ['is_disrupted']},
        ]
    )
}}

with timeline as (
    select * from {{ ref('int_order_timeline') }}
),

baselines as (
    select * from {{ ref('int_stage_baselines') }}
),

-- Items por orden: seller y sku predominante
items as (
    select
        logistic_order_id,
        mode() within group (order by seller_id)        as main_seller_id,
        mode() within group (order by sku)              as main_sku,
        sum(quantity_number)                            as total_items,
        bool_or(has_insurance)                          as has_insurance,
        count(distinct sku)                             as distinct_skus
    from {{ ref('stg_event_items') }}
    group by logistic_order_id
),

final as (
    select
        t.logistic_order_id,
        t.source_order_type,
        t.fulfillment_order_type,
        t.year_month,
        t.service_category,
        t.is_click_and_collect,
        t.is_high_season,

        -- Actores clave
        t.created_by,
        t.packed_by,
        t.dispatched_by,
        t.completed_by,
        t.cancelled_by,

        -- Timestamps
        t.created_at,
        t.packed_at,
        t.dispatched_at,
        t.completed_at,
        t.cancelled_at,
        t.last_event_at,
        t.last_state,

        -- Promised date
        t.promised_date_to,
        t.promised_date_from,

        -- Flags de estado
        t.is_completed,
        t.is_cancelled,
        t.is_dispatched,
        t.is_packed,
        t.is_stuck,
        t.is_sla_breach,
        t.hours_sla_delta,

        -- Tiempos por tramo (horas)
        round(t.hours_created_to_packed::numeric, 2)    as hours_deposito,
        round(t.hours_packed_to_dispatched::numeric, 2) as hours_despacho,
        round(t.hours_dispatched_to_completed::numeric, 2) as hours_ultima_milla,
        round(t.hours_total_cycle::numeric, 2)          as hours_total,

        -- Desviacion de cada tramo respecto al baseline p95 del mes
        -- Positivo = tardó más de lo normal
        round((t.hours_created_to_packed
            - b.p95_deposito)::numeric, 2)              as delta_deposito_vs_p95,
        round((t.hours_packed_to_dispatched
            - b.p95_despacho)::numeric, 2)              as delta_despacho_vs_p95,
        round((t.hours_dispatched_to_completed
            - b.p95_ultima_milla)::numeric, 2)          as delta_ultima_milla_vs_p95,

        -- Tramo mas lento relativo a su baseline
        -- Este es el campo clave para atribucion de disrupcion
        case
            when t.hours_created_to_packed > b.p95_deposito
                and t.hours_created_to_packed >= coalesce(t.hours_packed_to_dispatched, 0)
                and t.hours_created_to_packed >= coalesce(t.hours_dispatched_to_completed, 0)
                then 'deposito'
            when t.hours_packed_to_dispatched > b.p95_despacho
                and t.hours_packed_to_dispatched >= coalesce(t.hours_created_to_packed, 0)
                and t.hours_packed_to_dispatched >= coalesce(t.hours_dispatched_to_completed, 0)
                then 'despacho'
            when t.hours_dispatched_to_completed > b.p95_ultima_milla
                then 'ultima_milla'
            else 'sin_disrupcion'
        end                                             as tramo_disruptivo,

        -- Actor responsable del tramo disruptivo
        case
            when t.hours_created_to_packed > b.p95_deposito
                and t.hours_created_to_packed >= coalesce(t.hours_packed_to_dispatched, 0)
                and t.hours_created_to_packed >= coalesce(t.hours_dispatched_to_completed, 0)
                then t.packed_by
            when t.hours_packed_to_dispatched > b.p95_despacho
                and t.hours_packed_to_dispatched >= coalesce(t.hours_created_to_packed, 0)
                and t.hours_packed_to_dispatched >= coalesce(t.hours_dispatched_to_completed, 0)
                then t.dispatched_by
            when t.hours_dispatched_to_completed > b.p95_ultima_milla
                then t.completed_by
            else null
        end                                             as actor_responsable,

        -- Flag general de disrupcion
        case
            when t.is_stuck then true
            when t.is_sla_breach then true
            when t.hours_created_to_packed > b.p95_deposito then true
            when t.hours_packed_to_dispatched > b.p95_despacho then true
            when t.hours_dispatched_to_completed > b.p95_ultima_milla then true
            else false
        end                                             as is_disrupted,

        -- Conteos de eventos
        t.total_events,
        t.total_actors,
        t.total_event_types,

        -- Items
        i.main_seller_id                                as seller_id,
        i.main_sku                                      as main_sku,
        i.total_items,
        i.distinct_skus,
        i.has_insurance

    from timeline t
    left join baselines b on t.year_month = b.year_month
    left join items i on t.logistic_order_id = i.logistic_order_id
)

select * from final

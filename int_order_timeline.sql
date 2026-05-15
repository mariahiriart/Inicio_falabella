-- models/intermediate/int_order_timeline.sql
-- Una fila por orden con el timeline completo de eventos.
-- Base para calcular tramos y detectar disrupciones.

{{
    config(
        materialized='table',
        schema='intermediate',
        indexes=[
            {'columns': ['logistic_order_id'], 'unique': true},
            {'columns': ['created_at']},
            {'columns': ['is_completed']},
        ]
    )
}}

with eventos as (
    select * from {{ ref('stg_events') }}
),

-- Timestamps de cada hito del ciclo de vida
hitos as (
    select
        logistic_order_id,
        source_order_type,
        fulfillment_order_type,
        year_month,

        -- Hito: CREATED
        min(case when event_type = 'FULFILMENT_ORDER_CREATED'
            then event_at end)                          as created_at,
        max(case when event_type = 'FULFILMENT_ORDER_CREATED'
            then executed_by end)                       as created_by,

        -- Hito: PACKED (sale del deposito)
        min(case when event_type = 'FULFILMENT_ORDER_ITEM_QUANTITY_PACKED'
            then event_at end)                          as packed_at,
        max(case when event_type = 'FULFILMENT_ORDER_ITEM_QUANTITY_PACKED'
            then executed_by end)                       as packed_by,

        -- Hito: DISPATCHED (sale hacia el cliente)
        min(case when event_type = 'FULFILMENT_ORDER_PACKAGES_DISPATCHED'
            then event_at end)                          as dispatched_at,
        max(case when event_type = 'FULFILMENT_ORDER_PACKAGES_DISPATCHED'
            then executed_by end)                       as dispatched_by,

        -- Hito: ITEM_DISPATCHED (confirmacion de items)
        min(case when event_type = 'FULFILMENT_ORDER_ITEM_QUANTITY_DISPATCHED'
            then event_at end)                          as item_dispatched_at,

        -- Hito: COMPLETED
        max(case when order_state = 'COMPLETED'
            then event_at end)                          as completed_at,
        max(case when order_state = 'COMPLETED'
            then executed_by end)                       as completed_by,

        -- Hito: CANCELLED
        max(case when event_type = 'FO_ITEMS_ANNULLED'
            then event_at end)                          as cancelled_at,
        max(case when event_type = 'FO_ITEMS_ANNULLED'
            then executed_by end)                       as cancelled_by,

        -- Ultimo evento general
        max(event_at)                                   as last_event_at,
        max(order_state)                                as last_state,

        -- Conteos
        count(*)                                        as total_events,
        count(distinct executed_by)                     as total_actors,
        count(distinct event_type)                      as total_event_types

    from eventos
    group by
        logistic_order_id,
        source_order_type,
        fulfillment_order_type,
        year_month
),

-- Agregar promised_date desde packages
promised as (
    select
        logistic_order_id,
        min(try_cast(promised_date_to as timestamp))    as promised_date_to,
        min(try_cast(promised_date_from as timestamp))  as promised_date_from,
        max(service_category)                           as service_category,
        bool_or(is_click_and_collect)                   as is_click_and_collect
    from {{ ref('stg_event_packages') }}
    where event_type = 'FULFILMENT_ORDER_PACKAGES_DISPATCHED'
    group by logistic_order_id
),

final as (
    select
        h.logistic_order_id,
        h.source_order_type,
        h.fulfillment_order_type,
        h.year_month,

        -- Timestamps de hitos
        h.created_at,
        h.created_by,
        h.packed_at,
        h.packed_by,
        h.dispatched_at,
        h.dispatched_by,
        h.item_dispatched_at,
        h.completed_at,
        h.completed_by,
        h.cancelled_at,
        h.cancelled_by,
        h.last_event_at,
        h.last_state,

        -- Promised date
        p.promised_date_to,
        p.promised_date_from,
        p.service_category,
        p.is_click_and_collect,

        -- Flags de estado
        case when h.completed_at is not null then true else false end    as is_completed,
        case when h.cancelled_at is not null then true else false end    as is_cancelled,
        case when h.dispatched_at is not null then true else false end   as is_dispatched,
        case when h.packed_at is not null then true else false end       as is_packed,

        -- Flag: orden atascada (nunca avanzó de NEW)
        case
            when h.created_at is not null
                and h.dispatched_at is null
                and h.cancelled_at is null
                and h.completed_at is null
            then true
            else false
        end                                                              as is_stuck,

        -- Tiempos de cada tramo en horas
        case when h.packed_at is not null and h.created_at is not null
            then extract(epoch from (h.packed_at - h.created_at)) / 3600.0
        end                                                              as hours_created_to_packed,

        case when h.dispatched_at is not null and h.packed_at is not null
            then extract(epoch from (h.dispatched_at - h.packed_at)) / 3600.0
        end                                                              as hours_packed_to_dispatched,

        case when h.completed_at is not null and h.dispatched_at is not null
            then extract(epoch from (h.completed_at - h.dispatched_at)) / 3600.0
        end                                                              as hours_dispatched_to_completed,

        case when h.completed_at is not null and h.created_at is not null
            then extract(epoch from (h.completed_at - h.created_at)) / 3600.0
        end                                                              as hours_total_cycle,

        -- SLA breach: llegó después de la fecha prometida?
        case
            when h.completed_at is not null and p.promised_date_to is not null
                and h.completed_at > p.promised_date_to
            then true
            when h.completed_at is not null and p.promised_date_to is not null
            then false
            else null  -- no se puede determinar
        end                                                              as is_sla_breach,

        -- Horas de retraso respecto a promised_date (positivo = tardó más)
        case
            when h.completed_at is not null and p.promised_date_to is not null
            then extract(epoch from (h.completed_at - p.promised_date_to)) / 3600.0
        end                                                              as hours_sla_delta,

        -- Mes de alta estacionalidad
        case
            when h.year_month in ('2025-06','2025-10','2025-11','2025-12')
            then true else false
        end                                                              as is_high_season,

        -- Conteos
        h.total_events,
        h.total_actors,
        h.total_event_types

    from hitos h
    left join promised p on h.logistic_order_id = p.logistic_order_id
)

select * from final

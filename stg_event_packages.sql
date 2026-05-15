-- models/staging/stg_event_packages.sql
-- Una fila por paquete por evento.
-- Extrae promised_date del MAP y limpia campos criticos.
-- NOTA: carrier y tracking_number estan vacios en los datos origen (confirmado).
-- El carrier se obtiene desde stg_events.executed_by en modelos intermediate.

{{
    config(
        materialized='view',
        schema='staging'
    )
}}

with source as (
    select * from {{ source('raw', 'event_packages') }}
),

limpio as (
    select
        -- Identificadores
        logistic_order_id,
        shipment_id,
        package_id,

        -- Tipo de evento y timestamps
        event_type,
        try_cast(event_dt as timestamp)                 as event_at,

        -- Geografia
        country,

        -- Estado del paquete
        package_status,

        -- Promised date — extraida del MAP
        -- dateTo es la fecha limite comprometida con el cliente (SLA breach)
        promised_date['dateTo']                         as promised_date_to,
        promised_date['dateFrom']                       as promised_date_from,
        promised_date['serviceCategory']                as service_category,
        promised_date['timeRangeFrom']                  as promised_time_from,
        promised_date['timeRangeTo']                    as promised_time_to,
        promised_date['collectAvailabilityDate']        as collect_availability_date,

        -- Flag click & collect
        case
            when promised_date['collectAvailabilityDate'] is not null then true
            else false
        end                                             as is_click_and_collect,

        -- Particion
        year_month

    from source
    where
        country = '{{ var("pais_foco") }}'
        and try_cast(event_dt as timestamp) >= '{{ var("fecha_minima") }}'
        and logistic_order_id is not null
)

select * from limpio

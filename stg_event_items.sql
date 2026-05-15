-- models/staging/stg_event_items.sql
-- Una fila por item por evento.
-- Desempaqueta el struct quantity y limpia seller/sku.

{{
    config(
        materialized='view',
        schema='staging'
    )
}}

with source as (
    select * from {{ source('raw', 'event_items') }}
),

limpio as (
    select
        -- Identificadores
        logistic_order_id,

        -- Evento
        event_type,
        try_cast(event_dt as timestamp)                 as event_at,

        -- Geografia
        country,

        -- Producto
        cast(sku as varchar)                            as sku,
        cast(seller_id as varchar)                      as seller_id,

        -- Cantidad — desempaquetar struct
        quantity.number                                 as quantity_number,
        quantity.unit                                   as quantity_unit,

        -- Stock reservado
        reserved_quantity,

        -- Seguro
        case
            when has_insurance = 1 then true
            else false
        end                                             as has_insurance,

        -- Particion
        year_month

    from source
    where
        country = '{{ var("pais_foco") }}'
        and try_cast(event_dt as timestamp) >= '{{ var("fecha_minima") }}'
        and logistic_order_id is not null
)

select * from limpio

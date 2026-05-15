-- models/staging/stg_events.sql
-- Una fila por evento logistico.
-- Limpieza, tipado y filtros basicos.
-- Fuente: parquets en S3 cargados en raw.events

{{
    config(
        materialized='view',
        schema='staging'
    )
}}

with source as (
    select * from {{ source('raw', 'events') }}
),

limpio as (
    select
        -- Identificadores
        h_logistic_order_id                             as logistic_order_id,
        h_source_order_id                               as source_order_id,
        d_logistic_order_id                             as d_logistic_order_id,

        -- Tipo de evento y entidad
        h_event_type                                    as event_type,
        h_entity_type                                   as entity_type,

        -- Geografia y canal
        h_country                                       as country,
        h_commerce                                      as commerce,

        -- Timestamps — casteo con timezone
        try_cast(h_datetime as timestamp)               as event_at,
        try_cast(d_executed_at as timestamp)            as executed_at,
        try_cast(publish_time as timestamp)             as published_at,

        -- Estado de la orden
        d_state                                         as order_state,

        -- Actores
        d_executed_by                                   as executed_by,

        -- Tipo de orden
        d_fulfillment_order_type                        as fulfillment_order_type,
        d_source_order_type                             as source_order_type,

        -- Flags
        d_is_international                              as is_international,
        cast(d_order_version as integer)                as order_version,

        -- Particiones (para filtros eficientes)
        year_month

    from source
    where
        -- Solo pais foco
        h_country = '{{ var("pais_foco") }}'
        -- Excluir eventos reprocesados fuera del rango valido
        and try_cast(h_datetime as timestamp) >= '{{ var("fecha_minima") }}'
        -- Excluir nulls en columnas criticas
        and h_logistic_order_id is not null
        and h_event_type is not null
        and h_datetime is not null
)

select * from limpio

SELECT 
    -- Attributs de la table d'agrégat / fait
    f.country_name,
    f.year,
    f.item_key,
    f.measurement_form,
    f.unit,
    f.value_avg,
    f.row_count,

    -- Attributs des dimensions croisées (optionnel)
    geo.country_name,
    ind.indicator_name,
    ind.indicator_key

FROM `opentrace-prod-5ga4.mart_dev.agg_economics_country_year` AS f

-- Jointure avec les dimensions via les clés explicites (_key)
LEFT JOIN `opentrace-prod-5ga4.mart_dev.dim_geography` AS geo 
    ON f.geography_key = geo.geography_key

LEFT JOIN `opentrace-prod-5ga4.mart_dev.dim_indicator` AS ind 
    ON f.item_key = ind.indicator_key

WHERE f.year >= 2020
ORDER BY f.country_name ASC, f.year DESC;

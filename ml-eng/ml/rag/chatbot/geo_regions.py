"""African region / zone → country expansion for retrieval, BQ, and geo purity.

Canonical catalog for political, REC, climatic, agroecological, and ops geographies.
Zones with ``countries`` expand to FAOSTAT/news-compatible country names.
Zones with empty ``countries`` are strip-only (never used as ``geo_country`` filters).

Default expand caps keep huge unions (SSA, COMESA, AU) bounded for BQ IN lists;
override via ``max_countries`` on ``expand_regions_in_decomposition``.
"""
from __future__ import annotations

import re
from typing import Any, TypedDict

# Soft default when expanding very large zones without an explicit caller cap.
DEFAULT_EXPAND_CAP = 54


class ZoneSpec(TypedDict, total=False):
    category: str
    aliases: tuple[str, ...]
    countries: tuple[str, ...]


# Member lists use names that match FAOSTAT / news geo filters in query_decomposer.
ZONE_SPEC: dict[str, ZoneSpec] = {
    # --- UN M49 / common geographic bands ---
    "west_africa": {
        "category": "un_band",
        "aliases": (
            "west africa",
            "western africa",
            "afrique de l'ouest",
            "afrique de louest",
            "afrique occidentale",
        ),
        "countries": (
            "Benin",
            "Burkina Faso",
            "Cabo Verde",
            "Côte d'Ivoire",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Liberia",
            "Mali",
            "Mauritania",
            "Niger",
            "Nigeria",
            "Senegal",
            "Sierra Leone",
            "Togo",
        ),
    },
    "east_africa": {
        "category": "un_band",
        "aliases": (
            "east africa",
            "eastern africa",
            "afrique de l'est",
            "afrique de lest",
            "afrique orientale",
        ),
        "countries": (
            "Burundi",
            "Comoros",
            "Djibouti",
            "Eritrea",
            "Ethiopia",
            "Kenya",
            "Madagascar",
            "Malawi",
            "Mauritius",
            "Mozambique",
            "Rwanda",
            "Seychelles",
            "Somalia",
            "South Sudan",
            "Tanzania",
            "Uganda",
            "Zambia",
            "Zimbabwe",
        ),
    },
    "southern_africa": {
        "category": "un_band",
        "aliases": (
            "southern africa",
            "south africa region",
            "afrique australe",
            "afrique du sud region",
        ),
        "countries": (
            "Angola",
            "Botswana",
            "Eswatini",
            "Lesotho",
            "Malawi",
            "Mozambique",
            "Namibia",
            "South Africa",
            "Zambia",
            "Zimbabwe",
        ),
    },
    "north_africa": {
        "category": "un_band",
        "aliases": (
            "north africa",
            "northern africa",
            "afrique du nord",
            "afrique septentrionale",
        ),
        "countries": (
            "Algeria",
            "Egypt",
            "Libya",
            "Morocco",
            "Sudan",
            "Tunisia",
        ),
    },
    "central_africa": {
        "category": "un_band",
        "aliases": (
            "central africa",
            "middle africa",
            "afrique centrale",
            "afrique du centre",
        ),
        "countries": (
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Congo",
            "Democratic Republic of the Congo",
            "Equatorial Guinea",
            "Gabon",
        ),
    },
    "sub_saharan_africa": {
        "category": "continental",
        "aliases": (
            "sub-saharan africa",
            "sub saharan africa",
            "subsaharan africa",
            "ssa",
            "afrique subsaharienne",
            "afrique sub-saharienne",
        ),
        # Union of west + east + central + southern (excludes Maghreb core).
        "countries": (
            "Angola",
            "Benin",
            "Botswana",
            "Burkina Faso",
            "Burundi",
            "Cabo Verde",
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Comoros",
            "Congo",
            "Côte d'Ivoire",
            "Democratic Republic of the Congo",
            "Djibouti",
            "Equatorial Guinea",
            "Eritrea",
            "Eswatini",
            "Ethiopia",
            "Gabon",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Kenya",
            "Lesotho",
            "Liberia",
            "Madagascar",
            "Malawi",
            "Mali",
            "Mauritania",
            "Mauritius",
            "Mozambique",
            "Namibia",
            "Niger",
            "Nigeria",
            "Rwanda",
            "Sao Tome and Principe",
            "Senegal",
            "Seychelles",
            "Sierra Leone",
            "Somalia",
            "South Africa",
            "South Sudan",
            "Tanzania",
            "Togo",
            "Uganda",
            "Zambia",
            "Zimbabwe",
        ),
    },
    # --- Continental / named regions ---
    "maghreb": {
        "category": "continental",
        "aliases": ("maghreb", "maghrib", "atlas region", "grande maghreb"),
        "countries": ("Algeria", "Libya", "Mauritania", "Morocco", "Tunisia"),
    },
    "sahel": {
        "category": "climate_belt",
        "aliases": (
            "sahel",
            "sahelian",
            "sahel region",
            "sahelian belt",
            "sahel zone",
            "region du sahel",
            "région du sahel",
        ),
        "countries": (
            "Burkina Faso",
            "Chad",
            "Mali",
            "Mauritania",
            "Niger",
            "Senegal",
            "Sudan",
        ),
    },
    "horn_of_africa": {
        "category": "continental",
        "aliases": (
            "horn of africa",
            "the horn",
            "corne de l'afrique",
            "corne de lafrique",
            "hoa",
        ),
        "countries": ("Djibouti", "Eritrea", "Ethiopia", "Somalia"),
    },
    "greater_horn_of_africa": {
        "category": "ops",
        "aliases": (
            "greater horn of africa",
            "ghoa",
            "greater horn",
        ),
        "countries": (
            "Djibouti",
            "Eritrea",
            "Ethiopia",
            "Kenya",
            "Somalia",
            "South Sudan",
            "Sudan",
            "Uganda",
        ),
    },
    "great_lakes": {
        "category": "continental",
        "aliases": (
            "great lakes",
            "african great lakes",
            "great lakes region",
            "grands lacs",
            "region des grands lacs",
        ),
        "countries": (
            "Burundi",
            "Democratic Republic of the Congo",
            "Kenya",
            "Rwanda",
            "Tanzania",
            "Uganda",
        ),
    },
    "congo_basin": {
        "category": "ecology",
        "aliases": (
            "congo basin",
            "congo rainforest",
            "congo basin forest",
            "bassin du congo",
            "foret du congo",
            "forêt du congo",
        ),
        "countries": (
            "Cameroon",
            "Central African Republic",
            "Congo",
            "Democratic Republic of the Congo",
            "Equatorial Guinea",
            "Gabon",
        ),
    },
    "gulf_of_guinea": {
        "category": "continental",
        "aliases": ("gulf of guinea", "golfe de guinee", "golfe de guinée"),
        "countries": (
            "Benin",
            "Cameroon",
            "Côte d'Ivoire",
            "Equatorial Guinea",
            "Gabon",
            "Ghana",
            "Guinea",
            "Liberia",
            "Nigeria",
            "Sao Tome and Principe",
            "Sierra Leone",
            "Togo",
        ),
    },
    "indian_ocean_islands": {
        "category": "continental",
        "aliases": (
            "indian ocean islands",
            "western indian ocean islands",
            "iles de locean indien",
            "îles de l'océan indien",
        ),
        "countries": ("Comoros", "Madagascar", "Mauritius", "Seychelles"),
    },
    "lusophone_africa": {
        "category": "linguistic",
        "aliases": (
            "lusophone africa",
            "portuguese speaking africa",
            "palop",
            "afrique lusophone",
        ),
        "countries": (
            "Angola",
            "Cabo Verde",
            "Guinea-Bissau",
            "Mozambique",
            "Sao Tome and Principe",
        ),
    },
    "francophone_africa": {
        "category": "linguistic",
        "aliases": ("francophone africa", "french speaking africa", "afrique francophone"),
        "countries": (
            "Benin",
            "Burkina Faso",
            "Burundi",
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Comoros",
            "Congo",
            "Côte d'Ivoire",
            "Democratic Republic of the Congo",
            "Djibouti",
            "Gabon",
            "Guinea",
            "Madagascar",
            "Mali",
            "Mauritania",
            "Niger",
            "Rwanda",
            "Senegal",
            "Togo",
        ),
    },
    "anglophone_africa": {
        "category": "linguistic",
        "aliases": ("anglophone africa", "english speaking africa", "afrique anglophone"),
        "countries": (
            "Botswana",
            "Eswatini",
            "Gambia",
            "Ghana",
            "Kenya",
            "Lesotho",
            "Liberia",
            "Malawi",
            "Namibia",
            "Nigeria",
            "Sierra Leone",
            "South Africa",
            "South Sudan",
            "Tanzania",
            "Uganda",
            "Zambia",
            "Zimbabwe",
        ),
    },
    # --- RECs and related ---
    "ecowas": {
        "category": "rec",
        "aliases": ("ecowas", "cedeao", "economic community of west african states"),
        "countries": (
            "Benin",
            "Burkina Faso",
            "Cabo Verde",
            "Côte d'Ivoire",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Liberia",
            "Mali",
            "Niger",
            "Nigeria",
            "Senegal",
            "Sierra Leone",
            "Togo",
        ),
    },
    "uemoa": {
        "category": "rec",
        "aliases": ("uemoa", "waemu", "west african economic and monetary union"),
        "countries": (
            "Benin",
            "Burkina Faso",
            "Côte d'Ivoire",
            "Guinea-Bissau",
            "Mali",
            "Niger",
            "Senegal",
            "Togo",
        ),
    },
    "cemac": {
        "category": "rec",
        "aliases": (
            "cemac",
            "economic and monetary community of central africa",
            "communauté économique et monétaire de l'afrique centrale",
        ),
        "countries": (
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Congo",
            "Equatorial Guinea",
            "Gabon",
        ),
    },
    "eccas": {
        "category": "rec",
        "aliases": (
            "eccas",
            "ceeac",
            "economic community of central african states",
        ),
        "countries": (
            "Angola",
            "Burundi",
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Congo",
            "Democratic Republic of the Congo",
            "Equatorial Guinea",
            "Gabon",
            "Rwanda",
            "Sao Tome and Principe",
        ),
    },
    "eac": {
        "category": "rec",
        "aliases": ("eac", "east african community", "communaute d afrique de l est"),
        "countries": (
            "Burundi",
            "Democratic Republic of the Congo",
            "Kenya",
            "Rwanda",
            "Somalia",
            "South Sudan",
            "Tanzania",
            "Uganda",
        ),
    },
    "igad": {
        "category": "rec",
        "aliases": (
            "igad",
            "intergovernmental authority on development",
        ),
        "countries": (
            "Djibouti",
            "Eritrea",
            "Ethiopia",
            "Kenya",
            "Somalia",
            "South Sudan",
            "Sudan",
            "Uganda",
        ),
    },
    "sadc": {
        "category": "rec",
        "aliases": (
            "sadc",
            "southern african development community",
        ),
        "countries": (
            "Angola",
            "Botswana",
            "Comoros",
            "Democratic Republic of the Congo",
            "Eswatini",
            "Lesotho",
            "Madagascar",
            "Malawi",
            "Mauritius",
            "Mozambique",
            "Namibia",
            "Seychelles",
            "South Africa",
            "Tanzania",
            "Zambia",
            "Zimbabwe",
        ),
    },
    "comesa": {
        "category": "rec",
        "aliases": (
            "comesa",
            "common market for eastern and southern africa",
        ),
        "countries": (
            "Burundi",
            "Comoros",
            "Democratic Republic of the Congo",
            "Djibouti",
            "Egypt",
            "Eritrea",
            "Eswatini",
            "Ethiopia",
            "Kenya",
            "Libya",
            "Madagascar",
            "Malawi",
            "Mauritius",
            "Rwanda",
            "Seychelles",
            "Somalia",
            "Sudan",
            "Tunisia",
            "Uganda",
            "Zambia",
            "Zimbabwe",
        ),
    },
    "amu": {
        "category": "rec",
        "aliases": (
            "amu",
            "uma",
            "arab maghreb union",
            "union du maghreb arabe",
        ),
        "countries": ("Algeria", "Libya", "Mauritania", "Morocco", "Tunisia"),
    },
    "cilss": {
        "category": "rec",
        "aliases": (
            "cilss",
            "permanent interstate committee for drought control in the sahel",
        ),
        "countries": (
            "Burkina Faso",
            "Cabo Verde",
            "Chad",
            "Gambia",
            "Guinea",
            "Guinea-Bissau",
            "Mali",
            "Mauritania",
            "Niger",
            "Senegal",
        ),
    },
    "lcbc": {
        "category": "basin",
        "aliases": (
            "lcbc",
            "lake chad basin commission",
            "commission du bassin du lac tchad",
        ),
        "countries": ("Cameroon", "Central African Republic", "Chad", "Niger", "Nigeria"),
    },
    "omvs": {
        "category": "basin",
        "aliases": (
            "omvs",
            "senegal river basin development authority",
            "organisation pour la mise en valeur du fleuve senegal",
        ),
        "countries": ("Guinea", "Mali", "Mauritania", "Senegal"),
    },
    "mru": {
        "category": "rec",
        "aliases": ("mru", "mano river union"),
        "countries": ("Côte d'Ivoire", "Guinea", "Liberia", "Sierra Leone"),
    },
    "ioc": {
        "category": "rec",
        "aliases": (
            "ioc",
            "indian ocean commission",
            "commission de l'ocean indien",
        ),
        "countries": ("Comoros", "Madagascar", "Mauritius", "Seychelles"),
    },
    "nile_basin": {
        "category": "basin",
        "aliases": (
            "nile basin",
            "nile basin initiative",
            "nbi",
            "bassin du nil",
        ),
        "countries": (
            "Burundi",
            "Democratic Republic of the Congo",
            "Egypt",
            "Eritrea",
            "Ethiopia",
            "Kenya",
            "Rwanda",
            "South Sudan",
            "Sudan",
            "Tanzania",
            "Uganda",
        ),
    },
    "niger_river_basin": {
        "category": "basin",
        "aliases": (
            "niger river basin",
            "niger basin",
            "bassin du niger",
            "nba",
        ),
        "countries": (
            "Benin",
            "Burkina Faso",
            "Cameroon",
            "Chad",
            "Côte d'Ivoire",
            "Guinea",
            "Mali",
            "Niger",
            "Nigeria",
        ),
    },
    "senegal_river_basin": {
        "category": "basin",
        "aliases": ("senegal river basin", "bassin du fleuve senegal"),
        "countries": ("Guinea", "Mali", "Mauritania", "Senegal"),
    },
    "zambezi_basin": {
        "category": "basin",
        "aliases": ("zambezi", "zambezi basin", "zambezi river basin"),
        "countries": (
            "Angola",
            "Botswana",
            "Malawi",
            "Mozambique",
            "Namibia",
            "Tanzania",
            "Zambia",
            "Zimbabwe",
        ),
    },
    "volta_basin": {
        "category": "basin",
        "aliases": ("volta", "volta basin", "volta river basin"),
        "countries": ("Benin", "Burkina Faso", "Côte d'Ivoire", "Ghana", "Mali", "Togo"),
    },
    "lake_victoria_basin": {
        "category": "basin",
        "aliases": ("lake victoria basin", "victoria basin", "bassin du lac victoria"),
        "countries": ("Burundi", "Kenya", "Rwanda", "Tanzania", "Uganda"),
    },
    "lake_chad_basin": {
        "category": "ops",
        "aliases": (
            "lake chad basin",
            "lac tchad",
            "bassin du lac tchad",
            "lake chad",
        ),
        "countries": ("Cameroon", "Central African Republic", "Chad", "Niger", "Nigeria"),
    },
    # --- Climate / ecology belts ---
    "sudano_sahelian": {
        "category": "climate_belt",
        "aliases": (
            "sudano-sahelian",
            "sudano sahelian",
            "soudano-sahelien",
            "soudano-sahélien",
        ),
        "countries": (
            "Burkina Faso",
            "Chad",
            "Mali",
            "Mauritania",
            "Niger",
            "Nigeria",
            "Senegal",
            "Sudan",
        ),
    },
    "sudanian_zone": {
        "category": "agroeco",
        "aliases": (
            "sudanian zone",
            "sudan belt",
            "sudanian savanna",
            "soudanian",
            "zone soudanienne",
            # Avoid bare "sudan" — that is a country.
            "sudanian",
        ),
        "countries": (
            "Benin",
            "Burkina Faso",
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Côte d'Ivoire",
            "Ghana",
            "Guinea",
            "Mali",
            "Nigeria",
            "Senegal",
            "South Sudan",
            "Sudan",
            "Togo",
        ),
    },
    "guinean_zone": {
        "category": "agroeco",
        "aliases": (
            "guinean zone",
            "guinean forest-savanna",
            "guinean forest savanna",
            "guineo-congolian",
            "guinea-congolian",
            "zone guineenne",
            "zone guinéenne",
        ),
        "countries": (
            "Benin",
            "Cameroon",
            "Côte d'Ivoire",
            "Ghana",
            "Guinea",
            "Liberia",
            "Nigeria",
            "Sierra Leone",
            "Togo",
        ),
    },
    "kalahari": {
        "category": "ecology",
        "aliases": ("kalahari", "kalahari desert", "kalahari basin"),
        "countries": ("Botswana", "Namibia", "South Africa"),
    },
    "namib": {
        "category": "ecology",
        "aliases": ("namib", "namib desert"),
        "countries": ("Angola", "Namibia"),
    },
    "sahara": {
        "category": "ecology",
        "aliases": ("sahara", "saharan", "sahara desert", "desert du sahara"),
        "countries": (
            "Algeria",
            "Chad",
            "Egypt",
            "Libya",
            "Mali",
            "Mauritania",
            "Morocco",
            "Niger",
            "Sudan",
            "Tunisia",
        ),
    },
    "cape_floristic": {
        "category": "ecology",
        "aliases": ("cape floristic", "fynbos", "cape floral kingdom"),
        "countries": ("South Africa",),
    },
    "miombo": {
        "category": "ecology",
        "aliases": ("miombo", "miombo woodland", "miombo belt"),
        "countries": (
            "Angola",
            "Democratic Republic of the Congo",
            "Malawi",
            "Mozambique",
            "Tanzania",
            "Zambia",
            "Zimbabwe",
        ),
    },
    # --- Agroecological / farming systems (concrete member sets) ---
    "highland_agriculture": {
        "category": "agroeco",
        "aliases": (
            "highland agriculture",
            "african highlands",
            "east african highlands",
            "ethiopian highlands",
            "coffee highlands",
        ),
        "countries": ("Burundi", "Ethiopia", "Kenya", "Rwanda", "Uganda"),
    },
    "coastal_west_africa": {
        "category": "agroeco",
        "aliases": ("coastal west africa", "west african coast", "littoral ouest africain"),
        "countries": (
            "Benin",
            "Côte d'Ivoire",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Liberia",
            "Nigeria",
            "Senegal",
            "Sierra Leone",
            "Togo",
        ),
    },
    "inland_west_africa": {
        "category": "agroeco",
        "aliases": ("inland west africa", "interior west africa"),
        "countries": ("Burkina Faso", "Mali", "Niger"),
    },
    "cocoa_belt": {
        "category": "agroeco",
        "aliases": ("cocoa belt", "cacao belt", "west african cocoa"),
        "countries": ("Cameroon", "Côte d'Ivoire", "Ghana", "Nigeria", "Togo"),
    },
    "cotton_zone_sahel": {
        "category": "agroeco",
        "aliases": ("cotton zone", "sahel cotton", "cotton belt sahel"),
        "countries": ("Burkina Faso", "Chad", "Mali", "Senegal"),
    },
    "maize_belt_esa": {
        "category": "agroeco",
        "aliases": (
            "maize belt",
            "eastern southern africa maize",
            "esa maize belt",
        ),
        "countries": (
            "Kenya",
            "Malawi",
            "South Africa",
            "Tanzania",
            "Zambia",
            "Zimbabwe",
        ),
    },
    "rice_systems_west_africa": {
        "category": "agroeco",
        "aliases": ("rice systems west africa", "west african rice", "riziculture ouest afrique"),
        "countries": (
            "Benin",
            "Côte d'Ivoire",
            "Ghana",
            "Guinea",
            "Mali",
            "Nigeria",
            "Senegal",
        ),
    },
    "livestock_corridor_horn_sahel": {
        "category": "agroeco",
        "aliases": (
            "livestock corridor",
            "pastoral corridor",
            "horn sahel livestock",
        ),
        "countries": (
            "Djibouti",
            "Ethiopia",
            "Kenya",
            "Mali",
            "Niger",
            "Somalia",
            "South Sudan",
            "Sudan",
        ),
    },
    # --- Food-security / ops geographies ---
    "central_sahel": {
        "category": "ops",
        "aliases": ("central sahel", "sahel central", "sahel centre"),
        "countries": ("Burkina Faso", "Mali", "Niger"),
    },
    "liptako_gourma": {
        "category": "ops",
        "aliases": ("liptako-gourma", "liptako gourma", "liptako"),
        "countries": ("Burkina Faso", "Mali", "Niger"),
    },
    "southern_africa_drought_corridor": {
        "category": "ops",
        "aliases": (
            "southern africa drought",
            "southern africa drought corridor",
            "esa drought corridor",
        ),
        "countries": (
            "Botswana",
            "Eswatini",
            "Lesotho",
            "Malawi",
            "Mozambique",
            "Namibia",
            "South Africa",
            "Zambia",
            "Zimbabwe",
        ),
    },
    "red_sea_coast": {
        "category": "ops",
        "aliases": ("red sea coast", "red sea africa", "western red sea"),
        "countries": ("Djibouti", "Egypt", "Eritrea", "Sudan"),
    },
    # --- Strip-only (aliases never become geo_country; no expand) ---
    "africa": {
        "category": "continental",
        "aliases": (
            "africa",
            "african continent",
            "continent africain",
            "afrique",
        ),
        "countries": (),
    },
    "african_union": {
        "category": "rec",
        "aliases": ("african union", "au", "union africaine"),
        "countries": (),
    },
    "afcfta": {
        "category": "rec",
        "aliases": (
            "afcfta",
            "african continental free trade area",
            "zone de libre-echange continentale africaine",
        ),
        "countries": (),
    },
    "cen_sad": {
        "category": "rec",
        "aliases": ("cen-sad", "censad", "community of sahel-saharan states"),
        "countries": (),
    },
    "afrotropical": {
        "category": "ecology",
        "aliases": ("afrotropical", "afrotropics", "afrotropic"),
        "countries": (),
    },
    "arid_africa": {
        "category": "climate_belt",
        "aliases": ("arid africa", "arid zone africa", "drylands africa"),
        "countries": (),
    },
    "semi_arid_africa": {
        "category": "climate_belt",
        "aliases": ("semi-arid africa", "semi arid africa", "semiarid africa"),
        "countries": (),
    },
    "humid_tropics_africa": {
        "category": "climate_belt",
        "aliases": ("humid tropics africa", "humid tropical africa"),
        "countries": (),
    },
    "pastoral_belt": {
        "category": "agroeco",
        "aliases": ("pastoral belt", "pastoralism africa", "pastoral systems"),
        "countries": (),
    },
    "agro_pastoral": {
        "category": "agroeco",
        "aliases": ("agro-pastoral", "agropastoral", "agro pastoral"),
        "countries": (),
    },
    "mixed_crop_livestock": {
        "category": "agroeco",
        "aliases": ("mixed crop-livestock", "mixed crop livestock", "crop livestock systems"),
        "countries": (),
    },
    "cereal_root_crop_zone": {
        "category": "agroeco",
        "aliases": ("cereal-root crop zone", "cereal root crop", "root crop zone"),
        "countries": (),
    },
    "lowland_agriculture": {
        "category": "agroeco",
        "aliases": ("lowland agriculture", "african lowlands", "lowland farming"),
        "countries": (),
    },
}

# Backward-compatible views (keys use spaces for historical callers).
_KEY_TO_LEGACY: dict[str, str] = {
    "west_africa": "west africa",
    "east_africa": "east africa",
    "southern_africa": "southern africa",
    "north_africa": "north africa",
    "central_africa": "central africa",
    "sub_saharan_africa": "sub-saharan africa",
}


def _legacy_key(zone_key: str) -> str:
    return _KEY_TO_LEGACY.get(zone_key, zone_key.replace("_", " "))


REGION_COUNTRIES: dict[str, tuple[str, ...]] = {}
for _zk, _spec in ZONE_SPEC.items():
    _countries = tuple(_spec.get("countries") or ())
    if not _countries:
        continue
    REGION_COUNTRIES[_legacy_key(_zk)] = _countries
    REGION_COUNTRIES[_zk] = _countries

# Longer aliases first for substring matching.
_REGION_ALIASES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            (alias.lower(), zone_key)
            for zone_key, spec in ZONE_SPEC.items()
            for alias in (spec.get("aliases") or ())
        ),
        key=lambda pair: (-len(pair[0]), pair[0]),
    )
)

_ZONE_LABELS_LOWER: frozenset[str] = frozenset()


def _rebuild_zone_labels() -> frozenset[str]:
    labels: set[str] = {
        "global",
        "worldwide",
        "international",
    }
    for zone_key, spec in ZONE_SPEC.items():
        labels.add(zone_key.replace("_", " ").lower())
        labels.add(zone_key.lower())
        labels.add(_legacy_key(zone_key).lower())
        for alias in spec.get("aliases") or ():
            labels.add(str(alias).strip().lower())
    return frozenset(labels)


_ZONE_LABELS_LOWER = _rebuild_zone_labels()


def all_non_country_geo_labels() -> frozenset[str]:
    """Every zone key/alias — never valid as a single-country geo filter."""
    return _ZONE_LABELS_LOWER


def is_zone_label(value: str | None) -> bool:
    if not value or not str(value).strip():
        return False
    return str(value).strip().lower() in _ZONE_LABELS_LOWER


def detect_regions_in_text(text: str) -> list[str]:
    """Return canonical zone keys found in text (order preserved, unique).

    Prefers legacy spaced keys (``west africa``) when defined for backward
    compatibility with tests and callers; otherwise returns the ZONE_SPEC key.
    """
    q = (text or "").lower()
    found: list[str] = []
    seen: set[str] = set()
    for alias, key in _REGION_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", q):
            out_key = _legacy_key(key)
            if out_key not in seen:
                seen.add(out_key)
                found.append(out_key)
    return found


def countries_for_regions(region_keys: list[str], *, max_countries: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    cap = max_countries if max_countries is not None else DEFAULT_EXPAND_CAP
    for key in region_keys:
        members = (
            REGION_COUNTRIES.get(key)
            or ZONE_SPEC.get(key, {}).get("countries")
            or ZONE_SPEC.get(key.replace(" ", "_").replace("-", "_"), {}).get("countries")
            or ()
        )
        for c in members:
            if c not in seen:
                seen.add(c)
                out.append(c)
            if len(out) >= cap:
                return out
    return out


def expand_regions_in_decomposition(
    decomposition: dict[str, Any],
    query: str,
    *,
    max_countries: int | None = None,
) -> dict[str, Any]:
    """
    Replace region/zone labels with member countries in decomposition.geography.

    Scans the user query and existing geography/entities for zone tokens.
    Strip-only zones clear the label from geography and stamp expanded_regions
    without inventing countries.
    """
    if not isinstance(decomposition, dict):
        return decomposition

    texts = [query or ""]
    for key in ("geography", "entities"):
        raw = decomposition.get(key)
        if isinstance(raw, list):
            texts.extend(str(x) for x in raw)

    blob = " ".join(texts)
    regions = detect_regions_in_text(blob)
    if not regions:
        return decomposition

    countries = countries_for_regions(regions, max_countries=max_countries)
    out = dict(decomposition)
    existing = out.get("geography")
    existing_list = [str(g).strip() for g in existing] if isinstance(existing, list) else []
    kept = [g for g in existing_list if g and not is_zone_label(g)]
    merged: list[str] = []
    seen: set[str] = set()
    cap = max_countries if max_countries is not None else DEFAULT_EXPAND_CAP
    for c in kept + countries:
        if c not in seen:
            seen.add(c)
            merged.append(c)
        if len(merged) >= cap:
            break
    out["geography"] = merged
    out["expanded_regions"] = regions
    return out


__all__ = [
    "DEFAULT_EXPAND_CAP",
    "REGION_COUNTRIES",
    "ZONE_SPEC",
    "all_non_country_geo_labels",
    "countries_for_regions",
    "detect_regions_in_text",
    "expand_regions_in_decomposition",
    "is_zone_label",
]

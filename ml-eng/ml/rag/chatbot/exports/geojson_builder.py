"""Export observations as RFC 7946 features using verified locations only."""
from __future__ import annotations

import json
import math
from typing import Any

from geojson_pydantic import Feature, FeatureCollection


def _validate_positions(coordinates: Any) -> None:
    # geojson-pydantic validates geometry shapes; enforce WGS84 ranges as well.
    if not coordinates:
        return
    if isinstance(coordinates[0], (tuple, list)):
        for part in coordinates:
            _validate_positions(part)
        return
    if not all(math.isfinite(n) for n in coordinates):
        raise ValueError("GeoJSON coordinates must be finite")
    longitude, latitude = coordinates[:2]
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError("GeoJSON requires WGS84 longitude/latitude coordinates")


def _validate_geometry(geometry: Any) -> None:
    if geometry.type == "GeometryCollection":
        for part in geometry.geometries:
            _validate_geometry(part)
    else:
        _validate_positions(geometry.coordinates)


def _reject_legacy_crs(geometry: Any) -> None:
    if not isinstance(geometry, dict):
        return
    if "crs" in geometry:
        raise ValueError("GeoJSON requires WGS84 geometry without a legacy CRS")
    if geometry.get("type") == "GeometryCollection":
        for part in geometry.get("geometries") or []:
            _reject_legacy_crs(part)


def build_geojson(
    rows: list[dict[str, Any]], *, filename: str = "export.geojson",
) -> tuple[bytes, str]:
    if not rows:
        raise ValueError("No observations available for GeoJSON export")
    features = []
    for row in rows:
        geometry = row.get("geometry")
        if isinstance(geometry, str):
            geometry = json.loads(geometry)
        _reject_legacy_crs(geometry)
        if geometry is None:
            latitude, longitude = row.get("latitude"), row.get("longitude")
            if latitude is not None or longitude is not None:
                if latitude is None or longitude is None:
                    raise ValueError("Both latitude and longitude are required for a point")
                geometry = {"type": "Point", "coordinates": [longitude, latitude]}
        feature = Feature.model_validate({
            "type": "Feature", "geometry": geometry,
            "properties": {key: value for key, value in row.items() if key != "geometry"},
        }, strict=True)
        if feature.geometry is not None:
            _validate_geometry(feature.geometry)
        features.append(feature)
    collection = FeatureCollection(type="FeatureCollection", features=features)
    name = filename if filename.endswith(".geojson") else f"{filename}.geojson"
    return collection.model_dump_json().encode("utf-8"), name

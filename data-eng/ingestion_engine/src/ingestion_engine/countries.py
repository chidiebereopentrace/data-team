"""Country names and ISO codes used to fill country_iso3."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_COUNTRY_FILE = Path(__file__).with_name("country_names.yaml")
AFRICA_FILE = Path(__file__).with_name("africa_iso3.txt")


def norm_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.casefold().split())


@dataclass
class Countries:
    to_iso3: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)

    def resolve(self, value: str, standard: str) -> str | None:
        key = norm_key(value)
        if not key:
            return None
        if standard == "iso3":
            code = value.strip().upper()
            if code in self.names:
                return code
            return None
        if standard == "iso2":
            return self.to_iso3.get(key)
        if standard == "name":
            return self.to_iso3.get(key)
        return None

    def country_name(self, iso3: str) -> str:
        return self.names.get(iso3, "")


def load_africa(path: Path | None = None) -> frozenset[str]:
    """Return the 54 African ISO3 codes a subject country must match."""
    file = path or AFRICA_FILE
    codes = []
    for line in file.read_text(encoding="utf-8").splitlines():
        code = line.strip().upper()
        if code and not code.startswith("#"):
            codes.append(code)
    return frozenset(codes)


def load_countries(path: Path | None = None) -> Countries:
    file = path or DEFAULT_COUNTRY_FILE
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    countries = Countries()
    for row in data.get("countries") or []:
        name = str(row["name"]).strip()
        iso2 = str(row["iso2"]).strip().upper()
        iso3 = str(row["iso3"]).strip().upper()
        countries.names[iso3] = name
        labels = [name, iso2, iso3, *[str(alias) for alias in row.get("aliases") or []]]
        for label in labels:
            if not str(label).strip():
                continue
            countries.to_iso3[norm_key(label)] = iso3
    return countries

"""Query-form catalog for African agricultural intelligence.

A form is a template. Country, year shape, indicator, and crop are slots filled
from the warehouse inventory. The shape declaration is what the compiler must
emit. No country, year, or indicator is hardcoded into the compiler.
"""
from __future__ import annotations

from dataclasses import dataclass

from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.bq_sql_validate import metric_discriminator_names
from ml.rag.chatbot.bq_table_schema_yaml import (
    _AFRICA_COUNTRY_ISO3,
    compile_table_bind_contract,
    measure_columns,
    measure_columns_mart,
    value_samples_for_mart_tables,
    year_column,
)
from ml.rag.chatbot.compile_sql_from_bind import compile_sql_from_bind
from ml.rag.chatbot.geo_regions import ZONE_SPEC

YEAR_SHAPES = ("single_year", "span", "latest")


@dataclass(frozen=True)
class QueryForm:
    form_id: str
    template: str
    shape: str
    year_shape: str
    named_discriminator: bool

    def render(
        self,
        *,
        country: str,
        year: int | str = "{year}",
        year_end: int | str = "{year_end}",
        indicator: str,
        crop: str = "{crop}",
        discriminator: str = "{discriminator}",
    ) -> str:
        return self.template.format(
            country=country,
            year=year,
            year_end=year_end,
            indicator=indicator,
            crop=crop,
            discriminator=discriminator,
        )


def _forms_from(
    frames: tuple[str, ...],
    *,
    shape: str,
    year_shape: str,
    named_discriminator: bool,
    prefix: str,
) -> list[QueryForm]:
    forms: list[QueryForm] = []
    for index, template in enumerate(frames):
        forms.append(
            QueryForm(
                form_id=f"{prefix}-{index:03d}",
                template=template,
                shape=shape,
                year_shape=year_shape,
                named_discriminator=named_discriminator,
            )
        )
    return forms


_POINT_FRAMES = (
    "what is the {indicator} in {country} in {year}",
    "what is the {indicator} of {crop} in {country} in {year}",
    "how much {indicator} did {country} record in {year}",
    "how much {indicator} of {crop} did {country} record in {year}",
    "{indicator} in {country} for {year}",
    "{indicator} of {crop} in {country} for {year}",
    "give me the {indicator} in {country} in {year}",
    "give me the {indicator} of {crop} in {country} in {year}",
    "show the {indicator} for {country} in {year}",
    "show the {indicator} of {crop} for {country} in {year}",
    "tell me the {indicator} in {country} during {year}",
    "tell me the {indicator} of {crop} in {country} during {year}",
    "what was the {indicator} in {country} in {year}",
    "what was the {indicator} of {crop} in {country} in {year}",
    "report the {indicator} for {country} in {year}",
    "report the {indicator} of {crop} for {country} in {year}",
    "find the {indicator} in {country} in {year}",
    "find the {indicator} of {crop} in {country} in {year}",
    "state the {indicator} of {country} in {year}",
    "state the {indicator} of {crop} for {country} in {year}",
    "lookup the {indicator} in {country} in {year}",
    "lookup the {indicator} of {crop} in {country} in {year}",
    "value of {indicator} in {country} in {year}",
    "value of {indicator} of {crop} in {country} in {year}",
)

_NAMED_POINT_FRAMES = (
    "what is the {discriminator} {indicator} in {country} in {year}",
    "what is the {discriminator} {indicator} of {crop} in {country} in {year}",
    "how much {discriminator} {indicator} did {country} record in {year}",
    "how much {discriminator} {indicator} of {crop} did {country} record in {year}",
    "{discriminator} {indicator} in {country} for {year}",
    "{discriminator} {indicator} of {crop} in {country} for {year}",
    "give me the {discriminator} {indicator} in {country} in {year}",
    "give me the {discriminator} {indicator} of {crop} in {country} in {year}",
    "show the {discriminator} {indicator} for {country} in {year}",
    "show the {discriminator} {indicator} of {crop} for {country} in {year}",
)

_SERIES_FRAMES = (
    "{indicator} trend in {country} from {year} till {year_end}",
    "{indicator} of {crop} trend in {country} from {year} till {year_end}",
    "what is the trend of {indicator} in {country} from {year} till {year_end}",
    "what is the trend of {indicator} of {crop} in {country} from {year} till {year_end}",
    "show the {indicator} series for {country} from {year} till {year_end}",
    "show the {indicator} series of {crop} for {country} from {year} till {year_end}",
    "how did {indicator} in {country} change from {year} till {year_end}",
    "how did {indicator} of {crop} in {country} change from {year} till {year_end}",
    "{indicator} over time in {country} from {year} till {year_end}",
    "{indicator} of {crop} over time in {country} from {year} till {year_end}",
    "plot {indicator} in {country} between {year} and {year_end}",
    "plot {indicator} of {crop} in {country} between {year} and {year_end}",
    "track {indicator} in {country} from {year} till {year_end}",
    "track {indicator} of {crop} in {country} from {year} till {year_end}",
    "{indicator} time series in {country} from {year} till {year_end}",
    "{indicator} of {crop} time series in {country} from {year} till {year_end}",
    "annual {indicator} in {country} from {year} till {year_end}",
    "annual {indicator} of {crop} in {country} from {year} till {year_end}",
)

_NAMED_SERIES_FRAMES = (
    "{discriminator} {indicator} trend in {country} from {year} till {year_end}",
    "{discriminator} {indicator} of {crop} trend in {country} from {year} till {year_end}",
    "what is the trend of {discriminator} {indicator} in {country} from {year} till {year_end}",
    "what is the trend of {discriminator} {indicator} of {crop} in {country} from {year} till {year_end}",
    "show the {discriminator} {indicator} series for {country} from {year} till {year_end}",
    "show the {discriminator} {indicator} series of {crop} for {country} from {year} till {year_end}",
    "how did {discriminator} {indicator} in {country} change from {year} till {year_end}",
    "how did {discriminator} {indicator} of {crop} in {country} change from {year} till {year_end}",
)

_LATEST_FRAMES = (
    "latest {indicator} in {country}",
    "latest {indicator} of {crop} in {country}",
    "what is the latest {indicator} in {country}",
    "what is the latest {indicator} of {crop} in {country}",
    "most recent {indicator} for {country}",
    "most recent {indicator} of {crop} for {country}",
    "current {indicator} in {country}",
    "current {indicator} of {crop} in {country}",
    "give me the latest {indicator} in {country}",
    "give me the latest {indicator} of {crop} in {country}",
    "newest {indicator} figure for {country}",
    "newest {indicator} figure for {crop} in {country}",
    "last available {indicator} in {country}",
    "last available {indicator} of {crop} in {country}",
    "up to date {indicator} for {country}",
    "up to date {indicator} of {crop} for {country}",
)

_NAMED_LATEST_FRAMES = (
    "latest {discriminator} {indicator} in {country}",
    "latest {discriminator} {indicator} of {crop} in {country}",
    "what is the latest {discriminator} {indicator} in {country}",
    "what is the latest {discriminator} {indicator} of {crop} in {country}",
    "most recent {discriminator} {indicator} for {country}",
    "most recent {discriminator} {indicator} of {crop} for {country}",
)

_REPORT_FRAMES = (
    "brief me on {indicator} in {country}",
    "what are researchers saying about {indicator} in {country}",
    "summarize reporting on {indicator} of {crop} in {country}",
    "what is the news on {indicator} in {country} in {year}",
    "what does the literature say about {indicator} in {country}",
    "policy context for {indicator} in {country}",
    "explain {indicator} of {crop} in {country}",
    "background on {indicator} in {country} from {year} till {year_end}",
    "what should I know about {indicator} in {country}",
    "narrative on {indicator} of {crop} in {country} in {year}",
    "recent reporting on {indicator} in {country}",
    "synthesis of {indicator} in {country}",
    "what reports cover {indicator} in {country}",
    "what reports cover {indicator} of {crop} in {country}",
    "evidence review of {indicator} in {country} in {year}",
    "evidence review of {indicator} of {crop} in {country}",
    "qualitative picture of {indicator} in {country}",
    "qualitative picture of {indicator} of {crop} in {country} from {year} till {year_end}",
)


def build_query_forms() -> tuple[QueryForm, ...]:
    forms = (
        _forms_from(_POINT_FRAMES, shape="point", year_shape="single_year", named_discriminator=False, prefix="point")
        + _forms_from(_NAMED_POINT_FRAMES, shape="point", year_shape="single_year", named_discriminator=True, prefix="point-named")
        + _forms_from(_SERIES_FRAMES, shape="series", year_shape="span", named_discriminator=False, prefix="series")
        + _forms_from(_NAMED_SERIES_FRAMES, shape="series", year_shape="span", named_discriminator=True, prefix="series-named")
        + _forms_from(_LATEST_FRAMES, shape="point", year_shape="latest", named_discriminator=False, prefix="latest")
        + _forms_from(_NAMED_LATEST_FRAMES, shape="point", year_shape="latest", named_discriminator=True, prefix="latest-named")
        + _forms_from(_REPORT_FRAMES, shape="report", year_shape="none", named_discriminator=False, prefix="report")
    )
    return tuple(forms)


QUERY_FORMS: tuple[QueryForm, ...] = build_query_forms()


def african_country_names() -> list[str]:
    return list(_AFRICA_COUNTRY_ISO3)


def region_names() -> list[str]:
    names: list[str] = []
    for spec in ZONE_SPEC.values():
        if not spec.get("countries"):
            continue
        aliases = spec.get("aliases") or ()
        if aliases:
            names.append(str(aliases[0]))
    return names


def warehouse_indicator_ids() -> list[str]:
    return [mid for mid, spec in MEASURES.items() if spec.candidate_tables]


def report_indicator_ids() -> list[str]:
    return [mid for mid, spec in MEASURES.items() if not spec.candidate_tables]


def indicator_tables() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for mid, spec in MEASURES.items():
        for table in spec.candidate_tables:
            pairs.append((mid, str(table)))
    return pairs


def named_sample(table_id: str) -> tuple[str, str] | None:
    per_table = value_samples_for_mart_tables({table_id}).get(table_id) or {}
    wanted = metric_discriminator_names()
    for col, vals in per_table.items():
        if str(col).lower() in wanted and vals:
            return str(col), sorted(str(v) for v in vals)[0]
    return None


def compile_form(
    form: QueryForm,
    *,
    table_id: str,
    country: str,
    indicator: str,
    year: int = 2016,
    year_end: int = 2019,
    crop: str = "maize",
    project_id: str = "proj",
    dataset: str = "mart_dev",
) -> str | None:
    """Compile one form against one table. Report forms are not SQL."""
    if form.shape == "report":
        return None
    facets: dict[str, object] = {
        "geography": [country],
        "primary_measures": [indicator],
        "entities": [crop] if "{crop}" in form.template else [],
        "job": "trend" if form.shape == "series" else "fact",
        "pattern": "time_series" if form.shape == "series" else "",
    }
    if form.year_shape == "single_year":
        facets["time_start"] = f"{year}-01-01"
        facets["time_end"] = f"{year}-12-31"
    elif form.year_shape == "span":
        facets["time_start"] = f"{year}-01-01"
        facets["time_end"] = f"{year_end}-12-31"
    named = named_sample(table_id) if form.named_discriminator else None
    query = form.render(
        country=country,
        year=year,
        year_end=year_end,
        indicator=indicator,
        crop=crop,
        discriminator=named[1] if named else "value",
    )
    contract = compile_table_bind_contract(
        table_id,
        facets=facets,
        query=query,
        country_labels=[country],
        project_id=project_id,
        dataset=dataset,
    )
    if named and named[0].lower() not in (contract.required_filters_sql or "").lower():
        literal = named[1].replace("'", "''")
        contract.required_filters_sql = (
            f"{contract.required_filters_sql} AND {named[0]} = '{literal}'"
        ).strip()
    if not str(contract.required_filters_sql or "").strip():
        return None
    if not contract.measure_columns:
        contract.measure_columns = measure_columns_mart(table_id) or measure_columns(table_id)
    if not contract.measure_columns:
        return None
    return compile_sql_from_bind(
        contract,
        project_id=project_id,
        dataset=dataset,
        job=str(facets["job"]),
        pattern=str(facets["pattern"]),
    )


def form_expects_between(form: QueryForm, table_id: str) -> bool:
    return form.shape == "series" and year_column(table_id) is not None


__all__ = [
    "QUERY_FORMS",
    "YEAR_SHAPES",
    "QueryForm",
    "african_country_names",
    "compile_form",
    "form_expects_between",
    "indicator_tables",
    "region_names",
    "report_indicator_ids",
    "warehouse_indicator_ids",
]

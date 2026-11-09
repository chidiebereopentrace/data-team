# Continental OTA Dashboard

OpenTrace Africa's view of what data OFIA holds and what is missing, across every indicator in the catalogue, all 54 African countries, and every crop and livestock or fish value chain grown in each.

**Status: 84,482,473 datapoints and 71,969 knowledge passages.** The letters come from BigQuery. Green means the cell is in the mart, amber means it has only landed, and red means neither side has a value. A datapoint is one warehouse value cell. A knowledge passage is one point in the six Qdrant collections, and a category's passage count includes the points in the collection that belongs to that category.

## Open it

- `dist/Continental_OTA_Dashboard.html` is the dashboard: one self-contained file. Open it in any browser, no server needed, or at https://continental-ota-dashboard-production.up.railway.app. It loads Google Fonts when online and falls back to system fonts offline.
- `dist/OpenTrace_Continental_OTA_Dashboard_Master.xlsx` is the same content as a workbook, with formulas that recalculate when the status letters are replaced with real coverage.

## Rebuild it

Needs Python 3.9+ with `openpyxl`, and Node 18+.

```bash
pip install -r requirements.txt
bash build.sh          # or: npm run build
```

`build.sh` installs the three map packages on first run, then runs five steps:

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `src/parse_indicators.py` | `input/…Sources.xlsx` (Indicators, Additional Indicators tabs) | `build/ind_raw.json` |
| 2 | `src/build_data.py` | the workbook's five country tabs, plus `crops_master.py`, `countries_def.py`, `additions.py` | `build/data.json` |
| 3 | `src/geo.mjs` | `world-atlas` 1:50m country shapes | `build/geo.json` (projected SVG paths for the 54) |
| 4 | `src/build_xlsx.py` | `build/data.json` | `dist/…Master.xlsx` |
| 5 | `src/assemble_html.py` | `src/template.html`, `build/data.json`, `build/geo.json` | `dist/Continental_OTA_Dashboard.html` |

The build is deterministic: the same inputs always give the same output.

## Where things live

| To change | Edit |
|---|---|
| The original indicator list | `input/Africa_Agricultural_Data_Platforms__Sources.xlsx`, then rebuild |
| Indicators added to fill gaps (new categories and additions inside existing ones) | `src/additions.py`: `NEW_CATS`, `EXTRA_IN` |
| Livestock and fisheries value chains, and which countries get which | `src/additions.py`: `LIVESTOCK`, `livestock_for()` |
| The crop universe (names, category, tier 1 to 3, spelling aliases used to match the sheets) | `src/crops_master.py` |
| Countries, AU regions, the 12 priority markets, and crop lists for the 49 generated countries | `src/countries_def.py`: `COUNTRIES`, `PRIORITY`, `T` (templates), `GEN` (per-country adds and drops) |
| Template enrichment shared across regions | `src/additions.py`: `ENRICH` |
| Rules tagging an indicator as crop-level, its unit, frequency and granularity | `src/build_data.py`: `CROP_LEVEL`, `unit_of`, `FREQ`, `GRAN` |
| Dashboard layout, colours, behaviour | `src/template.html` (plain HTML, CSS and JavaScript, no framework) |

Nigeria, Kenya, Ethiopia, Rwanda and Sierra Leone always take their crop lists from the provided tabs (`PROVIDED` in `countries_def.py`). Duplicates within a tab are merged. Each crop keeps the exact name it was listed under.

## Data model (`build/data.json`)

- `cats`: main categories `{code, name, full, source, added}`
- `inds`: sub-indicators `{id, cat, name, tab, origin: Catalogue|Added, level: Country|Crop|Value chain, unit, freq, gran, also}`
- `countries`: `{iso, num, name, region, priority, provenance, nCrops, nLive}`
- `crops`: the crop and value chain universe `{k, name, cat, tier}`
- `ccrops[iso]`: each country's list as `[key, name as listed, group in source tab, category]`
- `datapoints`: deduped warehouse value cells. `passages`: distinct points in the six Qdrant collections
- `covcount[indicatorId]` and `passagecount[indicatorId]`: one integer per country. `cropn[iso]` aligns with `cropcov`
- `cov[indicatorId]`: one status letter per country, in `countries` order
- `covmeta[indicatorId]`: per country `[series start year, number of sources, ACF confidence]`, or zeros for a gap
- `cropInds`: ids of the crop-level and value-chain-level indicators
- `cropcov[iso]`: one string per row of `ccrops[iso]`, one status letter per `cropInds` entry. `-` means not applicable (a crop indicator on a livestock chain, or the reverse)

Status letters: `F` Federated, `P` Partial, `R` Reconstructed, `G` Gap.

## Coverage

`src/build_data.py` fills `cov`, `covmeta`, and `cropcov` from `mart_dev` and the landing datasets through `src/coverage_bq.py`. A cell in the mart is Federated, a cell only in landing is Partial, and a cell in neither is a Gap. Reconstructed is not emitted. The third coverage number is `acf.score_evidence` for a national structural question. Passage counts use ontology targets where a point has them, and otherwise the point's domain label when that label names a catalogue category. A category's passage count also includes every point in the collection that belongs to that category.

## Map

Shapes come from Natural Earth via `world-atlas@2` (public domain), projected with `d3-geo`. Somaliland is merged into Somalia. Western Sahara is drawn neutral, since it is not among the 54. Small island states (Cabo Verde, Comoros, Mauritius, São Tomé and Príncipe, Seychelles) also get a marker dot so they can be clicked.

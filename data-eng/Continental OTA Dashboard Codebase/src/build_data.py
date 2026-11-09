import json, re, openpyxl, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from crops_master import MASTER, BYKEY, C, R, L, O, F, V, S, I, N, FD, SP
from countries_def import COUNTRIES, PRIORITY, PROVIDED, GEN
from additions import NEW_CATS, EXTRA_IN, LIVESTOCK, LV, livestock_for
from coverage_bq import fill_coverage

wb = openpyxl.load_workbook(os.path.join(ROOT, 'input', 'Africa_Agricultural_Data_Platforms__Sources.xlsx'), read_only=True, data_only=True)

# ---------- 1. indicator catalogue ----------
raw = json.load(open(os.path.join(ROOT, 'build', 'ind_raw.json'), encoding='utf-8'))
CAT_META = {  # code, base availability for the illustrative coverage, dominant source family
 "Production & Yield Indicators": ("PROD", .62, "National statistics, FAOSTAT"),
 "Market & Price Indicators": ("MKT", .55, "Market monitoring, trade data"),
 "Input Use Indicators": ("INP", .38, "Household and farm surveys"),
 "Environmental & Climate Indicators": ("CLIM", .80, "Satellite and reanalysis"),
 "Economic & Livelihood Indicators": ("ECON", .34, "Household surveys"),
 "Resilience & Adaptation Indicators": ("RES", .30, "Programme and survey data"),
 "Technology & Innovation Indicators": ("TECH", .32, "Telecom and programme data"),
 "Market Access & Infrastructure Indicators": ("MACC", .40, "Geospatial and infrastructure data"),
 "Nutrition & Food Security Indicators": ("NUT", .60, "DHS, IPC, WFP surveys"),
 "Policy & Institutional Indicators": ("POL", .42, "Policy registers and budgets"),
 "Food Systems & Value Chain Indicators": ("FSVC", .30, "Value chain studies"),
 "Land Use & Soil Health Indicators": ("LAND", .58, "Soil grids, land cover, cadastre"),
 "Gender, Youth & Inclusion Indicators": ("GYI", .24, "Disaggregated surveys"),
 "Humanitarian & Agricultural Emergency Indicators": ("HUM", .64, "Humanitarian assessments"),
 "Investment Readiness & Enterprise Indicators (for value chain actors)": ("INV", .22, "Business registries, investor data"),
}
CROP_LEVEL = re.compile(r"production \(metric|area harvested|yield per hectare|planting and harvest|irrigated vs rainfed|crop yield|harvested area per crop|sown area|"
    r"crop loss|inputs used per hectare|seed type|farmgate|wholesale|export/import|price volatility|warehouse|value chain mapping|fertilizer type|pesticide|"
    r"crop contribution|value addition \(|crop failure|climate risk exposure|climate-resilient crop|land under cultivation per crop|perishability|processed locally|"
    r"food loss|value addition share|taxation on agri|bankable|certified seeds|nutrition-sensitive|traceability|processing capacity", re.I)
FREQ = [(r"daily|weekly|prices", "Weekly"), (r"rainfall|temperature|ndvi|soil moisture|flood|drought", "Monthly"),
        (r"season|calendar|planting|crop loss|ipc|food assistance", "Seasonal")]
GRAN = [(r"by region|region\)|rural roads|distance|ndvi|rainfall|soil|temperature|flood|drought|cold storage|market", "Sub-national")]
def unit_of(s):
    m = re.search(r"\(([^)]*(?:kg|ha|mm|km|%|tons|days|hours|score|index|yes/no)[^)]*)\)", s, re.I)
    if m: return m.group(1)
    if s.startswith('%') or '(%' in s: return "%"
    if re.search(r"index|score", s, re.I): return "Index / score"
    if re.search(r"number of|per 1000", s, re.I): return "Count / rate"
    return "Mixed / categorical"

cats, inds = [], []
for ci, name in enumerate(raw['order']):
    code, base, src = CAT_META[name]
    short = re.sub(r"\s*Indicators.*$", "", name)
    cats.append({"code": code, "name": short, "full": name, "source": src})
    seen = set()
    for k, (s, tab) in enumerate(raw['cats'][name]):
        key = s.lower()
        if key in seen: continue
        seen.add(key)
        iid = f"{code}-{k+1:02d}"
        freq = next((f for p, f in FREQ if re.search(p, s, re.I)), "Annual")
        gran = next((g for p, g in GRAN if re.search(p, s, re.I)), "National")
        lvl = "Value chain" if s.startswith("Livestock numbers") else ("Crop" if CROP_LEVEL.search(s) else "Country")
        inds.append({"id": iid, "cat": code, "name": s, "tab": "Indicators" if tab == "Indicators" else "Additional Indicators",
                     "level": lvl, "unit": unit_of(s), "freq": freq, "gran": gran, "origin": "Catalogue"})
    for k2, (s, lvl) in enumerate(EXTRA_IN.get(code, [])):
        freq = next((f for p, f in FREQ if re.search(p, s, re.I)), "Annual")
        gran = next((g for p, g in GRAN if re.search(p, s, re.I)), "National")
        inds.append({"id": f"{code}-A{k2+1:02d}", "cat": code, "name": s, "tab": "Added", "level": lvl, "unit": unit_of(s),
                     "freq": freq, "gran": gran, "origin": "Added"})
for name, (code, base, src, items) in NEW_CATS.items():
    CAT_META[name] = (code, base, src)
    cats.append({"code": code, "name": re.sub(r"\s*(Indicators|Events)$", "", name) if name.endswith("Indicators") else name, "full": name, "source": src, "added": True})
    for k, (s, lvl) in enumerate(items):
        freq = next((f for p, f in FREQ if re.search(p, s, re.I)), "Annual")
        gran = next((g for p, g in GRAN if re.search(p, s, re.I)), "National")
        if code == "EVT": freq = "Event-based"
        inds.append({"id": f"{code}-{k+1:02d}", "cat": code, "name": s, "tab": "Added", "level": lvl, "unit": unit_of(s),
                     "freq": freq, "gran": gran, "origin": "Added"})
# flag the same indicator appearing in two categories
names = {}
for d in inds: names.setdefault(d['name'].lower(), []).append(d['id'])
for d in inds:
    others = [x for x in names[d['name'].lower()] if x != d['id']]
    d['also'] = others

# ---------- 2. crops ----------
HEAD = re.compile(r"(crops|staples|fruits|vegetables|spices|herbs|condiments|legumes|cereals|grains|oilseeds|nuts|fodder|forage|industrial|tubers|pulses|horticultural)", re.I)
def is_header(s):
    s2 = s.strip()
    if s2.endswith(':'): return True
    if re.match(r"^(Cash Crops|Spices, Herbs|Cash and Export|Oil and Industrial|Forage / Fodder|Vegetables \(Leafy)", s2): return True
    if len(s2) < 60 and HEAD.search(s2) and not re.search(r"\(.*(major|export|for|used|local|grown|ethiopia|kenya)", s2, re.I) and \
       re.match(r"^[A-Z][\w &/,\-()]+$", s2) and (('(' in s2 and ',' in s2) or '/' in s2 or s2.split()[0] in
       ("Cereals","Root","Legumes","Fruits","Vegetables","Spices","Oil","Cash","Forage","Food","Oilseeds","Plantation","Other","Industrial","Horticultural")):
        return True
    return False
HMAP = [(r"cereal|grain", C), (r"root|tuber", R), (r"legume|pulse", L), (r"oil", O), (r"fruit", F), (r"vegetable|leafy", V),
        (r"spice|herb|condiment|medicinal", S), (r"forage|fodder", FD), (r"cash|industrial|plantation|export", I), (r"nut", N),
        (r"specialty|emerging|other|horticultural", SP)]
def hcat(h):
    for p, c in HMAP:
        if re.search(p, h, re.I): return c
    return SP
ALIAS = []
for key, nm, cat, tier, al in MASTER:
    for a in al: ALIAS.append((a.lower(), key))
ALIAS.sort(key=lambda x: -len(x[0]))
def match(name):
    n = name.lower(); best = None
    for a, k in ALIAS:
        m = re.search(r"(^|[^a-z])" + re.escape(a), n)
        if m:
            pos = m.start() + (1 if m.group(1) else 0)
            if best is None or pos < best[0]: best = (pos, k)
    return best[1] if best else None

provided = {}
extra_master = {}
for iso, sheet in PROVIDED.items():
    rows = list(wb[sheet].iter_rows(values_only=True))
    cur, sub = {}, {}
    out, keys = [], set()
    for r in rows:
        for j, c in enumerate(r):
            if c is None or not str(c).strip(): continue
            s = re.sub(r"\s+", " ", str(c)).strip()
            if is_header(s):
                if s.endswith(':'): sub[j] = s[:-1]
                else: cur[j] = s; sub[j] = None
                continue
            h = sub.get(j) or cur.get(j, "Other")
            items = [s]
            m = re.match(r"^(Spices and herbs.*?):\s*(.*)$", s) or re.match(r"^(Oilseeds) \((.*)\)$", s)
            if m: items = [x.strip() for x in m.group(2).split(',')]
            for it in items:
                k = match(it)
                if not k:
                    k = re.sub(r"[^a-z0-9]+", "_", it.split('(')[0].lower()).strip('_')
                    extra_master.setdefault(k, (k, it.split('(')[0].strip(), hcat(h), 3, []))
                if k in keys: continue
                keys.add(k)
                cat = BYKEY[k][2] if k in BYKEY else extra_master[k][2]
                out.append({"k": k, "orig": it, "group": h, "cat": cat})
    provided[iso] = out

allmaster = {m[0]: m for m in MASTER}
allmaster.update(extra_master)
for k, nm, tier in LIVESTOCK: allmaster[k] = (k, nm, LV, tier, [])

# ---------- 3. countries + crop lists ----------
countries, ccrops = [], {}
for name, iso3, num, reg in COUNTRIES:
    if iso3 in provided:
        lst = provided[iso3]; prov = "Provided"
    else:
        lst = [{"k": k, "orig": allmaster[k][1], "group": None, "cat": allmaster[k][2]} for k in dict.fromkeys(GEN[iso3]) if k in allmaster]
        prov = "Generated, to verify"
    lst = lst + [{"k": k, "orig": allmaster[k][1], "group": "Added value chain", "cat": LV} for k in livestock_for(iso3)]
    ccrops[iso3] = lst
    countries.append({"iso": iso3, "num": num, "name": name, "region": reg, "priority": iso3 in PRIORITY, "provenance": prov,
                      "nCrops": sum(1 for x in lst if x['cat'] != LV), "nLive": sum(1 for x in lst if x['cat'] == LV)})

used = {c['k'] for v in ccrops.values() for c in v}
crops = [{"k": k, "name": m[1], "cat": m[2], "tier": m[3]} for k, m in allmaster.items() if k in used]

# ---------- 4. coverage from BigQuery ----------
crop_inds = [d['id'] for d in inds if d['level'] in ("Crop", "Value chain")]
LEVEL = {d['id']: d['level'] for d in inds}
cov, covmeta, cropcov, covcount, cropn, passages, passagecount, datapoints = fill_coverage(
    countries, inds, ccrops, crop_inds, LEVEL, match, LIVESTOCK, cats)

data = {"generated": "2026-11-08", "cats": cats, "inds": inds, "countries": countries, "crops": crops,
        "ccrops": {k: [[x['k'], x['orig'], x['group'] or "", x['cat']] for x in v] for k, v in ccrops.items()},
        "cov": cov, "covmeta": covmeta, "cropInds": crop_inds, "cropcov": cropcov,
        "datapoints": datapoints, "passages": passages, "covcount": covcount, "cropn": cropn,
        "passagecount": passagecount}
json.dump(data, open(os.path.join(ROOT, 'build', 'data.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print("categories", len(cats), "indicators", len(inds), "crop-level", len(crop_inds))
print("countries", len(countries), "master crops used", len(crops), "country-crop rows", sum(len(v) for v in ccrops.values()))
for c in countries: print(f"{c['iso']} {c['nCrops']:4d} {c['provenance']}")
print("extra master from sheets:", sorted(extra_master))
print("added indicators", sum(1 for d in inds if d['origin']=='Added'), "catalogue", sum(1 for d in inds if d['origin']=='Catalogue'))

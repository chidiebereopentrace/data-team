import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter as L
from openpyxl.formatting.rule import CellIsRule
from openpyxl.worksheet.table import Table, TableStyleInfo

d = json.load(open(os.path.join(ROOT, 'build', 'data.json'), encoding='utf-8'))
cats = {c['code']: c for c in d['cats']}
master = {c['k']: c for c in d['crops']}
countries = d['countries']
NC = len(countries)

FONT = "Arial"
H = Font(name=FONT, bold=True, color="FFFFFF", size=10)
HF = PatternFill("solid", fgColor="0B5132")
B = Font(name=FONT, size=10)
BB = Font(name=FONT, size=10, bold=True)
T1 = Font(name=FONT, size=16, bold=True, color="0B5132")
NOTE = Font(name=FONT, size=10, italic=True, color="7A5A00")
FILL = {"F": "B9DDB4", "P": "E3EFC0", "R": "CFE1F5", "G": "F4C7C0", "-": "F2F2F2"}
thin = Side(style="thin", color="E0D7BC")

wb = Workbook()

def header(ws, row, cols, widths=None):
    for j, c in enumerate(cols, 1):
        x = ws.cell(row=row, column=j, value=c); x.font = H; x.fill = HF
        x.alignment = Alignment(wrap_text=True, vertical="center")
    if widths:
        for j, w in enumerate(widths, 1): ws.column_dimensions[L(j)].width = w
    ws.freeze_panes = ws.cell(row=row + 1, column=1)

def status_cf(ws, rng):
    for k, col in FILL.items():
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{k}"'], fill=PatternFill("solid", fgColor=col)))

# ---------- Read me ----------
ws = wb.active; ws.title = "Read me"
ws.column_dimensions["A"].width = 30; ws.column_dimensions["B"].width = 110
ws["A1"] = "OpenTrace Africa | Continental OTA Dashboard: master file"; ws["A1"].font = T1
rows = [
 ("Status", "EXAMPLE. Coverage statuses in this file are illustrative, generated to show how the dashboard works. They are not OFIA holdings. Replace them with real coverage from OFIA as it is assessed."),
 ("Built", "8 Nov 2026, from Africa_Agricultural_Data_Platforms__Sources.xlsx (tabs Indicators, Additional Indicators, and the five country crop tabs)."),
 ("Indicator catalogue", f"{sum(1 for i in d['inds'] if i['origin']=='Catalogue')} sub-indicators from the source workbook, kept word for word, plus {sum(1 for i in d['inds'] if i['origin']=='Added')} added to fill gaps (marked Added). {len(d['cats'])} main categories."),
 ("Added indicators", "New categories: livestock and animal health, fisheries and aquaculture, crop pest and disease, policy programme and trade events, conflict and displacement, agricultural finance and macro, rural energy and digital, demographics and labour, water and irrigation. Plus additions inside market, production, nutrition and climate. Review and confirm."),
 ("Countries", "All 54 UN-recognised African Union member states, by AU region. The 12 priority markets are flagged."),
 ("Crops and value chains", f"{sum(c['nCrops'] for c in countries)} country crop entries across {len([c for c in d['crops'] if c['cat']!='Livestock & fisheries'])} distinct crops, plus livestock and fisheries value chains for every country. Nigeria, Kenya, Ethiopia, Rwanda and Sierra Leone use the provided crop tabs (duplicates within a tab merged). The other 49 lists are built from agro-ecological templates plus country specifics and are marked 'Generated, to verify' for the research team."),
 ("Indicator level", "Country = measured for the country or its regions. Crop = measured per crop (production, prices, inputs, losses). Value chain = measured per livestock or fish value chain."),
 ("Status codes", "F = Federated (in OFIA, good depth)   P = Partial (some years or regions missing)   R = Reconstructed (gap rebuilt from corroborating indicators)   G = Gap (nothing held)   - = not applicable"),
 ("How to update", "Overwrite the status letters in 'Coverage by country' and 'Coverage by commodity'. All counts, percentages and the 'Gap summary' recalculate from them. Do not edit the formula columns at the right of each coverage sheet."),
 ("Sheets", "Indicator catalogue | Countries | Crop master | Country crops | Coverage by country | Coverage by commodity | Gap summary"),
]
for i, (a, b) in enumerate(rows, 3):
    ws.cell(row=i, column=1, value=a).font = BB
    c = ws.cell(row=i, column=2, value=b); c.font = NOTE if a == "Status" else B; c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[i].height = 44
for k, (code, lab) in enumerate([("F", "Federated"), ("P", "Partial"), ("R", "Reconstructed"), ("G", "Gap")], 14):
    ws.cell(row=k, column=1, value=code).fill = PatternFill("solid", fgColor=FILL[code]); ws.cell(row=k, column=1).font = BB
    ws.cell(row=k, column=2, value=lab).font = B

# ---------- Indicator catalogue ----------
ws = wb.create_sheet("Indicator catalogue")
cols = ["ID", "Category code", "Main category", "Sub-indicator", "Origin", "Source tab", "Level", "Unit", "Frequency", "Granularity", "Also listed as", "Typical source family"]
header(ws, 1, cols, [10, 9, 34, 70, 11, 18, 11, 22, 12, 13, 14, 34])
for r, i in enumerate(d['inds'], 2):
    vals = [i['id'], i['cat'], cats[i['cat']]['name'], i['name'], i['origin'], i['tab'], i['level'], i['unit'], i['freq'], i['gran'], ", ".join(i.get('also', [])), cats[i['cat']]['source']]
    for j, v in enumerate(vals, 1):
        c = ws.cell(row=r, column=j, value=v); c.font = B
        if i['origin'] == 'Added' and j == 5: c.font = Font(name=FONT, size=10, bold=True, color="A65E12")
ws.auto_filter.ref = f"A1:{L(len(cols))}{len(d['inds'])+1}"

# ---------- Coverage by country (indicator x country) ----------
wc = wb.create_sheet("Coverage by country")
cols = ["ID", "Category code", "Main category", "Sub-indicator", "Level"] + [c['iso'] for c in countries] + ["Federated", "Partial", "Reconstructed", "Gap", "Covered % (F+P)"]
header(wc, 1, cols, [10, 9, 26, 58, 11] + [5.2] * NC + [10, 9, 13, 7, 12])
for j, c in enumerate(countries, 6):
    wc.cell(row=1, column=j).alignment = Alignment(text_rotation=90, horizontal="center")
first, last = L(6), L(5 + NC)
nI = len(d['inds'])
for r, i in enumerate(d['inds'], 2):
    for j, v in enumerate([i['id'], i['cat'], cats[i['cat']]['name'], i['name'], i['level']], 1):
        wc.cell(row=r, column=j, value=v).font = B
    for j, ch in enumerate(d['cov'][i['id']], 6):
        c = wc.cell(row=r, column=j, value=ch); c.font = B; c.alignment = Alignment(horizontal="center")
    base = 6 + NC
    for k, code in enumerate("FPRG"):
        wc.cell(row=r, column=base + k, value=f'=COUNTIF({first}{r}:{last}{r},"{code}")').font = B
    c = wc.cell(row=r, column=base + 4, value=f"=({L(base)}{r}+{L(base+1)}{r})/{NC}"); c.number_format = "0%"; c.font = BB
status_cf(wc, f"{first}2:{last}{nI+1}")
wc.freeze_panes = "F2"
wc.auto_filter.ref = f"A1:{L(len(cols))}{nI+1}"
COVROWS = nI + 1

# ---------- Country crops (long list) ----------
wk = wb.create_sheet("Country crops")
cols = ["ISO", "Country", "Region", "Commodity key", "Commodity", "Name as listed", "Category", "Group in source tab", "List provenance"]
header(wk, 1, cols, [6, 24, 10, 20, 30, 52, 24, 34, 20])
r = 2
for c in countries:
    for k, orig, grp, cat in d['ccrops'][c['iso']]:
        prov = "Added value chain" if cat == "Livestock & fisheries" else c['provenance']
        for j, v in enumerate([c['iso'], c['name'], c['region'], k, master[k]['name'], orig, cat, grp, prov], 1):
            wk.cell(row=r, column=j, value=v).font = B
        r += 1
KROWS = r - 1
wk.auto_filter.ref = f"A1:I{KROWS}"

# ---------- Coverage by commodity ----------
wm = wb.create_sheet("Coverage by commodity")
ci = d['cropInds']; ind = {i['id']: i for i in d['inds']}
cols = ["ISO", "Country", "Commodity", "Category", "Tier"] + ci + ["Federated", "Partial", "Reconstructed", "Gap", "Applicable", "Covered % (F+P)"]
header(wm, 1, cols, [6, 20, 28, 22, 6] + [5.2] * len(ci) + [10, 9, 13, 7, 11, 12])
for j, iid in enumerate(ci, 6):
    x = wm.cell(row=1, column=j, value=iid); x.alignment = Alignment(text_rotation=90, horizontal="center")
wm.row_dimensions[1].height = 70
f2, l2 = L(6), L(5 + len(ci))
r = 2
for c in countries:
    for (k, orig, grp, cat), row in zip(d['ccrops'][c['iso']], d['cropcov'][c['iso']]):
        for j, v in enumerate([c['iso'], c['name'], master[k]['name'], cat, master[k]['tier']], 1):
            wm.cell(row=r, column=j, value=v).font = B
        for j, ch in enumerate(row, 6):
            wm.cell(row=r, column=j, value=ch)
        base = 6 + len(ci)
        for kk, code in enumerate("FPRG"):
            wm.cell(row=r, column=base + kk, value=f'=COUNTIF({f2}{r}:{l2}{r},"{code}")')
        wm.cell(row=r, column=base + 4, value=f"=SUM({L(base)}{r}:{L(base+3)}{r})")
        x = wm.cell(row=r, column=base + 5, value=f"=IF({L(base+4)}{r}=0,0,({L(base)}{r}+{L(base+1)}{r})/{L(base+4)}{r})"); x.number_format = "0%"
        r += 1
MROWS = r - 1
status_cf(wm, f"{f2}2:{l2}{MROWS}")
wm.freeze_panes = "F2"
wm.auto_filter.ref = f"A1:{L(len(cols))}{MROWS}"
mb = 6 + len(ci)
mF, mP, mApp = L(mb), L(mb + 1), L(mb + 4)

# ---------- Countries ----------
wn = wb.create_sheet("Countries", 2)
cols = ["ISO", "Country", "AU region", "Priority market", "Crop list provenance", "Crops (count)", "Livestock and fish value chains", "Indicators covered % (F+P)", "Indicator gaps (G)", "Commodity cells covered % (F+P)"]
header(wn, 1, cols, [6, 26, 10, 10, 20, 10, 12, 13, 11, 14])
for r, c in enumerate(countries, 2):
    col = L(6 + countries.index(c))
    vals = [c['iso'], c['name'], c['region'], "Yes" if c['priority'] else "", c['provenance'],
            f"=COUNTIFS('Country crops'!$A$2:$A${KROWS},A{r},'Country crops'!$G$2:$G${KROWS},\"<>Livestock & fisheries\")",
            f"=COUNTIFS('Country crops'!$A$2:$A${KROWS},A{r},'Country crops'!$G$2:$G${KROWS},\"Livestock & fisheries\")",
            f"=(COUNTIF('Coverage by country'!{col}$2:{col}${COVROWS},\"F\")+COUNTIF('Coverage by country'!{col}$2:{col}${COVROWS},\"P\"))/{nI}",
            f"=COUNTIF('Coverage by country'!{col}$2:{col}${COVROWS},\"G\")",
            f"=IF(SUMIF('Coverage by commodity'!$A$2:$A${MROWS},A{r},'Coverage by commodity'!${mApp}$2:${mApp}${MROWS})=0,0,"
            f"(SUMIF('Coverage by commodity'!$A$2:$A${MROWS},A{r},'Coverage by commodity'!${mF}$2:${mF}${MROWS})+SUMIF('Coverage by commodity'!$A$2:$A${MROWS},A{r},'Coverage by commodity'!${mP}$2:${mP}${MROWS}))"
            f"/SUMIF('Coverage by commodity'!$A$2:$A${MROWS},A{r},'Coverage by commodity'!${mApp}$2:${mApp}${MROWS}))"]
    for j, v in enumerate(vals, 1):
        x = wn.cell(row=r, column=j, value=v); x.font = B
        if j in (8, 10): x.number_format = "0%"
wn.auto_filter.ref = f"A1:J{NC+1}"

# ---------- Crop master ----------
wr = wb.create_sheet("Crop master", 3)
cols = ["Commodity key", "Commodity", "Category", "Tier (1 major, 3 minor)", "Countries listing it"]
header(wr, 1, cols, [20, 34, 26, 12, 12])
cl = sorted(d['crops'], key=lambda x: (x['cat'], x['name']))
for r, c in enumerate(cl, 2):
    for j, v in enumerate([c['k'], c['name'], c['cat'], c['tier'], f"=COUNTIF('Country crops'!$D$2:$D${KROWS},A{r})"], 1):
        wr.cell(row=r, column=j, value=v).font = B
wr.auto_filter.ref = f"A1:E{len(cl)+1}"

# ---------- Gap summary (category x country) ----------
wg = wb.create_sheet("Gap summary")
wg["A1"] = "Share of sub-indicators held (Federated or Partial), by main category and country. Recalculates from 'Coverage by country'."; wg["A1"].font = NOTE
cols = ["Code", "Main category", "Sub-indicators"] + [c['iso'] for c in countries] + ["Africa"]
header(wg, 2, cols, [7, 34, 10] + [5.2] * NC + [8])
for j in range(4, 4 + NC): wg.cell(row=2, column=j).alignment = Alignment(text_rotation=90, horizontal="center")
for r, cat in enumerate(d['cats'], 3):
    wg.cell(row=r, column=1, value=cat['code']).font = B
    wg.cell(row=r, column=2, value=cat['name']).font = B
    wg.cell(row=r, column=3, value=f"=COUNTIF('Coverage by country'!$B$2:$B${COVROWS},A{r})").font = B
    for j, c in enumerate(countries, 4):
        col = L(6 + j - 4)
        x = wg.cell(row=r, column=j, value=f"=IF($C{r}=0,0,(COUNTIFS('Coverage by country'!$B$2:$B${COVROWS},$A{r},'Coverage by country'!{col}$2:{col}${COVROWS},\"F\")+COUNTIFS('Coverage by country'!$B$2:$B${COVROWS},$A{r},'Coverage by country'!{col}$2:{col}${COVROWS},\"P\"))/$C{r})")
        x.number_format = "0%"; x.font = Font(name=FONT, size=9)
    x = wg.cell(row=r, column=4 + NC, value=f"=AVERAGE({L(4)}{r}:{L(3+NC)}{r})"); x.number_format = "0%"; x.font = BB
from openpyxl.formatting.rule import ColorScaleRule
wg.conditional_formatting.add(f"D3:{L(4+NC)}{len(d['cats'])+2}", ColorScaleRule(start_type="num", start_value=0, start_color="F4C7C0", mid_type="num", mid_value=0.5, mid_color="F6E9B8", end_type="num", end_value=1, end_color="8CC48A"))
wg.freeze_panes = "D3"

wb.save(os.path.join(ROOT, 'dist', 'OpenTrace_Continental_OTA_Dashboard_Master.xlsx'))
print("rows: coverage", COVROWS, "country crops", KROWS, "commodity coverage", MROWS)

"""Step 1. Read the INDICATORS and ADDITIONAL INDICATORS tabs into main categories and sub-indicators.
The tabs lay categories side by side in columns, with a new category header mid-column, so each column tracks its current header."""
import openpyxl, json, re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wb = openpyxl.load_workbook(os.path.join(ROOT, 'input', 'Africa_Agricultural_Data_Platforms__Sources.xlsx'), read_only=True, data_only=True)
HEAD = re.compile(r'Indicators\s*(\(.*\))?$', re.I)
cats, order = {}, []
for tab in ['Indicators', 'Additional Indicators']:
    cur = {}
    for r in wb[tab].iter_rows(values_only=True):
        for j, c in enumerate(r):
            if c is None or not str(c).strip(): continue
            s = re.sub(r'\s+', ' ', str(c)).strip()
            if HEAD.search(s) or s.endswith('Indicators'):
                name = re.sub(r'^\d+\.\s*', '', s); cur[j] = name
                if name not in cats: cats[name] = []; order.append(name)
            elif j in cur:
                cats[cur[j]].append((s, tab))
            else:
                print('Skipped cell with no category header above it:', tab, j, s)
json.dump({'order': order, 'cats': cats}, open(os.path.join(ROOT, 'build', 'ind_raw.json'), 'w', encoding='utf-8'), ensure_ascii=False)
print(len(order), 'categories,', sum(len(v) for v in cats.values()), 'sub-indicators')

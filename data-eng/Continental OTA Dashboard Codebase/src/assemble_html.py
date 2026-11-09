"""Step 4. Inline the data and map geometry into the HTML template, producing one self-contained page."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda *p: open(os.path.join(ROOT, *p), encoding='utf-8').read()
t = rd('src', 'template.html')
d = rd('build', 'data.json').replace('</', '<\\/')
g = rd('build', 'geo.json').replace('</', '<\\/')
out = t.replace('__DATA__', d).replace('__GEO__', g)
open(os.path.join(ROOT, 'dist', 'Continental_OTA_Dashboard.html'), 'w', encoding='utf-8').write(out)
print(f'dist/Continental_OTA_Dashboard.html  {len(out)/1e6:.2f} MB')

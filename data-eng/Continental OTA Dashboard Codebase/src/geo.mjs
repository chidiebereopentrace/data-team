import fs from 'fs';
import path0 from 'path';
import {fileURLToPath} from 'url';
const ROOT=path0.resolve(path0.dirname(fileURLToPath(import.meta.url)),'..');
import {feature} from 'topojson-client';
import {geoPath, geoConicEqualArea, geoMercator} from 'd3-geo';
const topo=JSON.parse(fs.readFileSync(path0.join(ROOT,'node_modules/world-atlas/countries-50m.json')));
const fc=feature(topo,topo.objects.countries);
const data=JSON.parse(fs.readFileSync(path0.join(ROOT,'build/data.json')));
const nums=new Map(data.countries.map(c=>[String(c.num).padStart(3,'0'),c.iso]));
const extra={'732':'ESH'}; // Western Sahara drawn neutral
const somaliland=fc.features.filter(f=>/Somaliland/i.test(f.properties.name));
console.error('somaliland',somaliland.map(f=>f.id+f.properties.name));
const feats=fc.features.filter(f=>nums.has(f.id)||extra[f.id]||/Somaliland/i.test(f.properties.name));
const proj=geoMercator().fitExtent([[10,10],[590,630]],{type:'FeatureCollection',features:feats.filter(f=>nums.has(f.id))});
const path=geoPath(proj);
const out={};
for(const f of feats){
  let iso=nums.get(f.id)||extra[f.id]||'SOM';
  const dd=path(f); if(!dd) continue;
  const r=dd.replace(/(\d+\.\d{1})\d+/g,'$1');
  out[iso]=(out[iso]||'')+r;
}
const missing=data.countries.filter(c=>!out[c.iso]).map(c=>c.iso);
console.error('missing',missing);
// centroids for small islands markers
const cent={};
for(const f of feats){const iso=nums.get(f.id); if(iso){const [x,y]=path.centroid(f); cent[iso]=[+x.toFixed(1),+y.toFixed(1)]; const a=path.area(f); cent[iso].push(+a.toFixed(0));}}
fs.writeFileSync(path0.join(ROOT,'build/geo.json'),JSON.stringify({paths:out,cent}));
console.error('size',JSON.stringify(out).length);

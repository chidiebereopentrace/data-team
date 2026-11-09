# Indicators missing from the current catalogue, and livestock / fisheries value chains.
# Everything here is flagged "Added" in the outputs so the team can review it separately from the original sheet.

# New categories: name -> (code, base availability for illustrative coverage, source family, [(sub-indicator, level)])
NEW_CATS = {
 "Livestock & Animal Health Indicators": ("LIV", .48, "Livestock census, veterinary services", [
   ("Livestock population by species (cattle, goats, sheep, camels, pigs, poultry)", "Value chain"),
   ("Milk production (litres per year, per animal)", "Value chain"),("Meat production by species (metric tons)", "Value chain"),
   ("Egg production (number per year)", "Country"),("Livestock market prices (live animal, per head)", "Value chain"),
   ("Animal disease outbreaks (FMD, PPR, CBPP, Newcastle, ASF)", "Value chain"),("Vaccination coverage by disease (% of herd)", "Value chain"),
   ("Livestock mortality rate (% per year)", "Value chain"),("Veterinary services per 1000 livestock keepers", "Country"),
   ("Rangeland and pasture condition (biomass, NDVI)", "Country"),("Feed and fodder availability and price", "Country"),
   ("Transhumance corridors and pastoral mobility", "Country"),("Hides, skins and leather output", "Value chain")]),
 "Fisheries & Aquaculture Indicators": ("FISH", .44, "FAO FishStat, fisheries departments", [
   ("Capture fisheries landings (marine and inland, metric tons)", "Value chain"),("Aquaculture production by species (metric tons)", "Value chain"),
   ("Fish market prices (landing site, retail)", "Value chain"),("Fish consumption per capita (kg per year)", "Country"),
   ("Registered fishing vessels and fishers", "Country"),("Number of aquaculture farms and hatcheries", "Country"),
   ("Fish post-harvest losses (%)", "Value chain"),("Fish feed availability and price", "Country")]),
 "Crop Pest & Disease Indicators": ("PEST", .36, "Plant protection services, FAO desert locust watch", [
   ("Fall armyworm incidence (% of maize area affected)", "Crop"),("Desert locust presence and swarm alerts", "Country"),
   ("Crop disease outbreaks (cassava mosaic, maize lethal necrosis, banana wilt, wheat rust)", "Crop"),
   ("Pest and disease yield loss (%)", "Crop"),("Aflatoxin and mycotoxin prevalence in grain and nuts", "Crop"),
   ("Pest surveillance coverage (% of districts)", "Country"),("Quarantine pest interceptions at borders", "Country")]),
 "Policy, Programme & Trade Events": ("EVT", .46, "Dated events register", [
   ("Input subsidy programmes (start, stop, scale, crops covered)", "Crop"),("Export bans and export restrictions (dated, by commodity)", "Crop"),
   ("Import tariffs and quotas on food and inputs", "Crop"),("Minimum support prices and price controls", "Crop"),
   ("National programme launches and closures (agriculture, food security)", "Country"),("Currency devaluations and exchange rate regime changes", "Country"),
   ("Fuel price changes and fuel subsidy removal", "Country"),("Trade agreement milestones (AfCFTA, REC protocols)", "Country"),
   ("Seed and fertiliser law changes", "Country")]),
 "Conflict, Security & Displacement Indicators": ("CONF", .55, "Conflict event and displacement trackers", [
   ("Conflict events affecting farming areas (count, by district)", "Country"),("Internally displaced and refugee populations in farming areas", "Country"),
   ("Farmer and herder clashes (count, location)", "Country"),("Farmland abandoned due to insecurity (hectares)", "Country"),
   ("Market closures and trade route disruptions", "Country")]),
 "Agricultural Finance & Macroeconomic Indicators": ("FIN", .58, "Central banks, ministries of finance, IMF, World Bank", [
   ("Agriculture share of GDP (%)", "Country"),("Agricultural credit volume (share of total lending)", "Country"),
   ("Interest rates on agricultural loans (%)", "Country"),("Food inflation (food CPI, year on year)", "Country"),
   ("Exchange rate (official and parallel)", "Country"),("Public agriculture expenditure (% of budget, Malabo 10% target)", "Country"),
   ("Official development assistance to agriculture (USD)", "Country"),("Climate finance flows to agriculture and land use (USD)", "Country"),
   ("Remittances received by rural households", "Country"),("Crop insurance premiums and payouts", "Crop")]),
 "Rural Energy & Digital Infrastructure Indicators": ("ENRG", .54, "Energy and telecom regulators, GSMA", [
   ("Rural electrification rate (% of households)", "Country"),("Mobile network coverage of farming areas (2G, 3G, 4G)", "Country"),
   ("Rural internet penetration (%)", "Country"),("Solar irrigation pumps installed (count)", "Country"),
   ("Diesel and petrol retail price", "Country"),("Mobile money accounts among rural adults (%)", "Country")]),
 "Demographic & Labour Indicators": ("DEM", .66, "Census, labour force surveys", [
   ("Rural population and growth rate", "Country"),("Agriculture share of employment (%)", "Country"),
   ("Number of smallholder farms and households", "Country"),("Average age of farmers", "Country"),
   ("Rural to urban migration rate", "Country"),("Farm household size", "Country")]),
 "Water & Irrigation Indicators": ("WATR", .50, "Water ministries, AQUASTAT, satellite altimetry", [
   ("Irrigation potential vs area equipped for irrigation (hectares)", "Country"),("Area actually irrigated by crop (hectares)", "Crop"),
   ("Groundwater levels and recharge", "Country"),("Reservoir and dam storage levels", "Country"),
   ("Water user associations (count, members)", "Country"),("Agricultural water withdrawal (% of total)", "Country")]),
}

# Additions inside existing categories (code -> list)
EXTRA_IN = {
 "MKT": [("Fertiliser and seed retail prices", "Country"),("Transport cost per tonne-km for produce", "Country"),
         ("Informal cross-border trade flows (volume, by border post)", "Crop"),("Commodity exchange and warehouse receipt trading volumes", "Crop")],
 "PROD": [("Post-harvest losses by crop and stage (%)", "Crop"),("Crop varieties released and in use", "Crop")],
 "NUT": [("Cost of a healthy diet (per person per day)", "Country"),("Biofortified crop adoption (vitamin A cassava, orange maize, iron beans)", "Crop")],
 "CLIM": [("Heat stress days during growing season", "Country"),("Seasonal rainfall forecasts (onset, cessation, totals)", "Country")],
}

# Livestock and fisheries value chains: key, name, tier
LV = "Livestock & fisheries"
LIVESTOCK = [("cattle_beef","Cattle (beef)",1),("dairy","Dairy (cattle milk)",1),("goats","Goats",1),("sheep","Sheep",1),("camels","Camels",2),
             ("pigs","Pigs",2),("poultry_broiler","Poultry (broilers)",1),("poultry_layer","Poultry (eggs)",1),("indigenous_chicken","Indigenous chicken",2),
             ("donkeys","Donkeys",3),("rabbits","Rabbits",3),("honey","Bees and honey",2),("snail","Snails (heliciculture)",3),("grasscutter","Grasscutter",3),
             ("fish_capture_marine","Marine capture fisheries",1),("fish_capture_inland","Inland capture fisheries",1),("tilapia","Aquaculture (tilapia)",1),
             ("catfish","Aquaculture (catfish)",2),("shrimp","Shrimp and prawns",2),("seaweed","Seaweed",3),("ostrich","Ostrich",3)]
COASTAL = set("DZA EGY LBY MRT MAR TUN BEN CPV CIV GMB GHA GIN GNB LBR NGA SEN SLE TGO CMR COG COD GNQ GAB STP COM DJI ERI KEN MDG MUS SYC SOM SDN TZA AGO MOZ NAM ZAF".split())
MUSLIM_MAJ = set("DZA EGY LBY MRT MAR TUN GMB GIN MLI NER SEN SOM SDN DJI COM TCD BFA".split())
ARID = set("DZA EGY LBY MRT MAR TUN MLI NER TCD SDN SOM DJI ERI ETH KEN BWA NAM".split())
ISLANDS = set("CPV COM MUS SYC STP".split())
WEST_CENTRAL = set("BEN CIV GHA GIN GNB LBR NGA SLE TGO CMR COG COD GNQ GAB CAF".split())
def livestock_for(iso):
    ks = ["cattle_beef","dairy","goats","sheep","poultry_broiler","poultry_layer","indigenous_chicken","honey"]
    if iso not in MUSLIM_MAJ: ks.append("pigs")
    if iso in ARID: ks.append("camels")
    if iso not in ISLANDS: ks += ["donkeys","fish_capture_inland"]
    if iso in COASTAL: ks.append("fish_capture_marine")
    if iso not in ("LBY","DJI","SOM","ERI","MRT","CPV","SYC","STP","COM"): ks.append("tilapia")
    if iso in WEST_CENTRAL or iso in ("UGA","KEN","EGY","TZA"): ks.append("catfish")
    if iso in WEST_CENTRAL: ks += ["snail","grasscutter"]
    if iso in ("MDG","MOZ","TZA","NGA","SEN","GAB","EGY","KEN","MUS","SYC"): ks.append("shrimp")
    if iso in ("TZA","MDG","ZAF","KEN","MUS"): ks.append("seaweed")
    if iso in ("ZAF","NAM","BWA"): ks.append("ostrich")
    if iso not in ("SYC","MUS","CPV","DJI"): ks.append("rabbits")
    return ks

# Enrichment of generated crop lists (keys appended where agronomically common)
ENRICH = {
 "sahel": "kersting lablab soybean ginger coriander stylosanthes brachiaria leucaena chili kale_skip spinach beetroot",
 "humid_west": "kersting soybean taro_skip mushroom spinach brachiaria leucaena napier sesame",
 "central": "cowpea sorghum millet tobacco coffee_arabica tea sweet_potato spinach brachiaria leucaena napier stylosanthes cucumber",
 "east_high": "lablab chickpea lentil beetroot cauliflower leek lettuce celery strawberry apple tree_tomato mushroom moringa leucaena sesbania",
 "southern": "lablab sesame beetroot cauliflower lettuce french_bean garden_pea peach apple tangerine litchi baobab moringa leucaena brachiaria",
 "north": "carob fig walnut asparagus apricot sweet_potato saffron",
 "horn": "pigeon_pea lablab sorghum frankincense prickly_pear",
 "islands": "anthurium mushroom lettuce eggplant spinach",
}

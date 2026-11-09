# 54 African Union member states recognised by the UN, AU regions, ISO codes, priority markets,
# and crop composition for the 49 countries without a provided crop sheet.
# Crop lists are agronomic templates plus country specifics. They are a starting list for the research team to verify.

COUNTRIES = [
 # name, iso3, iso numeric, AU region
 ("Algeria","DZA",12,"North"),("Egypt","EGY",818,"North"),("Libya","LBY",434,"North"),("Mauritania","MRT",478,"North"),
 ("Morocco","MAR",504,"North"),("Tunisia","TUN",788,"North"),
 ("Benin","BEN",204,"West"),("Burkina Faso","BFA",854,"West"),("Cabo Verde","CPV",132,"West"),("Côte d'Ivoire","CIV",384,"West"),
 ("Gambia","GMB",270,"West"),("Ghana","GHA",288,"West"),("Guinea","GIN",324,"West"),("Guinea-Bissau","GNB",624,"West"),
 ("Liberia","LBR",430,"West"),("Mali","MLI",466,"West"),("Niger","NER",562,"West"),("Nigeria","NGA",566,"West"),
 ("Senegal","SEN",686,"West"),("Sierra Leone","SLE",694,"West"),("Togo","TGO",768,"West"),
 ("Burundi","BDI",108,"Central"),("Cameroon","CMR",120,"Central"),("Central African Republic","CAF",140,"Central"),
 ("Chad","TCD",148,"Central"),("Congo","COG",178,"Central"),("DR Congo","COD",180,"Central"),
 ("Equatorial Guinea","GNQ",226,"Central"),("Gabon","GAB",266,"Central"),("São Tomé and Príncipe","STP",678,"Central"),
 ("Comoros","COM",174,"East"),("Djibouti","DJI",262,"East"),("Eritrea","ERI",232,"East"),("Ethiopia","ETH",231,"East"),
 ("Kenya","KEN",404,"East"),("Madagascar","MDG",450,"East"),("Mauritius","MUS",480,"East"),("Rwanda","RWA",646,"East"),
 ("Seychelles","SYC",690,"East"),("Somalia","SOM",706,"East"),("South Sudan","SSD",728,"East"),("Sudan","SDN",729,"East"),
 ("Tanzania","TZA",834,"East"),("Uganda","UGA",800,"East"),
 ("Angola","AGO",24,"Southern"),("Botswana","BWA",72,"Southern"),("Eswatini","SWZ",748,"Southern"),("Lesotho","LSO",426,"Southern"),
 ("Malawi","MWI",454,"Southern"),("Mozambique","MOZ",508,"Southern"),("Namibia","NAM",516,"Southern"),
 ("South Africa","ZAF",710,"Southern"),("Zambia","ZMB",894,"Southern"),("Zimbabwe","ZWE",716,"Southern"),
]
PRIORITY = {"NGA","SLE","GHA","SEN","ETH","RWA","KEN","UGA","MOZ","ZAF","ZMB","MAR"}
PROVIDED = {"NGA":"Nigeria","KEN":"Kenya","ETH":"Ethiopia","RWA":"Rwanda","SLE":"Sierra Leone"}

VEG_CORE = "tomato onion chili pepper okra cabbage carrot cucumber eggplant pumpkin garlic lettuce spinach".split()
FRUIT_CORE = "mango orange lemon lime papaya pineapple banana guava watermelon avocado".split()
SPICE_CORE = "ginger basil mint".split()

T = {
 "sahel": "millet pearl_millet sorghum fonio rice maize wheat cowpea groundnut bambara sesame cotton hibiscus_leaf gum_arabic shea baobab tamarind jujube moringa "
          "sweet_potato cassava watermelon melon okra onion tomato chili pepper garden_egg cabbage lettuce carrot cucumber eggplant pumpkin jute_mallow amaranth "
          "mango guava papaya citrus orange lemon lime date_palm sugarcane tobacco henna locust_bean garlic potato mint napier crop_residue melon_seed",
 "humid_west": "cassava yam cocoyam tannia plantain banana rice maize sorghum cowpea groundnut soybean bambara cocoa coffee_robusta oil_palm rubber cashew kola coconut "
          "shea cotton sugarcane pineapple mango orange tangerine lemon lime papaya avocado guava soursop star_apple bush_mango african_pear breadfruit "
          "garden_egg okra tomato chili pepper onion amaranth fluted_pumpkin waterleaf bitter_leaf jute_mallow cassava_leaves cucumber cabbage lettuce carrot "
          "ginger turmeric black_pepper alligator_pepper grains_paradise locust_bean basil lemongrass sweet_potato melon_seed watermelon moringa raffia "
          "guinea_grass stylosanthes",
 "central": "cassava plantain banana maize rice groundnut common_bean cocoyam yam sweet_potato oil_palm cocoa coffee_robusta rubber sugarcane african_pear "
          "cassava_leaves amaranth gnetum okra tomato chili onion eggplant pumpkin pineapple papaya avocado mango orange lemon guava soursop bush_mango kola "
          "coconut sesame soybean garden_egg jute_mallow ginger black_pepper raffia melon_seed napier",
 "east_high": "maize common_bean sorghum finger_millet wheat barley potato sweet_potato cassava banana matooke coffee_arabica tea pyrethrum sugarcane cotton "
          "sunflower groundnut soybean sesame pigeon_pea cowpea mung_bean field_pea avocado passion_fruit tree_tomato mango papaya pineapple orange lemon "
          "watermelon kale cabbage tomato onion carrot spinach amaranth nightshade spider_plant french_bean garden_pea eggplant chili pepper pumpkin cucumber "
          "garlic ginger coriander napier desmodium rhodes_grass brachiaria cut_flowers macadamia eucalyptus",
 "horn": "sorghum maize sesame cowpea mung_bean groundnut banana citrus lemon lime papaya mango watermelon guava date_palm onion tomato chili pepper okra "
          "cabbage pumpkin frankincense myrrh cotton sugarcane cassava rice wheat millet jujube tamarind sudan_grass",
 "southern": "maize sorghum pearl_millet finger_millet wheat barley groundnut soybean common_bean cowpea bambara pigeon_pea sweet_potato cassava potato tobacco "
          "cotton sugarcane sunflower macadamia citrus orange lemon grapefruit avocado mango banana papaya guava paprika chili cabbage kale tomato onion "
          "pumpkin carrot spinach okra garlic watermelon marula baobab monkey_orange mung_bean pepper lucerne_placeholder rhodes_grass",
 "north": "wheat durum_wheat barley maize oats sorghum chickpea lentil faba_bean field_pea olive date_palm orange tangerine lemon grape almond fig pomegranate "
          "apricot peach apple pear plum cherry quince loquat prickly_pear carob melon watermelon strawberry tomato potato onion garlic pepper chili artichoke "
          "zucchini carrot cabbage cauliflower lettuce eggplant beetroot radish sugar_beet sunflower cumin coriander mint anise rosemary fenugreek black_cumin "
          "alfalfa berseem henna",
 "islands": "rice cassava maize sweet_potato yam cocoyam banana plantain coconut breadfruit jackfruit litchi mango papaya pineapple avocado orange lemon "
          "passion_fruit sugarcane vanilla clove cinnamon black_pepper ylang ginger turmeric tomato chili cabbage onion carrot pumpkin cucumber lettuce garlic",
}
T = {k: [x for x in v.split() if x != "lucerne_placeholder"] + (["alfalfa"] if k == "southern" else []) for k, v in T.items()}

from additions import ENRICH
for _k, _v in ENRICH.items():
    for _x in _v.split():
        if _x not in T[_k]: T[_k].append(_x)

def mk(templates, add="", drop=""):
    s = []
    for t in templates:
        for k in T[t]:
            if k not in s: s.append(k)
    for k in add.split():
        if k not in s: s.append(k)
    d = set(drop.split())
    return [k for k in s if k not in d]

GEN = {
 "DZA": mk(["north"], "potato sweet_potato pistachio walnut saffron tobacco", "rice sorghum"),
 "EGY": mk(["north"], "rice cotton sugarcane mango banana guava soybean groundnut sesame jute_mallow okra sweet_potato onion strawberry", "argan"),
 "LBY": mk(["north"], "", "rice cherry plum quince artichoke sugar_beet"),
 "MRT": mk(["sahel"], "date_palm", "cocoa fonio shea cotton tobacco cassava cucumber"),
 "MAR": mk(["north"], "argan saffron blueberry raspberry avocado banana rice citrus sugarcane", ""),
 "TUN": mk(["north"], "pistachio", "sorghum rice"),
 "BEN": mk(["humid_west","sahel"], "kersting", "date_palm henna wheat rubber"),
 "BFA": mk(["sahel"], "cashew soybean yam", "date_palm wheat"),
 "CPV": ["maize","congo_bean","common_bean","cowpea","sweet_potato","cassava","potato","sugarcane","banana","papaya","mango","coffee_arabica","grape","coconut",
         "tomato","onion","cabbage","carrot","pepper","chili","pumpkin","lettuce","garlic","cucumber","guava","orange","lemon","avocado","peanut_placeholder",
         "groundnut","lablab","sorghum","watermelon","melon","tamarind","pigeon_pea","sweet_potato","mint"],
 "CIV": mk(["humid_west"], "coffee_arabica cotton sorghum millet sesame", ""),
 "GMB": mk(["sahel","humid_west"], "", "cocoa rubber coffee_robusta wheat date_palm kola black_pepper"),
 "GHA": mk(["humid_west","sahel"], "sesame", "date_palm henna wheat"),
 "GIN": mk(["humid_west"], "fonio millet coffee_arabica", "rubber"),
 "GNB": mk(["humid_west"], "millet fonio", "cocoa rubber coffee_robusta"),
 "LBR": mk(["humid_west"], "", "shea cotton sorghum"),
 "MLI": mk(["sahel"], "shea yam", ""),
 "NER": mk(["sahel"], "", "fonio cassava rice_placeholder"),
 "SEN": mk(["sahel","humid_west"], "", "cocoa rubber coffee_robusta kola tannia"),
 "TGO": mk(["humid_west","sahel"], "coffee_arabica", "rubber date_palm wheat henna"),
 "BDI": mk(["east_high"], "oil_palm cocoyam yam rice", "pyrethrum cut_flowers sesame"),
 "CMR": mk(["humid_west","central","east_high"], "millet sesame", "pyrethrum cut_flowers"),
 "CAF": mk(["central"], "sorghum millet cotton coffee_arabica tobacco shea", "rubber"),
 "TCD": mk(["sahel"], "yam", "date_palm_placeholder"),
 "COG": mk(["central"], "", "sesame soybean"),
 "COD": mk(["central","east_high"], "cinchona millet", "cut_flowers macadamia"),
 "GNQ": mk(["central"], "", "rice soybean sesame groundnut_placeholder"),
 "GAB": mk(["central"], "", "rice sesame soybean"),
 "STP": ["cocoa","coffee_robusta","coffee_arabica","coconut","oil_palm","banana","plantain","breadfruit","jackfruit","cassava","cocoyam","yam","sweet_potato","maize",
         "black_pepper","vanilla","cinnamon","papaya","pineapple","mango","avocado","orange","lemon","soursop","tomato","cabbage","chili","onion","pumpkin","cucumber",
         "lettuce","carrot","ginger","sugarcane","kola"],
 "COM": mk(["islands"], "", "rice"),
 "DJI": ["sorghum","maize","date_palm","tomato","onion","pepper","chili","watermelon","melon","lemon","mango","papaya","banana","cabbage","carrot","okra","eggplant",
         "lettuce","alfalfa","sudan_grass","cowpea","pumpkin","guava"],
 "ERI": mk(["horn"], "teff barley wheat finger_millet chickpea lentil faba_bean linseed niger_seed potato cactus_placeholder prickly_pear", "frankincense_placeholder"),
 "MDG": mk(["islands"], "cocoa coffee_robusta coffee_arabica groundnut common_bean bambara lima_bean cotton sisal tobacco potato soybean raffia", ""),
 "MUS": ["sugarcane","tea","potato","tomato","onion","cabbage","carrot","cucumber","pumpkin","chili","pepper","garlic","ginger","lettuce","eggplant","banana",
         "pineapple","litchi","papaya","mango","coconut","anthurium","cut_flowers","maize","groundnut","sweet_potato","cassava","orange","lemon","avocado","bean_placeholder",
         "french_bean","cauliflower","beetroot","vanilla","garden_pea"],
 "SYC": ["coconut","cinnamon","vanilla","banana","breadfruit","cassava","sweet_potato","papaya","mango","pineapple","jackfruit","avocado","orange","lemon","tomato",
         "cabbage","chili","cucumber","pumpkin","eggplant","lettuce","black_pepper","ginger","tea","sugarcane","coffee_robusta"],
 "SOM": mk(["horn"], "", "wheat"),
 "SSD": mk(["sahel","central"], "shea", "cocoa rubber coffee_robusta date_palm wheat fonio gnetum"),
 "SDN": mk(["sahel"], "sunflower date_palm", "fonio cassava shea"),
 "TZA": mk(["east_high","southern"], "cashew clove coconut sisal tobacco cocoa vanilla rice", "paprika"),
 "UGA": mk(["east_high"], "vanilla cocoa coffee_robusta oil_palm shea rice", ""),
 "AGO": mk(["southern","central"], "coffee_robusta coffee_arabica sisal", "rooibos"),
 "BWA": ["sorghum","maize","pearl_millet","cowpea","groundnut","bambara","sunflower","morama","melon","watermelon","pumpkin","cabbage","tomato","onion","potato","carrot",
         "spinach","kale","pepper","chili","beetroot","citrus","orange","grape","marula","devils_claw","monkey_orange","lablab","mung_bean","sweet_potato","cotton","alfalfa",
         "rhodes_grass","lettuce","garlic","butternut_placeholder"],
 "SWZ": mk(["southern"], "", "macadamia_placeholder paprika"),
 "LSO": ["maize","sorghum","wheat","barley","oats","common_bean","field_pea","potato","cabbage","kale","spinach","carrot","onion","tomato","beetroot","pumpkin",
         "apple","peach","pear","apricot","plum","quince","garlic","lettuce","alfalfa","sunflower","asparagus"],
 "MWI": mk(["southern"], "tea coffee_arabica macadamia rice pyrethrum", ""),
 "MOZ": mk(["southern"], "cashew coconut sesame rice litchi", ""),
 "NAM": ["pearl_millet","maize","sorghum","wheat","groundnut","cowpea","bambara","sunflower","grape","date_palm","citrus","orange","melon","watermelon","pumpkin","potato",
         "cabbage","tomato","onion","carrot","pepper","chili","marula","devils_claw","monkey_orange","cotton","lucerne_placeholder","alfalfa","mango","banana","sweet_potato","spinach"],
 "ZAF": mk(["southern","north"], "rooibos honeybush canola pecan rose_geranium blueberry raspberry asparagus", "date_palm henna argan carob durum_wheat"),
 "ZMB": mk(["southern"], "rice coffee_arabica", "rooibos"),
 "ZWE": mk(["southern"], "tea coffee_arabica", ""),
}

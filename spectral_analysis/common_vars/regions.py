ids_Cal1 = [762, 787]			# test:0 California -- 23 to 51 N
ids_Cal2 = [809, 831, 852, 868] # .. (cont) Para las longitudes menores a 128
ids_Cal = ids_Cal1+ids_Cal2
ids_Can = [709, 730, 750, 771] # Canarias -- 16 to 36 N
ids_Peru = [450, 572, 596] #616,636  Peru Chile -- 5 to 45 S
ids_Ben = [533, 578, 602] #459,556, Benguela -- 15 to 37 S ** Quitamos la **459** y ponemos la 602
ids_Kuro = [733, 751, 796]

all_ids = ids_Cal+ids_Can+ids_Peru+ids_Ben

ids_regions = {
    "California": list(reversed(ids_Cal)),
    "Canarias": list(reversed(ids_Can)),
    "Peru": list(reversed(ids_Peru)),
    "Benguela": list(reversed(ids_Ben)),
    "Kuroshio": list(reversed(ids_Kuro))
}

timezone_regions = {
    "Canarias": 0,
    "California": -8,
    "Peru": -5,
    "Benguela": 2
}

tzinfo_regions = {
    "Canarias": "Atlantic/Azores",
    "California": "US/Pacific",
    "Peru": "America/Lima",
    "Benguela": "" ## UTC (default)
}

#ids_regions = {
#    "California": [762],
#    "Canarias": [750, 730],
#    "Peru": [450, 572],
#    "Benguela": [533]
#}

faces_regions_all = {
    1: ids_Ben,
    2: ids_Can,
    7: ids_Kuro+ids_Cal2,
    10: ids_Cal1,
    11: ids_Peru
}
faces_regions = faces_regions_all

#faces_regions = {
#    1: [533],
#    2: [750,730],
#    7: [],
#    10: [762],
#    11: [450,572]
#}


face4id = {v:k for k,l in faces_regions.items() for v in l}
face4id[0] = 7 ## Todo: Delete this when I write better tests

lats4id = {
    762: 26.641, 787: 31.462, 809: 36.056, 831: 40.411, 852: 44.521, 868: 48.383,
    709: 16.398, 730: 21.611, 750: 26.641, 771: 31.462,
    450: -40.411, 572: -21.611, 596: -16.398, 616: -11.032, 636: -5.552,
    459: -36.056, 533: -26.641, 556: -21.611, 578: -16.398, 602: -11.032,
    733: 21.612, 751: 26.642, 796: 36.055
}

lons4id = {
    762: -125, 787: -125, 809: -131, 831: -131, 852: -131, 868: -137,
    709: -29, 730: -23, 750: -23, 771: -23,
    450: -83, 572: -77, 596: -83, 616: -83, 636: -89,
    459: 13, 533: 7, 556: 1, 578: 7, 602: 7,
    733: 151, 751: 151, 796: 151
}

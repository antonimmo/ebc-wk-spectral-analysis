ids_Cal1 = [762, 787]			# test:0 California -- 23 to 51 N
ids_Cal2 = [809, 831, 852, 868] # .. (cont) Para las longitudes menores a 128
ids_Cal = ids_Cal1+ids_Cal2
ids_Can = [709, 730, 750, 771] # Canarias -- 16 to 36 N
ids_Peru = [450, 572, 596] #616,636  Peru Chile -- 5 to 45 S
ids_Ben = [533, 578, 602] #459,556, Benguela -- 15 to 37 S ** Quitamos la **459** y ponemos la 602
ids_Kuro = [733, 751, 796]

ids_regions = {
    "California": list(reversed(ids_Cal)),
    "Canarias": list(reversed(ids_Can)),
    "Peru": list(reversed(ids_Peru)),
    "Benguela": list(reversed(ids_Ben)),
    "Kuroshio": list(reversed(ids_Kuro))
}

faces_regions = {
	1: ids_Ben,
	2: ids_Can,
	7: ids_Kuro+ids_Cal2,
	10: ids_Cal1,
	11: ids_Peru
}

lats4id = {
    762: 26.641, 787: 31.462, 809: 36.056, 831: 40.411, 852: 44.521, 868: 48.383,
    709: 16.398, 730: 21.611, 750: 26.641, 771: 31.462,
    450: -40.411, 572: -21.611, 596: -16.398, 616: -11.032, 636: -5.552,
    459: -36.056, 533: -26.641, 556: -21.611, 578: -16.398, 602: -11.032,
    733: 21.612, 751: 26.642, 796: 36.055
}

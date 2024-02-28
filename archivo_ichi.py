
import pandas as pd
wind_ava = pd.read_csv('wind_ava.csv.gz', compression="gzip")
columnas_seleccionadas = wind_ava[["energy", "t2m.13", "u10.13", "v10.13", "u100.13", "v100.13", "cape.13", "flsr.13", "fsr.13", "iews.13", "inss.13",
                                   "lai_hv.13","lai_lv.13","u10n.13","v10n.13","stl1.13","stl2.13","stl3.13","stl4.13","sp.13","p54.162.13","p59.162.13","p55.162.13"]]

# Imprimir el resultado
print(columnas_seleccionadas)
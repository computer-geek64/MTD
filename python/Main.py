# Main.py
# Ashish D'Souza
# November 15th, 2018

import os
import Data as data


def criteria(observation, dataset):
    return data.median(dataset) - 1.5 * data.iqr(dataset) <= observation <= data.median(dataset) + 1.5 * data.iqr(dataset)


soda = data.SODA("data.delaware.gov", "2bb6-s69t")
print(soda.get_columns())
where_query = soda.format_where_query(["countycode=\"3\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"", "NOT stt_datastatuscodetext=\"InVld\"", "mot_monitorname=\"Ozone\"", "sta_stationname=\"Martin Luther King\""])
results = soda.download(where=where_query, select="paramvalue")
results = [float(x["paramvalue"]) for x in results]
print("Dataset length: " + str(len(results)))
print(results)

dataset = data.remove_outliers([results], criteria)
print(dataset)
print(len(dataset))

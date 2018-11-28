# Main.py
# Ashish D'Souza
# November 15th, 2018

import os
import Data
import OutlierDetection


soda = Data.SODA("data.delaware.gov", "2bb6-s69t")
print(soda.get_columns())
where_query = soda.format_where_query(["countycode=\"3\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"", "NOT stt_datastatuscodetext=\"InVld\"", "sta_stationname=\"Martin Luther King\""])
results = soda.download(where=where_query, order="date_time DESC", limit=1945)
print("Dataset length: " + str(len(results)))

data_dict = {}
for observation in range(len(results)):
    if results[observation]["date_time"] not in data_dict.keys():
        data_dict[results[observation]["date_time"]] = {}
    data_dict[results[observation]["date_time"]][results[observation]["mot_monitorname"]] = float(results[observation]["paramvalue"])

data = []
for date_time in sorted(data_dict.keys()):
    data.append([])
    for monitor_name in sorted(data_dict[date_time].keys()):
        data[-1].append(data_dict[date_time][monitor_name])
if len(data[0]) != len(data[1]):
    data.pop(0)

outliers = OutlierDetection.knn_outlier_detection(data)
print(len(data))
print(outliers)

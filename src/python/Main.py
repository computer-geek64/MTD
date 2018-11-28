# Main.py
# Ashish D'Souza
# November 15th, 2018

import os
import Data
import OutlierDetection


soda = Data.SODA("data.delaware.gov", "2bb6-s69t")
print(soda.get_columns())
where_query = soda.format_where_query(["countycode=\"3\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"", "NOT stt_datastatuscodetext=\"InVld\"", "sta_stationname=\"Martin Luther King\""])
results = soda.download(where=where_query, order="date_time DESC", limit=120000)
print("Dataset length: " + str(len(results)))

data_dict = {}
for observation in range(len(results)):
    if results[observation]["date_time"] not in data_dict.keys():
        data_dict[results[observation]["date_time"]] = {}
    data_dict[results[observation]["date_time"]][results[observation]["mot_monitorname"]] = float(results[observation]["paramvalue"])
default_length = max(map(len, data_dict.values()))
original_length = len(data_dict)

data = []
for date_time in sorted(data_dict.keys()):
    data.append([])
    for monitor_name in sorted(data_dict[date_time].keys()):
        data[-1].append(data_dict[date_time][monitor_name])
    if len(data_dict[date_time]) != default_length:
        data.pop(-1)
        data_dict.pop(date_time)

if len(max(data)) != len(min(data)):
    print("Length of data matrix is inconsistent")
    exit(0)

print("Dataset size: " + str(original_length) + " --> " + str(len(data)))

k = int(round(len(data) * 0.001))
outliers = OutlierDetection.knn_outlier_detection(data, k=k)
print("k: " + str(k))
print(outliers)
print(len(outliers))
print(str(round(len(outliers) / len(data) * 100, 2)) + "%")
print(list(data_dict.keys())[0])
print(list(data_dict.keys())[-1])

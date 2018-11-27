# Main.py
# Ashish D'Souza
# November 15th, 2018

import os
import Data
import OutlierDetection


soda = Data.SODA("data.delaware.gov", "2bb6-s69t")
print(soda.get_columns())
where_query = soda.format_where_query(["countycode=\"3\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"", "NOT stt_datastatuscodetext=\"InVld\"", "sta_stationname=\"Martin Luther King\""])
results = soda.download(where=where_query, order="date_time DESC", limit=1000)
print("Dataset length: " + str(len(results)))
print(results)
monitor_names = []
for observation in range(len(results)):
    if results[observation]["mot_monitorname"] not in monitor_names:
        monitor_names.append(results[observation]["mot_monitorname"])
variables = [results[x]["mot_monitorname"] for x in range(len(results))]
values = [results[x]["paramvalue"] for x in range(len(results))]
data = [[results[observation]] for observation in range(len(results))]
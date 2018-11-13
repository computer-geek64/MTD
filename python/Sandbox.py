# Sandbox.py
# Ashish D'Souza
# November 13th, 2018

import os
import pandas as pd
from sodapy import Socrata
from Data import SODA


data = SODA("data.delaware.gov", "2bb6-s69t", environ="soda_token")
results = data.get_columns()
print(results)
print(type(results))
client = Socrata("data.delaware.gov", os.environ["soda_token"])
#results = client.get("2bb6-s69t", where="date_time=\"2015-10-28T20:00:00.000\" AND countycode=3 AND sta_stationname=\"Martin Luther King\"", limit=1)
results = client.get("2bb6-s69t", query="select countycode, date_time where countycode=3 order by date_time")
# results_df = pd.DataFrame.from_records(results)
# print(type(results_df))
# print(results_df)
print(results)
print(type(results))
print(len(results))
[print(x) for x in results]
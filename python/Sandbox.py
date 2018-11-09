# Sandbox.py
# Ashish D'Souza
# November 8th, 2018

import pandas as pd
from sodapy import Socrata
import os


client = Socrata("data.delaware.gov", os.environ["soda_token"])
results = client.get("2bb6-s69t", where="date_time=\"2015-10-28T20:00:00.000\" AND countycode=3 AND sta_stationname=\"Martin Luther King\"")
results_df = pd.DataFrame.from_records(results)
print(type(results))
print(type(results_df))
print(results)
print(results_df)
print(len(results))
print(results[0])
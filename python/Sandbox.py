# Sandbox.py
# Ashish D'Souza
# November 8th, 2018

import os
import pandas as pd
from sodapy import Socrata
from Data import Data


Data(website="data.delaware.gov", environ="soda_token")
exit(0)
client = Socrata("data.delaware.gov", os.environ["soda_token"])
results = client.get("2bb6-s69t", where="date_time=\"2015-10-28T20:00:00.000\" AND countycode=3 AND sta_stationname=\"Martin Luther King\"")
# results_df = pd.DataFrame.from_records(results)
# print(type(results_df))
# print(results_df)
print(results)
print(type(results))
print(len(results))
[print(x) for x in results if True]
# Main.py
# Ashish D'Souza
# November 13th, 2018

import os
from Data import SODA


data = SODA("data.delaware.gov", "2bb6-s69t")
print(data.get_columns())
where_query = data.format_where_query(["countycode=\"1\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"", "NOT stt_datastatuscodetext=\"InVld\""])
print(where_query)
results = data.download(select="stt_datastatuscodetext", where=where_query)
print(len(results))
print(results)
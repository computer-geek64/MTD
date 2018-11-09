# Data.py
# Ashish D'Souza
# November 9th, 2018

import os
import numpy as np
from sodapy import Socrata


class SODA:
    def __init__(self, domain: str, **kwargs: dict) -> None:
        soda_token = ""
        if "soda_token" in kwargs.keys():
            soda_token = kwargs["soda_token"]
        elif "soda_token" in os.environ:
            soda_token = os.environ["soda_token"]
        else:
            print("SODA authentication token not specified")
            exit(0)
        self.client = Socrata(domain, soda_token)

    def format_query(self, queries: list, **kwargs: dict) -> str:
        query = " AND ".join(queries)
        for key, value in kwargs.items():
            query += " AND " + key + "=\"" + value + "\""
        return query

    def download(self, dataset_identifier: str, query: str) -> list:
        return self.client.get(dataset_identifier, where=query)


class Format:

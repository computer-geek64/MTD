# Data.py
# Ashish D'Souza
# November 13th, 2018

import os
import numpy as np
from sodapy import Socrata


class SODA:
    def __init__(self, domain: str, dataset_identifier: str, **kwargs: dict) -> None:
        soda_token = ""
        if "soda_token" in kwargs.keys():
            soda_token = kwargs["soda_token"]
        elif "soda_token" in os.environ:
            soda_token = os.environ["soda_token"]
        else:
            print("SODA authentication token not specified")
            exit(0)
        self.client = Socrata(domain, soda_token)
        self.dataset_identifier = dataset_identifier
        self.where = ""

    def get_columns(self):
        return list(self.client.get(self.dataset_identifier, limit=1)[0].keys())

    def format_where_query(self, where_queries: list, **kwargs: dict) -> str:
        self.where = " AND ".join(where_queries)
        for key, value in kwargs.items():
            self.where += " AND " + key + "=\"" + value + "\""
        return self.where

    def download(self, **kwargs) -> list:
        return self.client.get(self.dataset_identifier, **kwargs)

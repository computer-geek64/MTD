# Data.py
# Ashish D'Souza
# November 9th, 2018

import os
import numpy as np
from sodapy import Socrata
import sodapy


class Data:
    def __init__(self, domain, **kwargs):
        soda_token = ""
        if "soda_token" in kwargs.keys():
            soda_token = kwargs["soda_token"]
        elif "soda_token" in os.environ:
            soda_token = os.environ["soda_token"]
        else:
            print("SODA authentication token not specified")
            exit(0)
        self.client = Socrata(domain, soda_token)

# Data.py
# Ashish D'Souza
# November 9th, 2018

import os
import numpy as np
from sodapy import Socrata


class Data:
    def __init__(self):
        if "soda_token" not in os.environ:
            print("SODA authentication token not set in \"soda_token\" environment variable")
            exit(0)


if "soda_token" not in os.environ:
    print("SODA authentication token not set in \"soda_token\" environment variable")
    exit(0)
client = Socrata("data.delaware.gov", os.environ["soda_token"])
client.get("")
#!/usr/bin/python3
# Main.py
# Ashish D'Souza
# @computer-geek64
# December 21st, 2018

import Data
import OutlierDetection
import numpy as np
from datetime import datetime


class DownloadData:
    def __init__(self, parameters):
        self.parameters = parameters
        self.soda = Data.SODA("data.delaware.gov", "2bb6-s69t")

    def download_training_data(self):
        where_query = self.soda.format_where_query(
            ["countycode=\"3\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"",
             "NOT stt_datastatuscodetext=\"InVld\"", "sta_stationname=\"Martin Luther King\"",
             "date_time<\"2018-01-01\"", "date_time>\"2016-12-31\""])
        results = self.soda.download(where=where_query, order="date_time ASC", limit=1000000)
        return results

    def download_testing_data(self):
        where_query = self.soda.format_where_query(
            ["countycode=\"3\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"",
             "NOT stt_datastatuscodetext=\"InVld\"", "sta_stationname=\"Martin Luther King\"",
             "date_time<\"2019-01-01\"", "date_time>\"2017-12-31\""])
        results = self.soda.download(where=where_query, order="date_time ASC", limit=1000000)
        return results

    def format_data(self, results):
        data_dict = {}
        for observation in range(len(results)):
            if results[observation]["date_time"] not in data_dict.keys():
                data_dict[results[observation]["date_time"]] = {}
            data_dict[results[observation]["date_time"]][results[observation]["mot_monitorname"]] = float(results[observation]["paramvalue"])
        all_parameters = []
        for observation in range(len(results)):
            if results[observation]["mot_monitorname"] not in all_parameters:
                all_parameters.append(results[observation]["mot_monitorname"])
        all_parameters = sorted(all_parameters)
        original_length = len(data_dict)

        data = []
        for date_time in sorted(data_dict.keys()):
            if len([self.parameters[i] for i in range(len(self.parameters)) if self.parameters[i] in list(data_dict[date_time].keys())]) == len(self.parameters):
                data.append([data_dict[date_time][monitor_name] for monitor_name in self.parameters])
                date = datetime.strptime(date_time.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                data[-1].insert(-1, date.hour)
                data[-1].insert(-1, date.month)
            else:
                data_dict.pop(date_time)

        if len(max(data)) != len(min(data)):
            print("ERROR: Length of data matrix is inconsistent")
            exit(0)

        return data

    def remove_outliers(self, data, k=3):
        knn_logdir = "./tensorboard/knn_outlier_detection/" + str(int(datetime.now().timestamp())) + "/" + str(k)
        outliers = OutlierDetection.knn_outlier_detection(data, k, knn_logdir)

        corrected_data = [data[i] for i in range(len(data)) if i not in outliers]
        y_data = np.array(corrected_data)[:, -1:]
        x_data = np.array(corrected_data)[:, :-1]
        return [x_data, y_data]

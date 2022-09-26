#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling
@File    ：Port_env.py
@Author  ：JacQ
@Date    ：2022/1/4 16:17
"""

from utils.log import Logger

logger = Logger(name='root').get_logger()


class PortEnv:
    def __init__(self, quay_cranes, buffers, lock_stations, crossovers, yard_blocks, yard_cranes):
        # 注意machines是dict的形式储存
        self.quay_cranes = quay_cranes
        self.buffers = buffers
        self.lock_stations = lock_stations
        self.crossovers = crossovers
        self.yard_blocks = yard_blocks
        self.yard_cranes = yard_cranes
        self.mission_list = []

        self.station_to_crossover_matrix = [[0 for i in range(len(self.crossovers))] for j in
                                            range(len(self.lock_stations))]
        self.exit_to_station_matrix = [0 for i in range(len(self.lock_stations))]
        self.station_to_crossover_min = [0 for i in range(len(self.crossovers))]
        self.exit_to_station_average = 0

    def cal_finish_time(self):
        max_makespan = 0
        for yard_crane in self.yard_cranes.values():
            if yard_crane.process_time:
                if yard_crane.process_time[-1][-1] > max_makespan:
                    max_makespan = yard_crane.process_time[-1][-1]
        return max_makespan

    def cal_station_makespan(self):
        max_makespan = 0
        for lock_station in self.lock_stations.values():
            if any(lock_station.process_time):
                if lock_station.process_time[-1][-1] > max_makespan:
                    max_makespan = lock_station.process_time[-1][-1]
        return max_makespan

    def get_station_assign_list(self):
        assign_list = []
        for lock_station in self.lock_stations.values():
            station_assign_list = []
            for mission in lock_station.mission_list:
                station_assign_list.append(mission.idx[1:])
                print(mission.idx[1:], end='\t')
            assign_list.append(assign_list)
            print()


if __name__ == "main":
    pass

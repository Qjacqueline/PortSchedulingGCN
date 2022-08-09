#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling
@File    ：dispatching_rules.py
@Author  ：JacQ
@Date    ：2022/1/5 12:26
"""


class sort_missions(object):
    # 按照上一阶段完成时间进行优先级排序
    @staticmethod
    def FCFS(missions):
        missions.sort(key=lambda x: x.total_process_time + x.release_time)  # 按照抵达exit时间进行排序

    @staticmethod
    def CHAR_ORDER(missions):
        missions.sort(key=lambda x: int(x.idx[1:]))  # 按照名称排序

    @staticmethod
    def RELEASE_ORDER(missions):
        missions.sort(key=lambda x: (x.release_time, int(x.idx[1:])))  # 按照释放时间排序

    @staticmethod
    def A_CROSSOVER_NB(missions):
        missions.sort(key=lambda x: (x.machine_start_time[5], int(x.idx[1:])))  # 按照到达交叉口时间排序

    @staticmethod
    def A_CROSSOVER_UA(missions, distance_v):
        missions.sort(key=lambda x: (x.total_process_time + x.release_time + distance_v[
            int(x.machine_list[4][-1]) - 1] / x.vehicle_speed))  # 按照到达交叉口时间排序,未分配之前

    @staticmethod
    def A_YARD_NB(missions):
        missions.sort(key=lambda x: (x.machine_start_time[7], int(x.idx[1:])))  # 按照到达场桥时间排序

    @staticmethod
    def A_YARD_UA(missions, crossover_loc):
        missions.sort(
            key=lambda x: (x.total_process_time + x.release_time + (abs(x.yard_stop_loc[0] - crossover_loc[0]) + abs(
                x.yard_stop_loc[1] - crossover_loc[1])) / x.vehicle_speed))  # 按照到达交叉口时间排序,未分配之前

    @staticmethod
    def A_STATION(missions, station):
        missions.sort(
            key=lambda x: (
                x.machine_start_time[1] + station.distance_to_exit / x.vehicle_speed, int(x.idx[1:])))  # 按照到达锁站时间排序

    @staticmethod
    def A_EXIT(missions):
        missions.sort(key=lambda x: (x.machine_start_time[1]))  # 按照到达锁站时间排序

    @staticmethod
    def A_STATION_NB(missions):
        missions.sort(
            key=lambda x: (
                x.machine_start_time[1], int(x.idx[1:])))  # 按照到达锁站时间排序


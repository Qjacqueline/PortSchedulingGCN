#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling
@File    ：mission_cal_methods.py
@Author  ：JacQ
@Date    ：2022/1/5 16:13
"""

import random
from typing import List

import conf.configs as Cf
from common.mission import Mission

random.seed(Cf.RANDOM_SEED)


# 寻找等待时间最长的mission
def find_longest_mission_for_machine(machine, stage=2):
    longest_wait_time = 0
    longest_mission = None
    for mission in machine.mission_list:
        wait_time = mission.machine_process_time[stage]
        if wait_time > longest_wait_time:
            longest_wait_time = wait_time
            longest_mission = mission
    if not longest_mission:
        longest_mission = find_random_one_mission_one_machine(machine)
    return longest_mission


# 随机选择一个机器上的任务
def find_random_one_mission_one_machine(machine):
    mission = random.choice(machine.mission_list)
    return mission


# 随机选择一个机器上的两个任务
def find_random_adjacent_two_mission_one_machine(machine):
    missionA = random.choice(machine.mission_list)
    if machine.mission_list.index(missionA) + 1 == len(machine.mission_list):
        missionB = machine.mission_list[machine.mission_list.index(missionA) - 1]
    else:
        missionB = machine.mission_list[machine.mission_list.index(missionA) + 1]
    return missionA, missionB


# 选择机器上最长wait的任务和他的相邻任务
def find_longest_adjacent_two_mission_one_machine(machine, stage=2):
    missionA = find_longest_mission_for_machine(machine, stage)
    return missionA, find_adjacent_mission_for_mission(missionA, machine)


def find_nearest_mission_for_mission_in_other_machines(target_mission, machine_a, machines):
    temp_machines = list(machines.values()).copy()
    del temp_machines[int(machine_a.idx[-1]) - 1]
    target_mission_arrive_time = target_mission.machine_start_time[2]
    target_mission_start_time = target_mission.machine_start_time[4]
    min_time = float('inf')
    min_time_mission = None
    min_station = None
    for cur_machine in temp_machines:
        for mission in cur_machine.mission_list:
            mission_arrive_time = (mission.machine_start_time[
                                       1] + machine_a.distance_to_exit / mission.vehicle_speed)  # mission_b能够到达a的时间
            if target_mission_arrive_time < mission_arrive_time:
                min_time_mission = mission
                min_station = cur_machine
            if target_mission_arrive_time < mission_arrive_time < target_mission_start_time:  # 找和他差不多时间但是比他晚到的任务，不然完成时间又提前了
                if mission.station_process_time < min_time:
                    min_time = mission.station_process_time
                    min_time_mission = mission
                    min_station = cur_machine
    if not min_time_mission:
        min_time_mission = temp_machines[-1].mission_list[-1]
        min_station = temp_machines[-1]
    return min_time_mission, min_station


# 寻找与当前任务的上阶段结束时间最近的任务
def find_nearest_mission_for_mission_in_other_machine(target_mission, machine_a, machine_b):
    target_mission_arrive_time = target_mission.machine_start_time[2]
    target_mission_start_time = target_mission.machine_start_time[4]
    min_time = float('inf')
    min_time_mission = None
    flag = True
    for mission in machine_b.mission_list:
        mission_arrive_time = (mission.machine_start_time[
                                   1] + machine_a.distance_to_exit / mission.vehicle_speed)  # mission_b能够到达a的时间
        if target_mission_arrive_time < mission_arrive_time and flag:
            min_time_mission = mission
            flag = False
        if target_mission_arrive_time < mission_arrive_time < target_mission_start_time:  # 找和他差不多时间但是比他晚到的任务，不然完成时间又提前了
            if mission.station_process_time < min_time:
                min_time = mission.station_process_time
                min_time_mission = mission
    if not min_time_mission:
        min_time_mission = machine_b.mission_list[-1]
    return min_time_mission


# def find_nearest_mission(target_mission, machine, stage=2):
#     target_mission_time = target_mission.machine_start_time[stage - 1]
#     min_time_gap = float('inf')
#     min_time_mission = None
#     for mission in machine.mission_list:
#         time_gap = abs(target_mission_time - mission.machine_start_time[stage - 1])
#         if time_gap < min_time_gap:
#             min_time_gap = time_gap
#             min_time_mission = mission
#     if not min_time_mission:
#         min_time_mission = random_choice_one_mission_one_machine(machine)
#     return min_time_mission

# 选择机器上的相邻任务
def find_adjacent_mission_for_mission(mission_a, machine):
    if machine.mission_list.index(mission_a) + 1 == len(machine.mission_list):
        missionB = machine.mission_list[machine.mission_list.index(mission_a) - 1]
    else:
        missionB = machine.mission_list[machine.mission_list.index(mission_a) + 1]
    return missionB


def find_least_wait_delta_position_for_mission(target_mission, machine):
    target_arrive_time = target_mission.machine_start_time[1] + machine.distance_to_exit / target_mission.vehicle_speed
    min_time_gap = float('inf')
    i = 0  # 后面insert时+1了
    min_time_pos = 0
    if not machine.mission_list:
        return 0
    for time in machine.whole_occupy_time:
        time_gap = target_arrive_time - time[2]
        if 0 < time_gap < min_time_gap:
            min_time_gap = time_gap
            min_time_pos = i
        i = i + 1
    return min_time_pos


def find_position_for_machine_to_relocate_inner_machine(mission_a, machine):
    target_arrive_time = mission_a.machine_start_time[2]
    begin = 0
    end = len(machine.mission_list)
    index = 0
    for cur_mission in machine.mission_list:
        if cur_mission.machine_start_time[2] < target_arrive_time < cur_mission.machine_start_time[4]:
            if begin == 0:
                begin = index
            end = index
        index += 1
    return random.randint(begin, end)


def derive_mission_attribute_list(iter_mission_list: List[Mission]):
    mission_attribute_list = {}
    for cur_mission in iter_mission_list:
        mission_attribute_list[cur_mission.idx] = [int(cur_mission.idx[1:]), int(cur_mission.quay_crane_id[-1]),
                                                   cur_mission.vehicle_speed, cur_mission.yard_crane_process_time,
                                                   cur_mission.station_process_time, cur_mission.yard_stop_loc[0],
                                                   cur_mission.yard_stop_loc[1]]
    return mission_attribute_list


def derive_mission_cross_yard_index_for_RNN(mission, yard_cranes_set):
    # index从0开始
    crossover_idx = mission.yard_block_loc[0]
    if crossover_idx[0] == 'A':
        c_idx = 0
    if crossover_idx[0] == 'B':
        c_idx = 1
    if crossover_idx[0] == 'C':
        c_idx = 2
    y_idx = yard_cranes_set.index(crossover_idx)
    return c_idx, y_idx

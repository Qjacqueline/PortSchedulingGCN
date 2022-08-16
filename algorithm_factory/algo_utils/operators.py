#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New
@File    ：operators.py
@Author  ：JacQ
@Date    ：2022/4/22 9:31
"""
from algorithm_factory.algo_utils import machine_cal_methods, mission_cal_methods
from algorithm_factory.algo_utils.missions_sort_rules import sort_missions
from common.port_env import PortEnv

from utils.log import Logger

logger = Logger().get_logger()


def inter_relocate_random_station_random_mission_to_random_machine(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    s_a = machine_cal_methods.find_random_machine(iter_env.lock_stations)
    if not s_a.mission_list:
        return 'end'
    s_b = machine_cal_methods.find_random_machine(iter_env.lock_stations, s_a)
    if s_a == s_b:
        return 'not end'
    m_a = mission_cal_methods.find_random_one_mission_one_machine(s_a)
    pos = mission_cal_methods.find_least_wait_delta_position_for_mission(m_a, s_b)
    logger.debug("执行relocate_operator3算子:" + s_a.idx + "-" + m_a.idx + '->' + s_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    missions_sort_rule = 'FCFS'
    machine_cal_methods.process_relocate_operator(m_a, pos, s_a, s_b, buffer_flag)
    getattr(sort_missions, missions_sort_rule)(iter_env.mission_list)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    getattr(sort_missions, missions_sort_rule)(iter_env.mission_list)
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


def inter_relocate_random_machine_longest_mission_to_random_machine(iter_env: PortEnv, buffer_flag: bool = True):
    s_a = machine_cal_methods.find_random_machine(iter_env.lock_stations)
    if not s_a.mission_list:
        return 'end'
    s_b = machine_cal_methods.find_random_machine(iter_env.lock_stations, s_a)
    if s_a == s_b:
        return 'not end'
    m_a = mission_cal_methods.find_longest_mission_for_machine(s_a)
    pos = mission_cal_methods.find_least_wait_delta_position_for_mission(m_a, s_b)
    logger.debug("执行relocate_operator2算子:" + s_a.idx + "-" + m_a.idx + '->' + s_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    missions_sort_rule = 'FCFS'
    machine_cal_methods.process_relocate_operator(m_a, pos, s_a, s_b, buffer_flag)
    getattr(sort_missions, missions_sort_rule)(iter_env.mission_list)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    getattr(sort_missions, missions_sort_rule)(iter_env.mission_list)
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


def inter_relocate_latest_station_random_mission_to_random_machine(iter_env: PortEnv, buffer_flag: bool = True):
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)
    s_b = machine_cal_methods.find_random_machine(iter_env.lock_stations, s_a)
    if s_a == s_b:
        return 'not end'
    m_a = mission_cal_methods.find_random_one_mission_one_machine(s_a)
    pos = mission_cal_methods.find_least_wait_delta_position_for_mission(m_a, s_b)
    logger.debug("执行station inter relocate算子:" + s_a.idx + "-" + m_a.idx + '->' + s_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    machine_cal_methods.process_relocate_operator(m_a, pos, s_a, s_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


def inter_relocate_latest_station_random_mission_to_earliest_station(iter_env: PortEnv, buffer_flag: bool = True):
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)
    s_b = machine_cal_methods.find_earliest_machine(iter_env.lock_stations)
    if s_a == s_b:
        return 'not end'
    m_a = mission_cal_methods.find_random_one_mission_one_machine(s_a)
    pos = mission_cal_methods.find_least_wait_delta_position_for_mission(m_a, s_b)
    logger.debug("执行relocate_operator4算子:" + s_a.idx + "-" + m_a.idx + '->' + s_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    machine_cal_methods.process_relocate_operator(m_a, pos, s_a, s_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


def inter_relocate_latest_station_longest_mission_to_earliest_machine(iter_env: PortEnv, buffer_flag: bool = True):
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)
    s_b = machine_cal_methods.find_earliest_machine(iter_env.lock_stations)
    if s_a == s_b:
        return 'not end'
    m_a = mission_cal_methods.find_longest_mission_for_machine(s_a)
    pos = mission_cal_methods.find_least_wait_delta_position_for_mission(m_a, s_b)
    logger.debug("执行relocate_operator1算子:" + s_a.idx + "-" + m_a.idx + '->' + s_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    machine_cal_methods.process_relocate_operator(m_a, pos, s_a, s_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


def inter_relocate_latest_yard_crane_random_mission(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    y_a = machine_cal_methods.find_latest_machine(iter_env.yard_cranes)
    m_a = mission_cal_methods.find_random_one_mission_one_machine(y_a)
    s_a = iter_env.lock_stations[m_a.machine_list[4]]
    s_b = machine_cal_methods.find_random_machine(iter_env.lock_stations, s_a)
    if s_a == s_b:
        return 'not end'
    pos = mission_cal_methods.find_least_wait_delta_position_for_mission(m_a, s_b)
    logger.debug("执行yard inter relocate算子:" + "station: " + s_a.idx + " " + m_a.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    machine_cal_methods.process_relocate_operator(m_a, pos, s_a, s_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


# check
def inner_swap_latest_station_longest_mission(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)  # 选择要进行操作的机器
    m_a, m_b = mission_cal_methods.find_longest_adjacent_two_mission_one_machine(
        s_a)  # 选择待交换的两个任务
    logger.debug("执行inner swap算子:" + "station: " + s_a.idx + " " + m_a.idx + "<->" + m_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.process_inner_swap(s_a, m_a, m_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


# check todo 选哪个mission可优化
def inner_swap_latest_yard_crane_random_mission(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    y_a = machine_cal_methods.find_latest_machine(iter_env.yard_cranes)
    m_a = mission_cal_methods.find_random_one_mission_one_machine(y_a)
    s_a = iter_env.lock_stations(m_a.machine_list[4])
    m_b = mission_cal_methods.find_adjacent_mission_for_mission(m_a, s_a)  # 选择待交换的两个任务
    logger.debug("执行yard inner swap算子:" + "station: " + s_a.idx + " " + m_a.idx + "<->" + m_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.process_inner_swap(s_a, m_a, m_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


# check
def inner_relocate_latest_station_random_mission(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)  # 选择要进行操作的机器
    m_a = mission_cal_methods.find_random_one_mission_one_machine(s_a)
    pos = mission_cal_methods.find_position_for_machine_to_relocate_inner_machine(m_a, s_a)
    logger.debug("执行station inner relocate算子:" + "station: " + s_a.idx + " " + m_a.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.process_inner_relocate(s_a, m_a, pos, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


# check
def inner_relocate_latest_station_longest_mission(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)  # 选择要进行操作的机器
    # iter_env.machine_chosen_num[int(curr_machine.idx[1:]) - 1] = iter_env.machine_chosen_num[
    #                                                              int(curr_machine.idx[1:]) - 1] + 1
    m_a = mission_cal_methods.find_longest_mission_for_machine(s_a)
    pos = mission_cal_methods.find_position_for_machine_to_relocate_inner_machine(m_a, s_a)
    logger.debug("执行inner relocate算子:" + "station: " + s_a.idx + " " + m_a.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.process_inner_relocate(s_a, m_a, pos, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


# check
def inner_relocate_latest_yard_crane_random_mission(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    y_a = machine_cal_methods.find_latest_machine(iter_env.yard_cranes)
    m_a = mission_cal_methods.find_random_one_mission_one_machine(y_a)
    s_a = iter_env.lock_stations[m_a.machine_list[4]]
    pos = mission_cal_methods.find_position_for_machine_to_relocate_inner_machine(m_a, s_a)
    # iter_env.machine_chosen_num[int(s_a.idx[1:]) - 1] = iter_env.machine_chosen_num[
    #                                                     int(s_a.idx[1:]) - 1] + 1
    logger.debug("执行yard inner relocate算子:" + "station: " + s_a.idx + " " + m_a.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.process_inner_relocate(s_a, m_a, pos, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


# check
def inter_swap_latest_station_longest_mission_earliest_machine(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)
    s_b = machine_cal_methods.find_earliest_machine(iter_env.lock_stations)
    if s_a == s_b:
        return 'not end'
    m_a = mission_cal_methods.find_longest_mission_for_machine(s_a)
    m_b = mission_cal_methods.find_nearest_mission_for_mission_in_other_machine(m_a, s_a, s_b)
    logger.debug("执行inter swap算子:" + s_a.idx + "-" + m_a.idx + "<->" + s_b.idx + "-" + m_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    machine_cal_methods.process_inter_swap(s_a, s_b, m_a, m_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


# check
def inter_swap_latest_station_longest_mission_all_machines(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)
    m_a = mission_cal_methods.find_longest_mission_for_machine(s_a)
    m_b, s_b = mission_cal_methods.find_nearest_mission_for_mission_in_other_machines(m_a, s_a, iter_env.lock_stations)
    if s_a == s_b:
        return 'not end'
    logger.debug("执行inter swap算子:" + s_a.idx + "-" + m_a.idx + "<->" + s_b.idx + "-" + m_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    machine_cal_methods.process_inter_swap(s_a, s_b, m_a, m_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'


# check
def inter_swap_latest_station_random_mission_earliest_machine(iter_env: PortEnv, buffer_flag: bool = True) -> str:
    s_a = machine_cal_methods.find_latest_machine(iter_env.lock_stations)
    s_b = machine_cal_methods.find_earliest_machine(iter_env.lock_stations)
    if s_a == s_b:
        return 'not end'
    m_a = mission_cal_methods.find_longest_mission_for_machine(s_a)
    m_b = mission_cal_methods.find_nearest_mission_for_mission_in_other_machine(m_a, s_a, s_b)
    logger.debug("执行inter swap算子:" + s_a.idx + "-" + m_a.idx + "<->" + s_b.idx + "-" + m_b.idx)
    machine_cal_methods.del_station_afterwards(iter_env, buffer_flag)  # 删除station之后的工序
    machine_cal_methods.del_machine(s_a, buffer_flag)
    machine_cal_methods.del_machine(s_b, buffer_flag)
    machine_cal_methods.process_inter_swap(s_a, s_b, m_a, m_b, buffer_flag)
    machine_cal_methods.crossover_process_by_order(iter_env, buffer_flag)  # 阶段五：交叉口
    machine_cal_methods.yard_crane_process_by_order(iter_env, buffer_flag)  # 阶段六：场桥
    return 'not end'

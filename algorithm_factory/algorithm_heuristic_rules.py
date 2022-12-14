#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：port_scheduling
@File    ：dispatching_rules.py
@Author  ：JacQ
@Date    ：2021/12/21 15:20
"""
import random
import time
from copy import deepcopy

import numpy as np
import torch

from algorithm_factory.algo_utils.machine_cal_methods import quay_crane_process_by_order, buffer_process_by_order, \
    exit_process_by_order, crossover_process_by_order, yard_crane_process_by_order, \
    station_process_by_random, station_process_by_least_mission_num, station_process_by_least_distance, \
    station_process_by_least_wait, output_solution, get_rnn_state_v2, station_process_by_fixed_order
from algorithm_factory.algo_utils.missions_sort_rules import sort_missions
from common import LockStation
from common.iter_solution import IterSolution
from data_process.input_process import read_input
from utils.log import Logger

logger = Logger().get_logger()


def Random_Choice(port_env, buffer_flag=True):
    T1 = time.time()
    solution = deepcopy(port_env)
    """
         阶段四：锁站策略     按照去往箱区进行划分，随机选
         """
    # logger.info("开始执行算法 Random_Choice.")
    quay_crane_process_by_order(solution)  # 阶段一：岸桥
    buffer_process_by_order(solution)  # 阶段二：缓冲区
    exit_process_by_order(solution)  # 阶段三：抵达岸桥exit
    station_process_by_random(solution, buffer_flag)  # 阶段四：锁站
    crossover_process_by_order(solution, buffer_flag)  # 阶段五：交叉口
    yard_crane_process_by_order(solution, buffer_flag)  # 阶段六：场桥
    # 更新任务attributes
    for mission in solution.mission_list:
        mission.cal_mission_attributes(buffer_flag)
    solution.last_step_makespan = solution.cal_finish_time()
    print("makespan为:" + str(solution.last_step_makespan) + " time:" + str(time.time() - T1))
    return solution.last_step_makespan, solution, output_solution(solution)


def Fixed_order(port_env, order, buffer_flag=True):
    T1 = time.time()
    solution = deepcopy(port_env)
    """
         阶段四：锁站策略     按照去往箱区进行划分，随机选
         """
    # logger.info("开始执行算法 Random_Choice.")
    quay_crane_process_by_order(solution)  # 阶段一：岸桥
    buffer_process_by_order(solution)  # 阶段二：缓冲区
    exit_process_by_order(solution)  # 阶段三：抵达岸桥exit
    station_process_by_fixed_order(solution, order, buffer_flag)  # 阶段四：锁站
    crossover_process_by_order(solution, buffer_flag)  # 阶段五：交叉口
    yard_crane_process_by_order(solution, buffer_flag)  # 阶段六：场桥
    # 更新任务attributes
    for mission in solution.mission_list:
        mission.cal_mission_attributes(buffer_flag)
    solution.last_step_makespan = solution.cal_finish_time()
    print("makespan为:" + str(solution.last_step_makespan) + " time:" + str(time.time() - T1))
    return solution.last_step_makespan, solution, output_solution(solution)


def Random_Choice_By_Mission(solu: IterSolution, s_num: int, m_max_num: int, mission_num: int,
                             buffer_flag: bool = True):
    done = 0
    pre_makespan = 0
    T1 = time.time()
    state = get_rnn_state_v2(solu.iter_env, 0, m_max_num)
    for step in range(mission_num):
        if step == 28:
            done = 1
        cur_mission = solu.iter_env.mission_list[step]
        action = random.randint(0, s_num - 1)
        makespan = solu.step_v2(action, cur_mission, step)
        if step == mission_num - 1:
            new_state = state
        else:
            new_state = get_rnn_state_v2(solu.iter_env, step + 1, m_max_num)
        reward = (pre_makespan - makespan) / 100  # exp(-makespan / 10000)
        pre_makespan = makespan
        state = new_state
        step += 1
    print("makespan为:" + str(solu.last_step_makespan) + " time:" + str(time.time() - T1))


def Least_Wait_Time_Choice(port_env, buffer_flag=True):
    T1 = time.time()
    solution = deepcopy(port_env)
    missions_sort_rule = 'FCFS'
    """
         阶段四：锁站策略   选等待时间更少的锁站
       :return:
        """
    # logger.info("开始执行算法Least_Wait_Time_Choice.")
    quay_crane_process_by_order(solution)  # 阶段一：岸桥
    buffer_process_by_order(solution)  # 阶段二：缓冲区
    exit_process_by_order(solution)  # 阶段三：抵达岸桥exit
    getattr(sort_missions, missions_sort_rule)(solution.mission_list)
    station_process_by_least_wait(solution, buffer_flag)  # 阶段四：锁站
    crossover_process_by_order(solution, buffer_flag)  # 阶段五：交叉口
    yard_crane_process_by_order(solution, buffer_flag)  # 阶段六：场桥
    # 更新任务attributes
    for mission in solution.mission_list:
        mission.cal_mission_attributes(buffer_flag)
    solution.last_step_makespan = solution.cal_finish_time()
    print("makespan为:" + str(solution.last_step_makespan) + " time:" + str(time.time() - T1))
    return solution.last_step_makespan, solution, output_solution(solution)


def Least_Mission_Num_Choice(port_env, buffer_flag=True):
    T1 = time.time()
    solution = deepcopy(port_env)
    """
         阶段四：锁站策略   选现有任务最少的锁站
       :return:
        """
    # logger.info("开始执行算法Least_Mission_Num_Choice.")
    quay_crane_process_by_order(solution)  # 阶段一：岸桥
    buffer_process_by_order(solution)  # 阶段二：缓冲区
    exit_process_by_order(solution)  # 阶段三：抵达岸桥exit
    # getattr(sort_missions, missions_sort_rule)(solution.mission_list)
    station_process_by_least_mission_num(solution, buffer_flag)  # 阶段四：锁站
    crossover_process_by_order(solution, buffer_flag)  # 阶段五：交叉口
    yard_crane_process_by_order(solution, buffer_flag)  # 阶段六：场桥
    # 更新任务attributes
    for mission in solution.mission_list:
        mission.cal_mission_attributes(buffer_flag)
    solution.last_step_makespan = solution.cal_finish_time()
    print("makespan为:" + str(solution.last_step_makespan) + " time:" + str(time.time() - T1))
    return solution.last_step_makespan, solution, output_solution(solution)


def Least_Mission_Num_Choice_By_Mission(solu: IterSolution, m_max_num: int, mission_num: int):
    T1 = time.time()
    state = get_rnn_state_v2(solu.iter_env, 0, m_max_num)
    for step in range(mission_num):
        cur_mission = solu.iter_env.mission_list[step]
        min_mission_num = 10000
        min_station: LockStation = None
        for station in solu.iter_env.lock_stations.values():
            if any(station.mission_list):
                cur_mission_num = len(station.mission_list)
            else:
                cur_mission_num = 0
            if cur_mission_num < min_mission_num:
                min_mission_num = cur_mission_num
                min_station = station
        action = int(min_station.idx[-1]) - 1
        makespan = solu.step_v2(action, cur_mission, step)
        if step == mission_num - 1:
            new_state = state
        else:
            new_state = get_rnn_state_v2(solu.iter_env, step + 1, m_max_num)
        state = new_state
    print("makespan为:" + str(solu.last_step_makespan) + " time:" + str(time.time() - T1))


def Least_Distance_Choice(port_env, buffer_flag=True):
    T1 = time.time()
    solution = deepcopy(port_env)
    """
         阶段四：锁站策略   选现有任务最少的锁站
       :return:
        """
    # logger.info("开始执行算法Least_Distance_Choice.")
    quay_crane_process_by_order(solution)  # 阶段一：岸桥
    buffer_process_by_order(solution)  # 阶段二：缓冲区
    exit_process_by_order(solution)  # 阶段三：抵达岸桥exit
    station_process_by_least_distance(solution, buffer_flag)  # 阶段四：锁站
    crossover_process_by_order(solution, buffer_flag)  # 阶段五：交叉口
    yard_crane_process_by_order(solution, buffer_flag)  # 阶段六：场桥
    # 更新任务attributes
    for mission in solution.mission_list:
        mission.cal_mission_attributes(buffer_flag)
    solution.last_step_makespan = solution.cal_finish_time()
    print("makespan为:" + str(solution.last_step_makespan) + " time:" + str(time.time() - T1))
    return solution.last_step_makespan, solution, output_solution(solution)


if __name__ == '__main__':
    # seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # env = read_input()
    # _, solution, _ = Least_Mission_Num_Choice(env.init_env)
    instance = read_input('train', 0, 'A')
    # mini_makespan = Least_Wait_Time_Choice(instance)
    # print("Least_Wait_Time:"+str(mini_makespan))
    # mini_makespan = Least_Mission_Num_Choice(instance)
    # print("Least_Mission_Num:" + str(mini_makespan))
    # mini_makespan = Least_Distance_Choice(instance)
    # print(" Least_Distance:" + str(mini_makespan))

    # instance.l2a_init()
    # # Least_Mission_Num_Choice_By_Mission(solu=instance, m_max_num=100, mission_num=300)
    # Random_Choice_By_Mission(solu=instance, s_num=4, m_max_num=100, mission_num=300)
    # calculate_statistics(instance.iter_env, 'v1')
    # a = 1

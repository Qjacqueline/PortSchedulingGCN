#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：port_scheduling
@File    ：Scheduling.py
@Author  ：JacQ
@Date    ：2021/12/21 14:21
"""
import random

import numpy as np

import conf.configs as Cf
from algorithm_factory.algorithm_SA import SA
from algorithm_factory.algorithm_heuristic_rules import Random_Choice, Least_Wait_Time_Choice, Least_Mission_Num_Choice, \
    Least_Distance_Choice, Fixed_order
from data_process.input_process import read_input

if __name__ == '__main__':
    random.seed(Cf.RANDOM_SEED)
    np.random.seed(Cf.RANDOM_SEED)
    # get heuristic result
    # env_names = []  # ['train_1_', 'train_2_', 'train_3_', 'train_4_', 'train_5_', 'train_6_', 'train_7_', 'train_8_',
    # 'train_9_', 'train_10_', 'train_11_', 'train_12_', 'train_13_', 'train_14_', 'train_15_', 'train_16_',
    # 'train_17_', 'train_18_', 'train_19_', 'train_0_']
    # for i in range(0, 1):
    #     env_names.append('train_' + str(i) + '_')
    env_names = ['train_0_']
    print(Cf.dataset + " " + str(Cf.MISSION_NUM_ONE_QUAY_CRANE))
    print("Fixed order")
    env = read_input('train', 0, 'A')
    makespan, _, _ = Fixed_order(env.init_env, [2, 2, 0, 3, 0, 0, 1, 1, 0, 2, 1, 3, 2, 1, 1])  # [1, 1, 0, 3, 1, 2]
    # [3, 2, 0, 3, 0, 2, 1, 2, 0, 3, 1, 3, 2, 1, 1, 0]
    print("Random_Choice")
    total_makespan = 0
    for env_name in env_names:
        env = read_input('train', 0, 'A')
        makespan, _, _ = Random_Choice(env.init_env)
        total_makespan += makespan
    print("total_makespan:" + str(total_makespan))

    total_makespan = 0
    print("Least_Wait_Time_Choice")
    for env_name in env_names:
        env = read_input('train', 0, 'A')
        makespan, _, _ = Least_Wait_Time_Choice(env.init_env)
        total_makespan += makespan
    print("total_makespan:" + str(total_makespan))
    total_makespan = 0
    print("Least_Mission_Num_Choice")
    for env_name in env_names:
        env = read_input('train', 0, 'A')
        makespan, _, _ = Least_Mission_Num_Choice(env.init_env)
        total_makespan += makespan
    print("total_makespan:" + str(total_makespan))
    total_makespan = 0
    print("Least_Distance_Choice")
    for env_name in env_names:
        env = read_input('train', 0, 'A')
        makespan, _, _ = Least_Distance_Choice(env.init_env)
        total_makespan += makespan
    print("total_makespan:" + str(total_makespan))
    total_makespan = 0
    print("sa")
    for env_name in env_names:
        env = read_input('train', 0, 'A')
        sa = SA(env, env_name)
        sa.iter_solu.l2i_init()
        print(env.last_step_makespan)
        sa.run()

    # output solution
    # data_name: List[str] = ['train_1_', 'train_2_', 'train_3_', 'train_4_']
    # train_solus: List[IterSolution] = []
    # for i in data_name:
    #     train_solus.append(read_input(i))
    # for i in range(len(data_name)):
    #     print("solution: " + str(i + 1))
    #     solu = train_solus[i]
    #     path = Cf.OUTPUT_SOLUTION_PATH + data_name[i]
    #     write_json_to_file(path + 'RC.json', Random_Choice(solu.init_env)[2])
    #     write_json_to_file(path + 'LW.json', Least_Wait_Time_Choice(solu.init_env)[2])
    #     write_json_to_file(path + 'LM.json', Least_Mission_Num_Choice(solu.init_env)[2])
    #     SA(solu, data_name[i]).run()

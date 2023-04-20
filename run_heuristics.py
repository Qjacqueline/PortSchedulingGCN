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

import conf.configs as cf
from algorithm_factory.algorithm_SA import SA
from algorithm_factory.algorithm_heuristic_rules import Random_Choice, Least_Wait_Time_Choice, Least_Mission_Num_Choice, \
    Least_Distance_Choice, Fixed_order
from data_process.input_process import read_input

if __name__ == '__main__':
    random.seed(cf.RANDOM_SEED)
    np.random.seed(cf.RANDOM_SEED)

    for i in range(8):
        f = open("output_result/sa_100_n.txt", "a") # 1000
        ls = [j + 40 for j in range(10)]
        profiles = [chr(65 + i) + '2' for _ in range(10)]  # '2_1000'
        # ls = [10, 12, 27, 10, 15, 21, 10, 11, 21,
        #       10, 11, 10, 14, 15, 10, 11, 12,
        #       10, 17, 21, 10, 14, 18, 10, 16, 23]
        # profiles = ['A2_t', 'A2_t', 'A2_t', 'B2_t', 'B2_t', 'B2_t', 'C2_t', 'C2_t', 'C2_t',
        #             'D2_t', 'D2_t', 'E2_t', 'E2_t', 'E2_t', 'F2_t', 'F2_t', 'F2_t',
        #             'G2_t', 'G2_t', 'G2_t', 'H2_t', 'H2_t', 'H2_t', 'Z2_t', 'Z2_t', 'Z2_t']

        # ls = [i for i in range(50)]

        # print("Fixed order")
        # env = read_input('train', 10, 'A2_t', 10)
        # makespan, _, _ = Fixed_order(env.init_env, [0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        # print("total_makespan:" + str(makespan))

        # total_makespan = 0
        # print("Random_Choice")
        # for i in range(len(ls)):
        #     env = read_input('train', str(ls[i]), profiles[i], mission_num=ls[i])
        #     makespan, _, _ = Random_Choice(env.init_env)
        #     total_makespan += makespan
        # print("total_makespan:" + str(total_makespan))
        #
        # total_makespan = 0
        # print("Least_Wait_Time_Choice")
        # for i in range(len(ls)):
        #     env = read_input('train', str(ls[i]), profiles[i], mission_num=ls[i])
        #     makespan, _, _ = Least_Wait_Time_Choice(env.init_env)
        #     total_makespan += makespan
        # print("total_makespan:" + str(total_makespan))
        #
        # total_makespan = 0
        # print("Least_Mission_Num_Choice")
        # for i in range(len(ls)):
        #     env = read_input('train', str(ls[i]), profiles[i], mission_num=ls[i])
        #     makespan, _, _ = Least_Mission_Num_Choice(env.init_env)
        #     total_makespan += makespan
        # print("total_makespan:" + str(total_makespan))
        #
        # total_makespan = 0
        # print("Least_Distance_Choice")
        # for i in range(len(ls)):
        #     env = read_input('train', str(ls[i]), profiles[i], mission_num=ls[i])
        #     makespan, _, _ = Least_Distance_Choice(env.init_env)
        #     total_makespan += makespan
        # print("total_makespan:" + str(total_makespan))
        #
        # total_makespan = 0
        # print("sa")
        res = []
        for i in range(len(ls)):
            env = read_input('train', str(ls[i]), profiles[i], mission_num=100)  # 1000
            sa = SA(env)
            sa.iter_solu.l2i_init()
            # print(env.last_step_makespan)
            best_makespan, _ = sa.run()
            res.append(best_makespan)
        f.write("instance\t" + profiles[0] + str(ls[0]) + "\t" + str(sum(res) / 10) + "\n")
        f.close()
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

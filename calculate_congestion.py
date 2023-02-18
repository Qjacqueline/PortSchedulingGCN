# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 11:01 PM
# @Author  : JacQ
# @File    : calculate_congestion.py

import random

import numpy as np

import conf.configs as cf
from algorithm_factory.algo_utils.machine_cal_methods import cal_congestion
from algorithm_factory.algorithm_heuristic_rules import Least_Mission_Num_Choice
from data_process.input_process import read_input

if __name__ == '__main__':
    random.seed(cf.RANDOM_SEED)
    np.random.seed(cf.RANDOM_SEED)
    total_makespan = 0
    f = open("output_result/congestion.txt", "a")
    for j in range(1):
        ls = [i for i in range(50)]
        profiles = [chr(65 + j) + '2' for _ in range(len(ls))]
        congestion_time = []
        for i in range(len(ls)):
            env = read_input('train', str(ls[i]), profiles[i], mission_num=100)
            _, solution, _ = Least_Mission_Num_Choice(env.init_env)
            congestion_time.append(cal_congestion(solution))
        f.write(chr(65 + j) + "_congestion\n")
        for i in range(len(ls)):
            f.write(str(congestion_time[i]) + "\n")

            # f.write(str(max(lb1s[i], lb3s[i])) + "\n")
    f.close()

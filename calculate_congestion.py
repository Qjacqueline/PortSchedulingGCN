# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 11:01 PM
# @Author  : JacQ
# @File    : calculate_congestion.py

import random
import time

import numpy as np

import conf.configs as cf
from algorithm_factory.algo_utils.machine_cal_methods import cal_congestion
from algorithm_factory.algorithm_heuristic_rules import Least_Mission_Num_Choice
from common.iter_solution import IterSolution
from data_process.input_process import read_input
from gurobi_solver import CongestionPortModel, solve_model

if __name__ == '__main__':
    random.seed(cf.RANDOM_SEED)
    np.random.seed(cf.RANDOM_SEED)
    total_makespan = 0

    for j in range(1):

        # ls = [i for i in range(50)]
        # profiles = [chr(65 + j) + '2' for _ in range(len(ls))]

        profiles = [ 'H2_t', 'Z2_t']
        ls = [1000 for _ in range(len(profiles))]
        congestion_time = []
        congestion_time_a = []
        time_forall = []
        for i in range(len(ls)):
            f = open("output_result/congestion_all.txt", "a")
            env = read_input('train', str(ls[i]), profiles[i], mission_num=ls[i])  # mission_num=100
            makespan, env, _ = Least_Mission_Num_Choice(env.init_env)
            congestion_time.append(cal_congestion(env))
            solution = IterSolution(env)
            solution.iter_env = env
            model = CongestionPortModel(solution)
            model.gamma = makespan
            model.construct()
            s_t_g = time.time()
            solve_model(MLP=model.MLP, inst_idx=profiles[i], solved_env=solution, tag='_fix_all')
            e_t_g = time.time()
            congestion_time_a.append(model.MLP.getVars()[-1].x)
            time_forall.append(e_t_g - s_t_g)
            f.write(str(ls[i]) + profiles[i] + "\t" + str(congestion_time[i]) + "\t" +
                    str(congestion_time_a[i]) + "\t" + str(time_forall[i]) + "\n")
            f.close()
        # f.write(chr(65 + j) + "_congestion\n")
        # for i in range(len(ls)):、
        #     f.write(str(congestion_time[i]) + "\n")
        # f.write(str(max(lb1s[i], lb3s[i])) + "\n")


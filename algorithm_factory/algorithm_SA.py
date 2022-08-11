#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling
@File    ：SA.py
@Author  ：JacQ
@Date    ：2022/2/15 14:26
"""
import random
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import conf.configs as Cf
from algorithm_factory.algo_utils.machine_cal_methods import output_solution
from common.iter_solution import IterSolution
from data_process.input_process import write_json_to_file
from utils.log import Logger

logger = Logger().get_logger()
random.seed(Cf.RANDOM_SEED)
np.random.seed(Cf.RANDOM_SEED)


def eval_solution(best_solution, tmp_solution, temp):
    makespan1 = best_solution.last_step_makespan
    makespan2 = tmp_solution.last_step_makespan
    dc = makespan2 - makespan1
    p = max(1e-1, np.exp(-dc / temp))
    if makespan2 < makespan1:
        return tmp_solution
    elif np.random.rand() <= p:
        return tmp_solution
    else:
        return best_solution


class SA(object):
    def __init__(self,
                 iter_solu: IterSolution,
                 data_name: str,
                 T0: float = Cf.T0,
                 Tend: float = Cf.TEND,
                 rate: float = Cf.RATE,
                 buffer_flag: bool = True):
        self.T0: float = T0
        self.Tend: float = Tend
        self.rate: float = rate
        self.iter_solu: IterSolution = iter_solu
        self.data_name = data_name
        self.buffer_flag: bool = buffer_flag
        self.iter_x: list = list()
        self.iter_y: list = list()
        self.best_makespan: float = float('inf')

    # 模拟退火总流程
    def run(self, stop_time=float('inf')):
        # logger.info("开始执行算法SA.")
        T1 = time.time()
        count = 0
        tf1 = True
        tf2 = True
        tf3 = True
        # 记录最优解
        best_solution: IterSolution = deepcopy(self.iter_solu)
        best_makespan: float = self.iter_solu.last_step_makespan
        while self.T0 > self.Tend and time.time() - T1 < stop_time:
            count += 1
            # 产生在这个温度下的随机解
            tmp_solution, flag = self.get_new_solution(deepcopy(self.iter_solu))
            if flag == 'end':
                break
            # 更新最优解
            if tmp_solution.last_step_makespan < best_makespan:
                best_makespan = tmp_solution.last_step_makespan
                best_solution = tmp_solution
                # path = Cf.OUTPUT_SOLUTION_PATH + self.data_name
                # if tmp_solution.last_step_makespan < 17500:
                #     write_json_to_file(
                #         path + 'SA_' + str(Cf.MISSION_NUM_ONE_QUAY_CRANE) + "_" + str(best_makespan) + '.json',
                #         output_solution(best_solution.iter_env))
            # 根据温度判断是否选择这个解
            self.iter_solu = eval_solution(best_solution, tmp_solution, self.T0)

            # 降低温度
            self.T0 *= self.rate
            # 记录路径收敛曲线
            self.iter_x.append(count)
            self.iter_y.append(self.iter_solu.last_step_makespan)
            if time.time() - T1 > 10 and tf1:
                print("best_makespan为:" + str(best_makespan), end=" ")
                tf1 = False
            if time.time() - T1 > 30 and tf2:
                print("best_makespan为:" + str(best_makespan), end=" ")
                tf2 = False
            if time.time() - T1 > 180 and tf3:
                print("best_makespan为:" + str(best_makespan), end=" ")
                tf3 = False
        self.iter_solu.reset()
        print("best_makespan为:" + str(best_makespan) + "time: " + str(time.time() - T1))
        return best_makespan, best_solution

    # 产生一个新的解：随机交换两个元素的位置
    @staticmethod
    def get_new_solution(solution: IterSolution):
        # operator 产生新解
        action = random.randint(0, Cf.ACTION_NUM_SA - 1)
        reward, flag = solution.step_v1(action)
        return solution, flag

    # 退火策略，根据温度变化有一定概率接受差的解

    def plot_result(self):
        # 解决中文乱码
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["font.family"] = "sans-serif"
        # 解决负号无法显示的问题
        plt.rcParams['axes.unicode_minus'] = False
        iterations = self.iter_x
        best_record = self.iter_y
        plt.plot(iterations, best_record)
        plt.title('收敛曲线')
        plt.show()

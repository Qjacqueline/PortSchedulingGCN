#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：port_scheduling
@File    ：algorithm_interface.py
@Author  ：JacQ
@Date    ：2021/12/22 11:01
"""
from algorithm_factory.algorithm_L2S import L2S
from algorithm_factory.algorithm_SA import SA
from algorithm_factory.algorithm_exact import Exact_Method
from algorithm_factory.algorithm_heuristic_rules import Random_Choice, Least_Wait_Time_Choice, Least_Mission_Num_Choice, \
    Least_Distance_Choice, Initial_Solution_RL_V1
from conf.configs import Configs as Cf
from utils.logger import Logger

logger = Logger(name='root').get_logger()


def algorithm_interface(test_env, algorithm_index, RL_config=Cf.RL_CONFIG, train_env=None, buffer_flag=True):
    """
    搜索算法统一接口
    :param RL_config:
    :param buffer_flag:
    :param algorithm_index:
    :param test_env
    :param train_env:
    :return:
    """

    if algorithm_index == 0:
        return Exact_Method(test_env)  # global_mini_makespan, global_mini_instance, global_mini_assign_list
    elif algorithm_index == 1:
        return Random_Choice(test_env, buffer_flag)
    elif algorithm_index == 2:
        return Least_Wait_Time_Choice(test_env, buffer_flag)
    elif algorithm_index == 3:
        return Least_Mission_Num_Choice(test_env, buffer_flag)
    elif algorithm_index == 4:
        return Least_Distance_Choice(test_env, buffer_flag)
    elif algorithm_index == 5:
        return SA(test_env).run()
    elif algorithm_index == 6:
        return L2S(train_envs=train_env, test_envs=test_env, buffer_flag=buffer_flag).run(RL_config)
    elif algorithm_index == 7:
        return Initial_Solution_RL_V1(test_env, buffer_flag)
    else:
        logger.error("算法选择参数配置错误.")
        exit()

# if __name__ == '__main__':
#     generate_data_for_test(10)  # 生成数据
#     port_env = read_input(10)  # 初始化，获取输入数据
#     for i in range(1, 2):
#         algorithm_interface(port_env, 3, True)

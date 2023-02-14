# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 4:26 PM
# @Author  : JacQ
# @File    : gurobi.py

import argparse
import datetime
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
from gurobipy import GRB
from tensorboardX import SummaryWriter

import conf.configs as cf
from algorithm_factory.algo_utils.data_buffer import LABuffer
from algorithm_factory.algo_utils.machine_cal_methods import get_state_n, cal_LB1, cal_LB2, cal_LB3, cal_LB4
from algorithm_factory.algo_utils.net_models import QNet, PQNet
from algorithm_factory.algorithm_heuristic_rules import Least_Distance_Choice, Least_Mission_Num_Choice
from algorithm_factory.rl_algo.lower_agent import DDQN, LACollector
from common import PortEnv
from common.iter_solution import IterSolution
from data_process.input_process import read_input, read_json_from_file, write_env_to_file
from gurobi_solver import CongestionPortModel, solve_model
from utils.log import exp_dir, Logger

logger = Logger().get_logger()


def get_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inst_type', type=str, default=cf.inst_type)
    parser.add_argument('--max_num', type=int, default=5)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gamma', type=float, default=0.9)  # 0.9
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch_num', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=128000)
    parser.add_argument('-save_path', type=str, default=cf.MODEL_PATH)

    command_list = []
    for key, value in kwargs.items():
        command_list.append(f'--{key}={value}')
    return parser.parse_args(command_list)


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # env
    train_solus = []
    test_solus = []

    ''' 单目标 '''
    m_num_ls = [100]
    inst_type_ls = ['Z2_t']
    # gammas = [898.034116453539]
    # # thetas = [1500]

    ''' 多目标epsilon'''
    # delta = 1
    # m_num_ls = [16 for _ in range(delta)]
    # inst_type_ls = ['Z2_t' for _ in range(delta)]
    # gammas = [876.271496288637, 898.034116453539]
    # thetas = [(2 * i + 1) * 0.05 * 268.991273748431 for i in range(2 * delta)]

    ''' 所有统一求'''
    # m_num_ls = [10 for _ in range(9)]
    # inst_type_ls = [chr(i + 65) + '2_t' for i in range(8)]
    # inst_type_ls.append('Z2_t')

    ''' AM'''
    # m_num_ls = [10, 12, 27, 10, 15, 21, 10, 11, 21,
    #             10, 11, 10, 14, 15, 10, 11, 12,
    #             10, 17, 21, 10, 14, 18, 10, 16, 23]
    # inst_type_ls = ['A2_t', 'A2_t', 'A2_t', 'B2_t', 'B2_t', 'B2_t', 'C2_t', 'C2_t', 'C2_t',
    #                 'D2_t', 'D2_t', 'E2_t', 'E2_t', 'E2_t', 'F2_t', 'F2_t', 'F2_t',
    #                 'G2_t', 'G2_t', 'G2_t', 'H2_t', 'H2_t', 'H2_t', 'Z2_t', 'Z2_t', 'Z2_t']

    ''' PM(0)'''
    # m_num_ls = [10, 13, 10, 15, 10, 11,
    #             11, 10, 14, 10, 11,
    #             10, 17, 10, 14, 10, 16]
    # inst_type_ls = ['A2_t', 'A2_t', 'B2_t', 'B2_t', 'C2_t', 'C2_t',
    #                 'D2_t', 'E2_t', 'E2_t', 'F2_t', 'F2_t',
    #                 'G2_t', 'G2_t', 'H2_t', 'H2_t', 'Z2_t', 'Z2_t']
    # gammas = [1183.97772688462, 1223.43166168913, 1048.76179632512,1316.52679188132,991.789726187869,1028.54129185621,
    #           928.892480514753,808.975563654737,914.062174210405,834.011686777709,871.938898423007,
    #           845.047666484882,1029.00345317571,823.93362806332,910.556489215056,780.521551937144,876.271496288637]

    ''' PM_N2'''
    # m_num_ls = [27, 21, 21, 15, 12, 21, 18, 23]
    # inst_type_ls = ['A2_t', 'B2_t', 'C2_t', 'E2_t', 'F2_t', 'G2_t', 'H2_t', 'Z2_t']

    ''' 求不出的'''
    # m_num_ls = [15, 10, 10, 14, 16]
    # inst_type_ls = ['B2_t', 'C2_t', 'D2_t', 'H2_t', 'Z2_t']
    # gammas = [1316.52679188132, 991.789726187869, 907.132467274069, 910.556489215056, 876.271496288637]

    lb1_flag, lb2_flag, lb3_flag = True, True, True
    makespan_forall = []
    congestion_forall = []
    time_forall = []
    for i in range(len(m_num_ls)):
        solu = read_input('train', str(m_num_ls[i]), inst_type_ls[i], m_num_ls[i])
        # 计算下界
        _, lb_env, _ = Least_Mission_Num_Choice(deepcopy(solu.init_env))
        lb1 = cal_LB1(lb_env)
        lb2, r_lb2 = cal_LB2(lb_env)
        lb3 = cal_LB4(lb_env, r_lb2)
        solu.l2a_init()
        model = CongestionPortModel(solu)
        # model.gamma2 = max(lb1, lb2, lb3)
        # model.gamma = 1035.8197896202403
        # model.gamma = gammas[i]
        # model.theta = thetas[i]
        model.construct()
        s_t_g = time.time()
        solve_model(MLP=model.MLP, inst_idx=inst_type_ls[i] + '_' + str(m_num_ls[i]), solved_env=solu, tag='_exact',
                    X_flag=False, Y_flag=False)
        e_t_g = time.time()
        if model.MLP.status != GRB.Status.OPTIMAL:
            makespan_forall.append('inf')
            congestion_forall.append('inf')
        else:
            makespan_forall.append(model.MLP.getVars()[-2].x)
            congestion_forall.append(model.MLP.getVars()[-1].x)
        time_forall.append(e_t_g - s_t_g)
        # if model.MLP.ObjVal == float('Inf'):
        #     break

    for i in range(len(makespan_forall)):
        print("算例为\t" + str(m_num_ls[i]) + "\tmakespan为\t" + str(makespan_forall[i]) +
              "\tcongestion为\t" + str(congestion_forall[i]) + "\t时间为\t" + str(time_forall[i]))

    ''' 检验多目标参数'''
    # m_num_ls = [16]
    # inst_type_ls = ['Z2_t']
    # gammas = [0]  # 876.271496288637
    # theta = [0.01]  # [0.02, 0.03, 0.04, 0.05, 0.08]
    # makespan_forall = []
    # congestion_forall = []
    # time_forall = []
    # for i in range(len(m_num_ls)):
    #     for j in range(len(theta)):
    #         solu = read_input('train', str(m_num_ls[i]), inst_type_ls[i], m_num_ls[i])
    #         solu.l2a_init()
    #         model = CongestionPortModel(solu)
    #         model.gamma = gammas[i] * (1 + theta[j])
    #         model.construct()
    #         s_t_g = time.time()
    #         solve_model(MLP=model.MLP, inst_idx=inst_type_ls[i] + '_' + str(m_num_ls[i]), solved_env=solu, tag='_exact',
    #                     X_flag=False, Y_flag=False)
    #         e_t_g = time.time()
    #         if model.MLP.status != GRB.Status.OPTIMAL:
    #             makespan_forall.append('inf')
    #             congestion_forall.append('inf')
    #         else:
    #             makespan_forall.append(model.MLP.getVars()[-2].x)
    #             congestion_forall.append(model.MLP.getVars()[-1].x)
    #         time_forall.append(e_t_g - s_t_g)
    #         # if model.MLP.ObjVal == float('Inf'):
    #         #     break
    #
    # for i in range(len(makespan_forall)):
    #     print("算例为\t" + str(theta[i % (len(theta))]) + "\tmakespan为\t" + str(makespan_forall[i]) +
    #           "\tcongestion为\t" + str(congestion_forall[i]) + "\t时间为\t" + str(time_forall[i]))

    ''' f_2'''
    # m_num_ls = [10]
    # inst_type_ls = ['D2_t']
    # makespan_forall = []
    # congestion_forall = []
    # time_forall = []
    # for i in range(len(m_num_ls)):
    #     solu = read_input('train', str(m_num_ls[i]), inst_type_ls[i], m_num_ls[i])
    #     solu.l2a_init()
    #     model = CongestionPortModel(solu)
    #     model.construct()
    #     s_t_g = time.time()
    #     solve_model(MLP=model.MLP, inst_idx=inst_type_ls[i] + '_' + str(m_num_ls[i]), solved_env=solu, tag='_exact',
    #                 X_flag=False, Y_flag=False)
    #     e_t_g = time.time()
    #     if model.MLP.status != GRB.Status.OPTIMAL:
    #         makespan_forall.append('inf')
    #         congestion_forall.append('inf')
    #     else:
    #         makespan_forall.append(model.MLP.getVars()[-2].x)
    #         congestion_forall.append(model.MLP.getVars()[-1].x)
    #     time_forall.append(e_t_g - s_t_g)
    #     # if model.MLP.ObjVal == float('Inf'):
    #     #     break
    #
    # for i in range(len(makespan_forall)):
    #     print("算例为\t" + str(m_num_ls[i]) + "\tmakespan为\t" + str(makespan_forall[i]) +
    #           "\tcongestion为\t" + str(congestion_forall[i]) + "\t时间为\t" + str(time_forall[i]))

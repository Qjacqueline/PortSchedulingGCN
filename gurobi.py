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
from tensorboardX import SummaryWriter

import conf.configs as cf
from algorithm_factory.algo_utils.data_buffer import LABuffer
from algorithm_factory.algo_utils.machine_cal_methods import get_state_n
from algorithm_factory.algo_utils.net_models import QNet, PQNet
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
    parser.add_argument('--epsilon', type=float, default=0.5)
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
    # ls = [cf.MISSION_NUM]
    m_num_ls = [17]  # 10,16,17,18,19
    inst_type_ls = ['Z', 'Z'] #'Z', 'Z','Z', 'Z','Z'

    makespan_forall = []
    time_forall = []
    for i in m_num_ls:
        solu = read_input('train', str(i), args.inst_type, i)
        solu.l2a_init()
        model = CongestionPortModel(solu)
        model.construct()
        s_t_g = time.time()
        solve_model(MLP=model.MLP, inst_idx=cf.inst_type + '_' + str(i), solved_env=solu, tag='_exact',
                    X_flag=False, Y_flag=False)
        e_t_g = time.time()
        makespan_forall.append(model.MLP.ObjVal)
        time_forall.append(e_t_g - s_t_g)
        # if model.MLP.ObjVal == float('Inf'):
        #     break

    for i in range(len(makespan_forall)):
        print("算例为" + str(m_num_ls[i]))
        print("gurobi后makespan为" + str(makespan_forall[i]))
        print("gurobi算法时间" + str(time_forall[i]))

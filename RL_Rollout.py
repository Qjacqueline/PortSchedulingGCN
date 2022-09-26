# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 10:16 AM
# @Author  : JacQ
# @File    : rollout.py
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New
@File    ：train_lower_agent.py
@Author  ：JacQ
@Date    ：2022/4/21 10:33
"""

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
from data_process.input_process import read_input, read_json_from_file
from utils.log import exp_dir, Logger

logger = Logger().get_logger()


def get_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=cf.dataset + '_' + str(cf.MISSION_NUM_ONE_QUAY_CRANE))
    # parser.add_argument('--task', type=str, default=cf.dataset + '_' + str(10))
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--mission_num', type=int, default=cf.MISSION_NUM)
    parser.add_argument('--mission_num_each', type=int, default=cf.MISSION_NUM_ONE_QUAY_CRANE)
    parser.add_argument('--dim_action', type=int, default=cf.LOCK_STATION_NUM)
    parser.add_argument('--max_num', type=int, default=5)
    parser.add_argument('--machine_num', type=int, default=22)

    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gamma', type=float, default=0.9)  # 0.9
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=128000)

    parser.add_argument('--epoch_num', type=int, default=60)

    parser.add_argument('-save_path', type=str, default=cf.MODEL_PATH)
    command_list = []
    for key, value in kwargs.items():
        command_list.append(f'--{key}={value}')
    return parser.parse_args(command_list)


if __name__ == '__main__':
    # ==============  Create environment & buffer  =============
    args = get_args()
    exp_dir = exp_dir(desc=f'{args.task}')
    rl_logger = SummaryWriter(exp_dir)
    rl_logger.add_text(tag='parameters', text_string=str(args))
    rl_logger.add_text(tag='characteristic', text_string='New State')  # 'debug'

    # env
    train_solus = []
    test_solus = []
    for i in range(0, 40):
        train_solus.append(read_input('train_' + str(i) + '_'))
    for i in range(0, 50):
        test_solus.append(read_input('train_' + str(i) + '_'))
    for solu in train_solus:
        solu.l2a_init()
    for solu in test_solus:
        solu.l2a_init()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ========================= load RL =========================
    agent = DDQN(
        eval_net=QNet(device=args.device, hidden=args.hidden, max_num=args.max_num),
        target_net=QNet(device=args.device, hidden=args.hidden, max_num=args.max_num),
        dim_action=args.dim_action,
        device=args.device,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr)

    data_buffer = LABuffer(buffer_size=args.buffer_size)
    collector = LACollector(train_solus=train_solus, test_solus=test_solus, data_buffer=data_buffer,
                            batch_size=args.batch_size, mission_num=args.mission_num, agent=agent, rl_logger=rl_logger,
                            save_path=args.save_path, max_num=args.max_num)
    # init eval
    agent.qf = torch.load(args.save_path + '/eval_' + args.task + '.pkl')
    agent.qf_target = torch.load(args.save_path + '/target_' + args.task + '.pkl')
    # makespan_forall, reward_forall = collector.eval()
    # for makespan in makespan_forall:
    #     print("初始la分配makespan为" + str(makespan))
    # print("*********************************************")

    # ========================= Rollout =========================
    s_t = time.time()
    makespan_forall = collector.rollout()
    for makespan in makespan_forall:
        print("rollout后makespan为" + str(makespan))
    e_t = time.time()
    print("算法时间" + str(e_t - s_t))

    # os.rename(exp_dir, f'{exp_dir}_done')
    rl_logger.close()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New
@File    ：train_lower_agent.py
@Author  ：JacQ
@Date    ：2022/4/21 10:33
"""

import argparse
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

import conf.configs as cf
from algorithm_factory.algo_utils.data_buffer import LABuffer
from algorithm_factory.algo_utils.net_models import QNet
from algorithm_factory.rl_algo.lower_agent import DDQN, LACollector
from data_process.input_process import read_input
from utils.log import exp_dir, Logger

logger = Logger().get_logger()


def get_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='la_' + cf.dataset)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--mission_num', type=int, default=cf.MISSION_NUM)
    parser.add_argument('--mission_num_each', type=int, default=cf.MISSION_NUM_ONE_QUAY_CRANE)
    parser.add_argument('--dim_action', type=int, default=cf.LOCK_STATION_NUM)

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gamma', type=float, default=0.5)  # 0.9
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=128000)

    parser.add_argument('--epoch_num', type=int, default=5)

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
    train_solus = [read_input('train_1_'), read_input('train_2_'), read_input('train_3_'), read_input('train_4_'),
                   read_input('train_5_'), read_input('train_6_'), read_input('train_7_'), read_input('train_8_'),
                   read_input('train_9_'), read_input('train_10_')]
    test_solus = [read_input('train_1_'), read_input('train_2_'), read_input('train_3_'), read_input('train_4_'),
                  read_input('train_5_'), read_input('train_6_'), read_input('train_7_'), read_input('train_8_'),
                  read_input('train_9_'), read_input('train_10_'), read_input('train_11_'), read_input('train_12_'),
                  read_input('train_13_'), read_input('train_14_'), read_input('train_15_'), read_input('train_16_'),
                  read_input('train_17_'), read_input('train_18_'), read_input('train_19_'), read_input('train_0_')]
    for solu in train_solus:
        solu.l2a_init()
    for solu in test_solus:
        solu.l2a_init()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ========================= Policy ======================
    agent = DDQN(
        eval_net=QNet(device=args.device),
        target_net=QNet(device=args.device),
        dim_action=args.dim_action,
        device=args.device,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr)

    # ======================== Data ==========================
    data_buffer = LABuffer(buffer_size=args.buffer_size)
    collector = LACollector(train_solus=train_solus, test_solus=test_solus, data_buffer=data_buffer,
                            batch_size=args.batch_size, mission_num=args.mission_num, agent=agent, rl_logger=rl_logger,
                            save_path=args.save_path)
    agent.qf = torch.load(args.save_path + '/eval_best_fixed.pkl')
    agent.qf_target = torch.load(args.save_path + '/target_best_fixed.pkl')
    makespan_forall, reward_forall = collector.eval()
    # =================== heuristic l_train ==================
    # collector.get_transition(
    #     read_json_from_file(cf.OUTPUT_SOLUTION_PATH + 'train_1_SA17139.76892920801.json'), test_solus[1])
    # data_name = ['train_1_', 'train_2_', 'train_3_', 'train_4_', 'train_5_', 'train_6_', 'train_7_', 'train_8_']
    # data_name = ['train_2_']
    # collector.collect_heuristics(data_name)

    # ======================= train =======================
    for i in range(1, args.epoch_num):
        collector.collect_rl()  # 200

    # ======================== eval =========================
    agent.qf = torch.load(args.save_path + '/eval_best_fixed.pkl')
    agent.qf_target = torch.load(args.save_path + '/target_best_fixed.pkl')
    makespan_forall, reward_forall = collector.eval()
    for makespan in makespan_forall:
        print("初始la分配makespan为" + str(makespan))

    # ===================== heuristic l_train ====================
    # collector.collect_heuristics(data_name)
    # l_train(epoch_num=args.epoch_num, dl_train=dl_train, agent=agent, collector=collector, rl_logger=rl_logger)
    # print(collector.best_result)
    os.rename(exp_dir, f'{exp_dir}_done')
    rl_logger.close()

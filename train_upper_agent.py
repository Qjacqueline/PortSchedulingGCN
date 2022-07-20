#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New_Version
@File    ：train_upper_agent_new.py
@Author  ：JacQ
@Date    ：2022/5/11 19:38
"""

import argparse
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

import conf.configs as cf
from algorithm_factory.algo_utils.data_buffer import UANewBuffer
from algorithm_factory.algo_utils.net_models import QNet, ActorNew, CriticNew
from algorithm_factory.rl_algo.lower_agent import DDQN
from algorithm_factory.rl_algo.upper_agent import ACUpper, UANewCollector
from data_process.input_process import read_input
from utils.log import exp_dir, Logger

logger = Logger().get_logger()


def get_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ha')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--quay_num', type=int, default=cf.CRANE_NUM)
    parser.add_argument('--quay_buffer_size', type=int, default=cf.QUAY_BUFFER_SIZE)
    parser.add_argument('--each_quay_m_num', type=int, default=cf.MISSION_NUM_ONE_QUAY_CRANE)
    parser.add_argument('--m_attri_num', type=int, default=7)
    parser.add_argument('--dim_attri', type=int, default=7)

    parser.add_argument('--mission_num', type=int, default=cf.MISSION_NUM)
    parser.add_argument('--m_max_num', type=int, default=10)
    parser.add_argument('--dim_mission_fea', type=float, default=6)
    parser.add_argument('--dim_mach_fea', type=int, default=3)
    parser.add_argument('--dim_yard_fea', type=int, default=cf.FEATURE_SIZE_MACHINE + 2)
    parser.add_argument('--dim_action', type=int, default=4)

    parser.add_argument('--actor_lr', type=float, default=1e-6)
    parser.add_argument('--critic_lr', type=float, default=1e-5)
    parser.add_argument('--u_gamma', type=float, default=0.9)

    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l_gamma', type=float, default=0.9)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=128000)  # 需要大于训练算例数乘以每个算例任务数

    parser.add_argument('--epoch_num', type=int, default=5)
    parser.add_argument('-save_path', type=str, default=cf.MODEL_PATH)

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

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
    rl_logger.add_text(tag='characteristic', text_string='init_ua_n')

    train_solus = [read_input('train_1_'), read_input('train_2_'), read_input('train_3_'), read_input('train_4_')]
    test_solus = [read_input('train_1_'), read_input('train_2_'), read_input('train_3_'), read_input('train_4_'),
                  read_input('train_5_'), read_input('train_6_'), read_input('train_7_'), read_input('train_8_'),
                  read_input('train_9_'), read_input('train_10_'), read_input('train_11_'), read_input('train_12_'),
                  read_input('train_13_'), read_input('train_14_'), read_input('train_15_'), read_input('train_16_'),
                  read_input('train_17_'), read_input('train_18_'), read_input('train_19_'), read_input('train_0_')]
    for solu in train_solus:
        solu.ua_n_init()
    for solu in test_solus:
        solu.ua_n_init()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ========================= Policy ======================
    u_agent = ACUpper(actor=ActorNew(args.device),
                      critic=CriticNew(args.device),
                      actor_lr=args.actor_lr,
                      critic_lr=args.critic_lr,
                      gamma=args.u_gamma,
                      device=args.device)
    l_agent = DDQN(
        eval_net=QNet(device=args.device),
        target_net=QNet(device=args.device),
        dim_action=args.dim_action,
        device=args.device,
        gamma=args.l_gamma,
        epsilon=args.epsilon,
        lr=args.lr)
    l_agent.qf = torch.load(args.save_path + '/eval_best_fixed.pkl')
    l_agent.qf_target = torch.load(args.save_path + '/target_best_fixed.pkl')

    # ======================== Data ==========================
    data_buffer = UANewBuffer(buffer_size=args.buffer_size)
    u_collector = UANewCollector(train_solus=train_solus, test_solus=test_solus, mission_num=args.mission_num,
                                 m_max_num=args.m_max_num, each_quay_m_num=args.each_quay_m_num,
                                 data_buffer=data_buffer, batch_size=args.batch_size, u_agent=u_agent,
                                 l_agent=l_agent, rl_logger=rl_logger, save_path=args.save_path)

    init_makespans, reward = u_collector.eval(l_eval_flag=True)
    for makespan in init_makespans:
        print("初始la分配makespan为" + str(makespan))

    # u_dl_train = DataLoader(dataset=data_buffer, batch_size=args.u_batch_size, shuffle=True)

    # ======================== collect and train (only upper) =========================
    # for i in range(100):
    #     u_collector.collect_rl()
    #     logger.info("开始第" + str(i) + "训练")
    #     u_train(epoch_num=1, dl_train=u_dl_train, agent=u_agent, collector=u_collector, rl_logger=rl_logger)
    #     u_data_buffer.clear()

    # ======================== collect and train (upper lower combine) =========================
    for i in range(args.epoch_num):
        u_collector.collect_rl()
        data_buffer.clear()

    # ======================== eval =========================
    l_agent.qf = torch.load(args.save_path + '/eval_best_la.pkl')
    l_agent.qf_target = torch.load(args.save_path + '/target_best_la.pkl')
    l_agent.u_agent.actor = torch.load(args.save_path + '/actor_best_fixed.pkl')
    l_agent.u_agent.critic = torch.load(args.save_path + '/critic_best_fixed.pkl')
    makespan_forall, reward = u_collector.eval()
    for makespan in makespan_forall:
        print("初始la分配makespan为" + str(makespan))

    os.rename(exp_dir, f'{exp_dir}_done')
    rl_logger.close()

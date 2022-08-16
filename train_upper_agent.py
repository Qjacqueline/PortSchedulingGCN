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
import time

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
    parser.add_argument('--task', type=str, default=cf.dataset + '_' + str(cf.MISSION_NUM_ONE_QUAY_CRANE))
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--each_quay_m_num', type=int, default=cf.MISSION_NUM_ONE_QUAY_CRANE)
    parser.add_argument('--mission_num', type=int, default=cf.MISSION_NUM)
    parser.add_argument('--m_max_num', type=int, default=2)
    parser.add_argument('--dim_action', type=int, default=4)
    parser.add_argument('--machine_num', type=int, default=22)

    parser.add_argument('--actor_lr', type=float, default=1e-5)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--u_gamma', type=float, default=0.9)

    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l_gamma', type=float, default=0.9)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=128000)  # 需要大于训练算例数乘以每个算例任务数

    parser.add_argument('--epoch_num', type=int, default=20)
    parser.add_argument('-save_path', type=str, default=cf.MODEL_PATH)

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    command_list = []
    for key, value in kwargs.items():
        command_list.append(f'--{key}={value}')
    return parser.parse_args(command_list)


if __name__ == '__main__':
    # ==============  Create environment & buffer  =============
    args = get_args()
    task = 'ha_' + args.task
    exp_dir = exp_dir(desc=f'{task}')
    rl_logger = SummaryWriter(exp_dir)
    rl_logger.add_text(tag='parameters', text_string=str(args))
    rl_logger.add_text(tag='characteristic', text_string='init_ua_n')
    s_t = time.time()
    # env
    train_solus = []
    test_solus = []
    for i in range(0, 40):
        train_solus.append(read_input('train_' + str(i) + '_'))
    for i in range(0, 50):
        test_solus.append(read_input('train_' + str(i) + '_'))
    for solu in train_solus:
        solu.ua_n_init()
    for solu in test_solus:
        solu.ua_n_init()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ========================= Policy ======================
    u_agent = ACUpper(
        actor=ActorNew(device=args.device, hidden=args.hidden, max_num=args.m_max_num, machine_num=args.machine_num),
        critic=CriticNew(device=args.device, hidden=args.hidden, max_num=args.m_max_num, machine_num=args.machine_num),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.u_gamma,
        device=args.device)
    l_agent = DDQN(
        eval_net=QNet(device=args.device, hidden=args.hidden, max_num=args.m_max_num, machine_num=args.machine_num),
        target_net=QNet(device=args.device, hidden=args.hidden, max_num=args.m_max_num, machine_num=args.machine_num),
        dim_action=args.dim_action,
        device=args.device,
        gamma=args.l_gamma,
        epsilon=args.epsilon,
        lr=args.lr)

    # ======================== Data ==========================
    data_buffer = UANewBuffer(buffer_size=args.buffer_size)
    u_collector = UANewCollector(train_solus=train_solus, test_solus=test_solus, mission_num=args.mission_num,
                                 m_max_num=args.m_max_num, each_quay_m_num=args.each_quay_m_num,
                                 data_buffer=data_buffer, batch_size=args.batch_size, u_agent=u_agent,
                                 l_agent=l_agent, rl_logger=rl_logger, save_path=args.save_path)

    # ======================== pre评估 ==========================
    # l_agent.qf = torch.load(args.save_path + '/eval_' + args.task + 'l.pkl')
    # l_agent.qf_target = torch.load(args.save_path + '/target_' + args.task + 'l.pkl')
    # u_agent.actor = torch.load(args.save_path + '/actor_' + args.task + 'l.pkl')
    # u_agent.critic = torch.load(args.save_path + '/critic_' + args.task + 'l.pkl')
    # makespan_forall, reward = u_collector.eval()
    # for makespan in makespan_forall:
    #     print("初始la分配makespan为" + str(makespan))

    # l_agent.qf = torch.load(args.save_path + '/eval_' + args.task + '_f.pkl')
    # l_agent.qf_target = torch.load(args.save_path + '/target_' + args.task + '_f.pkl')
    # for i in range(15):
    #     u_collector.init_release_time_gap = i * 10
    #
    #     init_makespans, reward = u_collector.eval(l_eval_flag=True)
    #     print(str(i * 10) + " : " + str(init_makespans[-2]))

    # ======================== collect and train (upper lower combine) =========================
    s_t = time.time()
    l_agent.qf = torch.load(args.save_path + '/eval_' + args.task + '_f.pkl')
    l_agent.qf_target = torch.load(args.save_path + '/target_' + args.task + '_f.pkl')
    for i in range(args.epoch_num):
        u_collector.collect_rl()

    e_t = time.time()
    print("training time" + str(e_t - s_t))

    # ======================== eval =========================
    l_agent.qf = torch.load(args.save_path + '/eval_' + args.task + 'l.pkl')
    l_agent.qf_target = torch.load(args.save_path + '/target_' + args.task + 'l.pkl')
    u_agent.actor = torch.load(args.save_path + '/actor_' + args.task + 'l.pkl')
    u_agent.critic = torch.load(args.save_path + '/critic_' + args.task + 'l.pkl')
    makespan_forall, reward = u_collector.eval()
    for makespan in makespan_forall:
        print("初始la分配makespan为" + str(makespan))

    os.rename(exp_dir, f'{exp_dir}_done')
    rl_logger.close()

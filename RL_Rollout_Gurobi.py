# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 8:48 AM
# @Author  : JacQ
# @File    : RL_Rollout_Gurobi.py
# -*- coding: utf-8 -*-

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
from utils.log import exp_dir, Logger

logger = Logger().get_logger()


def get_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inst_type', type=str, default=cf.inst_type)
    parser.add_argument('--mission_num', type=int, default=cf.MISSION_NUM)
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
    # ==============  Create environment & buffer  =============
    args = get_args()

    for j in range(8):
        # env
        train_solus = []
        test_solus = []

        '''同instance 环境批量输入'''
        ls = [i for i in range(1)]
        profiles = [chr(65 + j) + '2' for _ in range(1)]

        for i in range(len(ls)):
            train_solus.append(read_input('train', str(ls[i]), profiles[i], args.mission_num))
        for i in range(len(ls)):
            test_solus.append(read_input('train', str(ls[i]), profiles[i], args.mission_num))
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
            eval_net=QNet(device=args.device, in_dim_max=args.max_num, hidden=args.hidden,
                          out_dim=train_solus[0].init_env.ls_num, ma_num=train_solus[0].init_env.machine_num),
            target_net=QNet(device=args.device, in_dim_max=args.max_num, hidden=args.hidden,
                            out_dim=train_solus[0].init_env.ls_num, ma_num=train_solus[0].init_env.machine_num),
            dim_action=train_solus[0].init_env.ls_num,
            device=args.device,
            gamma=args.gamma,
            epsilon=args.epsilon,
            lr=args.lr)

        # ======================== Data ==========================
        data_buffer = LABuffer(buffer_size=args.buffer_size)
        collector = LACollector(inst_type=args.inst_type, train_solus=train_solus, test_solus=test_solus,
                                data_buffer=data_buffer, batch_size=args.batch_size,
                                mission_num=train_solus[0].init_env.J_num_all, agent=agent,
                                rl_logger=None, save_path=args.save_path, max_num=args.max_num)

        # init eval
        agent.qf = torch.load(args.save_path + '/eval_' + profiles[0][0:2] + '.pkl')
        agent.qf_target = torch.load(args.save_path + '/target_' + profiles[0][0:2] + '.pkl')

        # ========================= mode =========================
        RL_flag, rollout_flag, rollout_flag_100, rollout_flag_10, rollout_flag_5, rollout_flag_2, exact_flag, fix_xjm, fix_all = \
            True, False, False, False, False, True, False, False, True

        # ========================= RL =========================
        if RL_flag:
            makespan_forall_RL, reward_forall, time_forall_RL = collector.eval(False)

        # ========================= Rollout =========================
        if rollout_flag:
            makespan_forall_rollout, _, time_forall_rollout = collector.rollout()
        if rollout_flag_100:
            makespan_forall_rollout_100, _, time_forall_rollout_100 = collector.rollout_100()
        if rollout_flag_10:
            makespan_forall_rollout_10, _, time_forall_rollout_10 = collector.rollout_10()
        if rollout_flag_5:
            makespan_forall_rollout_5, _, time_forall_rollout_5 = collector.rollout_5()
        if rollout_flag_2:
            makespan_forall_rollout_2, _, time_forall_rollout_2 = collector.rollout_2()
        # write_env_to_file(solu.iter_env, 0, cf.MISSION_NUM_ONE_QUAY_CRANE)

        # ========================= Gurobi =========================
        # mode 1 直接精确算法求解
        if exact_flag:
            makespan_forall_gurobi, time_g = collector.exact(args.inst_type)

        # mode 2 fix Xjm加solver
        if fix_xjm:
            makespan_forall_gurobi2, time_g2 = collector.exact_fix_x(args.inst_type)

        # mode 3 fix all加solver
        if fix_all:
            makespan_forall_gurobi3, time_g3 = collector.exact_fix_all(args.inst_type)

        # branch_and_bound
        # makespan_forall_gurobi4, time_g4 = collector.bb_depth_wide(solu, global_UB=makespan_forall_gurobi3[0] + 1e-5)

        # ========================= Print Result =========================
        f = open("output_result/rl_100_exact.txt", "a")
        for i in range(len(ls)):
            f.write("instance:\t" + profiles[i] + " " + str(ls[i]) + "\t")
            if exact_flag:
                f.write(str(makespan_forall_gurobi[i]) + "\t")
                f.write(str(time_g[i]), end='\t')
            if RL_flag:
                f.write(str(makespan_forall_RL[i]) + "\t")
                f.write(str(time_forall_RL[i]) + "\t")
            if rollout_flag:
                f.write(str(makespan_forall_rollout[i]) + "\t")
                f.write(str(time_forall_rollout[i]) + "\t")
            if rollout_flag_100:
                f.write(str(makespan_forall_rollout_100[i]) + "\t")
                f.write(str(time_forall_rollout_100[i]) + "\t")
            if rollout_flag_10:
                f.write(str(makespan_forall_rollout_10[i]) + "\t")
                f.write(str(time_forall_rollout_10[i]) + "\t")
            if rollout_flag_5:
                f.write(str(makespan_forall_rollout_5[i]) + "\t")
                f.write(str(time_forall_rollout_5[i]) + "\t")
            if rollout_flag_2:
                f.write(str(makespan_forall_rollout_2[i]) + "\t")
                f.write(str(time_forall_rollout_2[i]) + "\t")
            if fix_xjm:
                f.write(str(makespan_forall_gurobi2[i]) + "\t")
                f.write(str(time_g2[i]), end='\t')
            if fix_all:
                f.write(str(makespan_forall_gurobi3[i]) + "\t")
                f.write(str(time_g3[i]) + "\t")
            f.write("\n")
        f.close()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Learning to schedule
"""
@Project ：Port_Scheduling
@File    ：Two_level_learning_controller.py
@Author  ：JacQ
@Date    ：2021/12/28 16:52
"""
from copy import deepcopy

import torch
from algorithm_factory.rl_algo import LA_V2
from algorithm_factory.RL_algorithm import LA_V3
from algorithm_factory.RL_algorithm import LA_V4
from algorithm_factory.RL_algorithm import LA_V5
from algorithm_factory.RL_algorithm import LA_V6
from algorithm_factory.RL_algorithm import UA_V1
from algorithm_factory.RL_algorithm.LA_V1_L2I import LA_V1
from algorithm_factory.RL_algorithm.LA_V61_RNN_DDQN import LA_V61
from algorithm_factory.RL_algorithm.UA_V2 import UA_V2
from machine_cal_methods import process_init_solution_for_RNN
from mission_cal_methods import derive_mission_attribute_list
from torch.utils.tensorboard import SummaryWriter

from algorithm_factory.algorithm_heuristic_rules import Initial_Solution_RL_V1
from conf.configs import Configs as Cf
from utils.exp_file import exp_file
from utils.logger import Logger

logger = Logger(name='root').get_logger()


class L2S(object):
    def __init__(self, train_envs, test_envs, buffer_flag=True):
        self.train_envs = train_envs
        self.test_envs = test_envs
        self.best_solution_lower = train_envs
        self.buffer_flag = buffer_flag
        self.best_makespan_lower = float('inf')
        self.epochs_policy_losses = []
        self.epochs_vf_losses = []
        self.epochs_policy_losses = []
        self.epochs_makespan_history = []

    def run(self, RL_config):
        if RL_config == 1:
            self.train_lower_agent_V1()
        if RL_config == 2:
            self.train_lower_agent_V2()
        if RL_config == 3:
            self.train_lower_agent_V3()
        if RL_config == 4:
            self.train_lower_agent_V4()
        if RL_config == 5:
            self.train_lower_agent_V5()
        if RL_config == 6:
            self.train_lower_agent_V6()
        if RL_config == 7:
            self.train_upper_level_agents()
        if RL_config == 8:
            self.train_upper_agent_L2A()

        return self.best_makespan_lower, self.best_solution_lower

    def train_lower_agent_V1(self):
        path, time_str = exp_file('v1_')
        writer = SummaryWriter(path)
        init_makespan, train_solution = Initial_Solution_RL_V1(self.train_envs, self.buffer_flag)
        lower_agent = LA_V1()
        logger.info("开始执行算法RLV1")
        # 执行RL算法
        for epoch in range(Cf.N_EPOCH_LA1):
            lower_agent.iter_solution = deepcopy(train_solution)
            lower_agent.run(epoch)
            if lower_agent.best_makespan < self.best_makespan_lower:
                self.best_makespan_lower = lower_agent.best_makespan
                self.best_solution_lower = lower_agent.best_solution
                torch.save(lower_agent.actor, Cf.MODEL_PATH + '/LAV1_actor_best_' + str(Cf.MISSION_NUM) + '.pkl')
                torch.save(lower_agent.critic, Cf.MODEL_PATH + '/LAV1_critic_best_' + str(Cf.MISSION_NUM) + '.pkl')
        writer.add_scalar("makespan", lower_agent.iter_solution.last_step_makespan, global_step=epoch)
        logger.info("训练最优makespan:" + str(self.best_makespan_lower))
        # 测试算法
        lower_agent.actor = torch.load(Cf.MODEL_PATH + '/LAV1_actor_best_' + str(Cf.MISSION_NUM) + '.pkl')
        lower_agent.critic = torch.load(Cf.MODEL_PATH + '/LAV1_critic_best_' + str(Cf.MISSION_NUM) + '.pkl')
        _, test_solution = Initial_Solution_RL_V1(self.test_envs, self.buffer_flag)
        lower_agent.iter_solution = deepcopy(test_solution)
        lower_agent.eval()

    def train_lower_agent_V2(self):
        path, time_str = exp_file('v2_')
        writer = SummaryWriter(path)
        lower_agent = LA_V2()
        lower_agent.iter_solution = deepcopy(self.train_envs)
        # 训练RL算法
        logger.info("开始执行算法RLV2")
        for epoch in range(Cf.N_EPOCH_LA2):
            lower_agent.run(epoch, writer)
            if lower_agent.iter_solution.last_step_makespan < self.best_makespan_lower:
                self.best_makespan_lower = lower_agent.iter_solution.last_step_makespan
                self.best_solution_lower = lower_agent.iter_solution
                torch.save(lower_agent.actor, Cf.MODEL_PATH + '/LAV2_actor_best' + time_str + '.pkl')
                torch.save(lower_agent.critic, Cf.MODEL_PATH + '/LAV2_critic_best' + time_str + '.pkl')
            writer.add_scalar("makespan", lower_agent.iter_solution.last_step_makespan, global_step=epoch)
            lower_agent.iter_solution = deepcopy(self.train_envs)
        torch.save(lower_agent.actor, Cf.MODEL_PATH + '/LAV2_actor_last' + time_str + '.pkl')
        torch.save(lower_agent.critic, Cf.MODEL_PATH + '/LAV2_critic_last' + time_str + '.pkl')
        logger.info("训练最优makespan:" + str(self.best_makespan_lower))
        # 测试算法
        lower_agent.actor = torch.load(Cf.MODEL_PATH + '/LAV2_actor_best' + time_str + '.pkl')
        lower_agent.critic = torch.load(Cf.MODEL_PATH + '/LAV2_critic_best' + time_str + '.pkl')
        lower_agent.iter_solution = deepcopy(self.test_envs)
        lower_agent.eval()

    def train_lower_agent_V3(self):
        path, time_str = exp_file('v3_')
        writer = SummaryWriter(path)
        lower_agent = LA_V3(deepcopy(self.train_envs))
        # 训练RL算法
        logger.info("开始执行算法RLV3")
        for epoch in range(Cf.N_EPOCH_LA3):
            lower_agent.run(epoch)
            if lower_agent.iter_solution.last_step_makespan < self.best_makespan_lower:
                self.best_makespan_lower = lower_agent.iter_solution.last_step_makespan
                self.best_solution_lower = lower_agent.iter_solution
                torch.save(lower_agent.construct_model, Cf.MODEL_PATH + '/LAV3_best.pkl')
            lower_agent.iter_solution = deepcopy(deepcopy(self.train_envs))
        logger.info("训练最优makespan:" + str(self.best_makespan_lower))
        # 测试算法
        lower_agent.construct_model = torch.load(Cf.MODEL_PATH + '/LAV3_best.pkl')
        lower_agent.iter_solution = deepcopy(self.test_envs)
        lower_agent.eval()

    def train_lower_agent_V4(self):
        path, time_str = exp_file('v4_')
        writer = SummaryWriter(path)
        lower_agent = LA_V4(deepcopy(self.train_envs))
        # 训练RL算法
        logger.info("开始执行算法RLV4")
        for epoch in range(Cf.N_EPOCH_LA4):
            lower_agent.run(epoch)
            if lower_agent.iter_solution.last_step_makespan < self.best_makespan_lower:
                self.best_makespan_lower = lower_agent.iter_solution.last_step_makespan
                self.best_solution_lower = lower_agent.iter_solution
                torch.save(lower_agent.actor, Cf.MODEL_PATH + '/LAV4_actor_best.pkl')
                torch.save(lower_agent.critic, Cf.MODEL_PATH + '/LAV4_critic_best.pkl')
            lower_agent.iter_solution = deepcopy(deepcopy(self.train_envs))
        logger.info("训练最优makespan:" + str(self.best_makespan_lower))
        # 测试算法
        lower_agent.actor = torch.load(Cf.MODEL_PATH + '/LAV4_actor_best.pkl')
        lower_agent.critic = torch.load(Cf.MODEL_PATH + '/LAV4_critic_best.pkl')
        lower_agent.iter_solution = deepcopy(self.test_envs)
        lower_agent.eval()

    def train_lower_agent_V5(self):
        path, time_str = exp_file('v5_')
        writer = SummaryWriter(path)
        lower_agent = LA_V5(deepcopy(self.train_envs))
        # 训练RL算法
        logger.info("开始执行算法RLV5")
        for epoch in range(Cf.N_EPOCH_LA5):
            lower_agent.run(epoch)
            if lower_agent.iter_solution.last_step_makespan < self.best_makespan_lower:
                self.best_makespan_lower = lower_agent.iter_solution.last_step_makespan
                self.best_solution_lower = lower_agent.iter_solution
                torch.save(lower_agent.eval_net, Cf.MODEL_PATH + '/LAV5_eval_best.pkl')
                torch.save(lower_agent.target_net, Cf.MODEL_PATH + '/LAV5_target_best.pkl')
            lower_agent.iter_solution = deepcopy(deepcopy(self.train_envs))
        logger.info("训练最优makespan:" + str(self.best_makespan_lower))
        # 测试算法
        lower_agent.eval_net = torch.load(Cf.MODEL_PATH + '/LAV5_eval_best.pkl')
        lower_agent.target_net = torch.load(Cf.MODEL_PATH + '/LAV5_target_best.pkl')
        lower_agent.iter_solution = deepcopy(self.test_envs)
        lower_agent.eval()

    def train_lower_agent_V6(self):
        path, time_str = exp_file('v6_')
        writer = SummaryWriter(path)
        lower_agent = LA_V61()
        # 训练RL算法
        logger.info("开始执行算法RLV6")
        # lower_agent.iter_solution = deepcopy(deepcopy(self.train_envs))
        # lower_agent.eval()
        for epoch in range(Cf.N_EPOCH_LA6):
            lower_agent.iter_solution = deepcopy(self.train_envs)
            lower_agent.run(epoch, writer)
            if lower_agent.iter_solution.last_step_makespan < self.best_makespan_lower:
                self.best_makespan_lower = lower_agent.iter_solution.last_step_makespan
                self.best_solution_lower = lower_agent.iter_solution
                torch.save(lower_agent.eval_net, Cf.MODEL_PATH + '/LAV6_eval_best_' + time_str + '.pkl')
                torch.save(lower_agent.target_net, Cf.MODEL_PATH + '/LAV6_target_best_' + time_str + '.pkl')
            writer.add_scalar("makespan", lower_agent.iter_solution.last_step_makespan, global_step=epoch)
            lower_agent.iter_solution = deepcopy(deepcopy(self.train_envs))
            lower_agent.eval()
        logger.info("训练最优makespan:" + str(self.best_makespan_lower))
        torch.save(lower_agent.eval_net, Cf.MODEL_PATH + '/LAV6_eval_last_' + time_str + '.pkl')
        torch.save(lower_agent.target_net, Cf.MODEL_PATH + '/LAV6_target_last_' + time_str + '.pkl')
        # 测试算法
        lower_agent.eval_net = torch.load(Cf.MODEL_PATH + '/LAV6_eval_best_' + time_str + '.pkl')
        lower_agent.target_net = torch.load(Cf.MODEL_PATH + '/LAV6_target_best_' + time_str + '.pkl')
        lower_agent.iter_solution = deepcopy(self.test_envs)
        lower_agent.eval()
        lower_agent.eval_net = torch.load(Cf.MODEL_PATH + '/LAV6_eval_last_' + time_str + '.pkl')
        lower_agent.target_net = torch.load(Cf.MODEL_PATH + '/LAV6_target_last_' + time_str + '.pkl')
        lower_agent.iter_solution = deepcopy(self.test_envs)
        lower_agent.eval()

    def train_upper_level_agents(self):
        mission_attribute_list = derive_mission_attribute_list(self.train_envs)
        upper_agent = UA_V1()  # iter_mission_list, mission_attribute_list, deepcopy(self.train_envs)
        upper_agent.l_train(,

    def train_upper_agent_L2A(self):
        path, time_str = exp_file('UA_')
        writer = SummaryWriter(path)
        upper_agent = UA_V2()  # iter_mission_list, mission_attribute_list, deepcopy(self.train_envs)
        upper_agent.lower_agent = LA_V6()
        upper_agent.lower_agent.eval_net = torch.load(Cf.MODEL_PATH + '/LAV6_eval_best04_14_10:01:51.pkl')
        upper_agent.lower_agent.target_net = torch.load(Cf.MODEL_PATH + '/LAV6_target_best04_14_10:01:51.pkl')
        yard_cranes_set_test = process_init_solution_for_RNN(self.test_envs)  # self.mission绑定上去
        mission_attribute_list_test = derive_mission_attribute_list(self.test_envs.mission_list)  # 得到mission属性向量列表
        # l_train upper u_agent
        for train_env in self.train_envs:
            # 处理初始环境，返回初始state和port_state
            yard_cranes_set = process_init_solution_for_RNN(train_env)  # self.mission绑定上去
            mission_attribute_list = derive_mission_attribute_list(train_env.mission_list)  # 得到mission属性向量列表
            for epoch in range(Cf.N_EPOCH_UA2):
                upper_agent.iter_solution = deepcopy(train_env)
                upper_agent.mission_attribute_list = mission_attribute_list.copy()
                upper_agent.yard_cranes_set = yard_cranes_set
                upper_agent.run(epoch, writer)  # derive result
                upper_agent.iter_solution = deepcopy(self.test_envs)
                upper_agent.mission_attribute_list = mission_attribute_list_test.copy()
                upper_agent.yard_cranes_set = yard_cranes_set_test
                upper_agent.eval()
                if upper_agent.iter_solution.last_step_makespan < self.best_makespan_lower:
                    self.best_makespan_lower = upper_agent.iter_solution.last_step_makespan
                    self.best_solution_lower = upper_agent.iter_solution
                    torch.save(upper_agent.actor, Cf.MODEL_PATH + '/UA2_actor_best_' + time_str + '.pkl')
                    torch.save(upper_agent.critic, Cf.MODEL_PATH + '/UA2_critic_best_' + time_str + '.pkl')
                logger.info(str(epoch) + "训练makespan:" + str(upper_agent.iter_solution.last_step_makespan))
                writer.add_scalar("makespan", upper_agent.iter_solution.last_step_makespan, global_step=epoch)

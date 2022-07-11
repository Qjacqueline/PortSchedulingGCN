#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling
@File    ：lower_agent_v4.py
@Author  ：JacQ
@Date    ：2022/3/31 16:08
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm_factory.RL_algorithm.State_Model import get_graph_state, Graph_Policy_Net, \
    Graph_Value_Net
from conf.configs import Configs as Cf
from machine_cal_methods import process_init_solution_for_RNN
from utils.logger import Logger

logger = Logger(name='root').get_logger()


# Strategy: learn to assign / RL:Actor Critic / State:Graph / Action 选锁站
class LA_V4(nn.Module):
    def __init__(self, init_solution):
        super(LA_V4, self).__init__()
        # 环境
        self.iter_solution = init_solution
        self.mission_num = Cf.MISSION_NUM
        self.device = Cf.DEVICE
        self.buffer_flag = True
        # RL网络
        self.gamma = Cf.GAMMA_LA4
        self.actor = Graph_Policy_Net(Cf.NUM_NODE_FEATURES_LA4, Cf.ACTION_NUM_LA4, Cf.MISSION_NUM, hidden_dim=64).to(
            self.device)  # policy network
        self.critic = Graph_Value_Net(Cf.NUM_NODE_FEATURES_LA4, Cf.ACTION_NUM_LA4, Cf.MISSION_NUM, hidden_dim=64).to(
            self.device)  # Value network
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), lr=Cf.POLICY_LR_LA4)
        self.vf_optimizer = torch.optim.Adam(self.critic.parameters(), lr=Cf.VF_LR_LA4)
        # 记录结果变量
        self.makespan_history = [init_solution.last_step_makespan]
        self.rewards = []
        self.saved_log_probs = []
        self.transition = []
        self.policy_losses = []
        self.vf_losses = []
        self.logger = logger
        self.best_makespan = float("inf")
        self.best_solution = deepcopy(init_solution)
        self.total_makespan_delta = 0

    def select_action(self, state):
        action, pi, log_pi, state = self.actor(state)  # pi每个动作的概率，log_pi
        v = self.critic(state)
        self.saved_log_probs.append([log_pi, v])
        return action.detach().cpu().numpy(), log_pi, v

    def train_model(self):
        log_pi, v, reward, next_state = self.transition
        next_v = self.critic(next_state)  # prediction V(s')

        # target for Q regression
        q = reward + self.gamma * next_v
        q.to(self.device)

        # td error：Advantage = Q - V
        advantage = q - v
        logger.debug('advantage: {}, q: {}, v: {}, log_pi: {}'.format(advantage, q, v, log_pi))

        # A2C losses
        policy_loss = -log_pi * advantage.detach()  # true_gradient = grad[logPi(s, a) * td_error]
        vf_loss = F.mse_loss(v, q.detach())  # gradient = grad[r + gamma * V(s_) - V(s)]

        # update value network parameter
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        # update policy network parameter
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save losses
        self.policy_losses.append(policy_loss.item())
        self.vf_losses.append(vf_loss.item())

    def run(self, epoch):
        self.vf_losses = []
        self.policy_losses = []
        self.rewards = []
        step_number = 0
        total_reward = 0
        yard_cranes_set = process_init_solution_for_RNN(self.iter_solution)  # self.mission绑定上去
        state = get_graph_state(self.iter_solution)
        # 训练步骤开始
        while step_number < Cf.N_STEPS_LA4:
            cur_mission = self.iter_solution.mission_list[step_number]
            action, log_pi, v = self.select_action(state)
            makespan_delta, station_makespan = self.iter_solution.step_v2(int(action), cur_mission, step_number,
                                                                          self.buffer_flag)
            reward = 1 / self.iter_solution.last_step_makespan
            self.rewards.append(reward)
            step_number += 1
            new_state = get_graph_state(self.iter_solution)
            state = new_state
            self.transition = log_pi, v, reward, new_state
            self.train_model()
            total_reward += reward
        logger.info("第" + str(epoch + 1) + "epoch训练最终makespan:" + str(self.iter_solution.last_step_makespan))
        # self.plot_train_result(epoch)

    def eval(self):
        print("----------测试开始----------")
        self.vf_losses = []
        self.policy_losses = []
        step_number = 0
        total_reward = 0
        yard_cranes_set = process_init_solution_for_RNN(self.iter_solution)  # self.mission绑定上去
        state = get_graph_state(self.iter_solution)
        # 测试步骤开始
        with torch.no_grad():
            while step_number < Cf.N_STEPS_LA4:
                cur_mission = self.iter_solution.mission_list[step_number]
                action, log_pi, v = self.select_action(state)
                self.iter_solution.step_v2(int(action), cur_mission, step_number, self.buffer_flag)
                self.makespan_history.append(self.iter_solution.last_step_makespan)
                step_number += 1
                new_state = get_graph_state(self.iter_solution)
                state = new_state
        logger.info("测试最终makespan:" + str(self.iter_solution.last_step_makespan))
        # self.plot_test_result()

    def plot_train_result(self, epoch):
        x1 = range(0, len(self.policy_losses))
        x2 = range(0, len(self.policy_losses))
        x3 = range(0, len(self.makespan_history))
        y1 = self.policy_losses
        y2 = self.vf_losses
        y3 = self.makespan_history
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        plt.subplot(3, 1, 1)
        plt.plot(x1, y1, 'o-', linewidth=1, markersize=3)
        # plt.title('Policy loss vs. iterations')
        plt.ylabel('policy loss')
        plt.title('第%depoch迭代结果' % epoch)
        plt.subplot(3, 1, 2)
        plt.plot(x2, y2, '.-', linewidth=1, markersize=3)
        # plt.xlabel('Vf_loss vs. iterations')
        plt.ylabel('Vf_loss')
        plt.subplot(3, 1, 3)
        x3 = range(0, len(self.rewards))
        y3 = self.rewards
        plt.plot(x3, y3, '.-', linewidth=1, markersize=3)
        plt.ylabel('reward')
        plt.xlabel('iterations')

        # plt.show()
        plt.savefig(
            Cf.LOSS_PLOT_PATH + str(epoch) + "_" + str(Cf.N_STEPS_LA2) + "_" + str(len(self.policy_losses)) + ".jpg")
        plt.show()

    def plot_test_result(self):
        x4 = range(0, len(self.makespan_history))
        y4 = self.makespan_history
        plt.plot(x4, y4, '.-', linewidth=1, markersize=3)
        plt.xlabel('iterations')
        plt.ylabel('makespan')

        x3 = range(0, len(self.rewards))
        y3 = self.rewards
        plt.plot(x3, y3, '.-', linewidth=1, markersize=3)
        plt.ylabel('reward')
        plt.xlabel('iterations')
        plt.savefig(
            Cf.LOSS_PLOT_PATH + str(Cf.N_STEPS_LA2) + "_" + str(len(self.policy_losses)) + ".jpg")
        plt.show()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Port_Scheduling_New_Version
@File    ：lower_agent_new.py
@Author  ：JacQ
@Date    ：2022/5/22 15:09
'''
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List

import numpy as np
import torch
from torch import nn

import conf.configs as cf
from algorithm_factory.algo_utils.data_buffer import LABuffer
from algorithm_factory.algo_utils.machine_cal_methods import get_rnn_state_v2
from algorithm_factory.algo_utils.net_models import QNet
from algorithm_factory.algo_utils.rl_methods import process_single_state
from algorithm_factory.algo_utils.rl_methods import soft_update
from common.iter_solution import IterSolution
from data_process.input_process import read_json_from_file
from utils.log import Logger

logger = Logger().get_logger()


class BaseAgent(ABC, nn.Module):
    def __init__(self):
        super(BaseAgent, self).__init__()
        """
        """

    @abstractmethod
    def forward(self, **kwargs):
        """
        """

    @abstractmethod
    def update(self, **kwargs):
        """

        """

    def sync_weight(self) -> None:
        """        Soft-update the weight for the target network.        """
        soft_update(tgt=self.qf_target, src=self.qf, tau=self.update_tau)


class DDQNNew(BaseAgent):
    def __init__(self,
                 eval_net: QNet,
                 target_net: QNet,
                 dim_action: int,
                 device: torch.device,
                 gamma: float,
                 epsilon: float,
                 lr: float,
                 soft_update_tau: float = 0.05,
                 loss_fn: Callable = nn.MSELoss(),
                 ) -> None:
        """

        Args:
            eval_net:
            target_net:
            device:
            gamma:
            epsilon:
            soft_update_tau:
            loss_fn:
        """
        super(DDQNNew, self).__init__()
        logger.info("创建DDQN u_agent")
        self.qf = eval_net.to(device)
        self.qf_target = target_net.to(device)
        self.optimizer = torch.optim.Adam(self.qf.parameters(), lr=lr)
        self.dim_action = dim_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.update_tau = soft_update_tau
        self.device = device
        self.loss_func = loss_fn
        self.train_count = 0

    def forward(self, state, eval_tag=True):
        if eval_tag:
            if np.random.rand() <= self.epsilon:  # greedy policy
                s_mission, s_station, s_cross, s_yard = state
                action = torch.max(self.qf(s_mission.unsqueeze(0).to(self.device), s_station, s_cross, s_yard), 1)[
                    1].item()
            else:  # random policy
                action = np.random.randint(0, self.dim_action)
        else:
            s_mission, s_station, s_cross, s_yard = state
            action = torch.max(self.qf(s_mission.to(self.device).unsqueeze(0), s_station, s_cross, s_yard), 1)[1].item()
        return action

    def update(self, batch: Dict[str, Any]):
        s_mission = batch['state_mission'].to(self.device)
        s_station = batch['state_station']
        s_cross = batch['state_cross']
        s_yard = batch['state_yard']
        action = batch['l_action'].unsqueeze(1).to(self.device)
        s_mission_ = batch['state_mission_'].to(self.device)
        s_station_ = batch['state_station_']
        s_cross_ = batch['state_cross_']
        s_yard_ = batch['state_yard_']
        rewards = batch['reward'].unsqueeze(1).to(self.device)
        done = batch['done'].unsqueeze(1).to(self.device)
        q_eval_value = self.qf.forward(s_mission, s_station, s_cross, s_yard)
        q_next_value = self.qf_target.forward(s_mission_, s_station_, s_cross_, s_yard_)
        q_eval = q_eval_value.gather(1, action)
        q_next = q_next_value.gather(1, torch.max(q_next_value, 1)[1].unsqueeze(1))
        q_target = rewards / 100.0 + self.gamma * q_next * (1.0 - done)
        loss = self.loss_func(q_eval, q_target.detach())
        # print(torch.mean(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.train_count % 1 == 0:
            self.sync_weight()
        self.train_count += 1
        return loss
    # self.writer.add_scalar("loss", loss, global_step=self.learn_step_counter)


class LACollector:
    def __init__(self,
                 train_solus: List[IterSolution],
                 test_solus: List[IterSolution],
                 data_buffer: LABuffer,
                 mission_num: int,
                 m_max_num: int,
                 agent: DDQNNew,
                 save_path: str):
        logger.info("创建data Collector")
        self.train_solus = train_solus
        self.test_solus = test_solus
        self.data_buffer = data_buffer
        self.mission_num = mission_num
        self.m_max_num = m_max_num
        self.agent = agent
        self.best_result = [float('Inf') for _ in range(len(self.test_solus) + 1)]
        self.save_path = save_path

    def collect_rl(self, collect_epoch_num: int):
        for i in range(collect_epoch_num):
            # logger.info("第" + str(i) + "轮收集RL交互数据")
            for solu in self.train_solus:
                done = 0
                pre_makespan = 0
                state = get_rnn_state_v2(solu.iter_env, 0, self.m_max_num)
                for step in range(self.mission_num):
                    cur_mission = solu.iter_env.mission_list[step]
                    action = self.agent.forward(process_single_state(state))
                    # print(action)
                    makespan = solu.step_v2(action, cur_mission, step)
                    reward = (pre_makespan - makespan)  # exp(-makespan / 10000)
                    if step == self.mission_num - 1:
                        done = 1
                        new_state = state
                        # reward = -makespan
                    else:
                        new_state = get_rnn_state_v2(solu.iter_env, step + 1, self.m_max_num)
                        # reward = 0
                    pre_makespan = makespan

                    self.data_buffer.append(state, action, new_state, reward, done)
                    state = new_state
                    step += 1
                solu.reset()

    def eval(self):
        with torch.no_grad():
            makespan_forall = []
            for i in range(len(self.test_solus)):
                solu = self.test_solus[i]
                state = get_rnn_state_v2(solu.iter_env, 0, self.m_max_num)
                for step in range(self.mission_num):
                    cur_mission = solu.iter_env.mission_list[step]
                    action = self.agent.forward(process_single_state(state), False)
                    makespan = solu.step_v2(action, cur_mission, step)
                    if step == self.mission_num - 1:
                        new_state = state
                    else:
                        new_state = get_rnn_state_v2(solu.iter_env, step + 1, self.m_max_num)
                    state = new_state
                    step += 1
                makespan_forall.append(makespan)
                if makespan < self.best_result[i]:
                    self.best_result[i] = makespan
                    torch.save(self.agent.qf, self.save_path + '/eval_best.pkl')
                    torch.save(self.agent.qf_target, self.save_path + '/target_best.pkl')
                solu.reset()
            makespan_forall.append(sum(makespan_forall))
            if sum(makespan_forall) < self.best_result[-1]:
                self.best_result[-1] = sum(makespan_forall)
                torch.save(self.agent.qf, self.save_path + '/eval_best_fixed.pkl')
                torch.save(self.agent.qf_target, self.save_path + '/target_best_fixed.pkl')
        return makespan_forall

    def collect_heuristics(self, data_ls: List[str]) -> None:
        logger.info("收集启发式方法数据")
        filenames = os.listdir(cf.OUTPUT_SOLUTION_PATH)
        for i in range(len(data_ls)):
            selected = [x for x in filenames if x[0:7] == data_ls[i][0:7]]
            solu = self.test_solus[int(data_ls[i][6]) - 1]
            for file in selected:
                print(file)
                self.get_transition(read_json_from_file(cf.OUTPUT_SOLUTION_PATH + file), solu)

    def get_transition(self, assign_dict: dict, solu: IterSolution) -> None:
        done = 0
        pre_makespan = 0
        state = get_rnn_state_v2(solu.iter_env, 0, self.m_max_num)
        for step in range(self.mission_num):
            cur_mission = solu.iter_env.mission_list[step]
            action = assign_dict[cur_mission.idx]
            makespan = solu.step_v2(action, cur_mission, step)
            reward = (pre_makespan - makespan)  # exp(-makespan / 10000)
            pre_makespan = makespan
            if step == self.mission_num - 1:
                done = 1
                new_state = state
                # reward = -makespan
            else:
                new_state = get_rnn_state_v2(solu.iter_env, step + 1, self.m_max_num)
                # reward = 0

            self.data_buffer.append(state, action, new_state, reward, done)
            state = new_state
            step += 1
        print(makespan)
        solu.reset()

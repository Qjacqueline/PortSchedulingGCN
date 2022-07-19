#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New
@File    ：lower_agent.py
@Author  ：JacQ
@Date    ：2022/4/21 15:24
"""
import os
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import conf.configs as cf
from algorithm_factory.algo_utils.data_buffer import LABuffer
from algorithm_factory.algo_utils.machine_cal_methods import get_state
from algorithm_factory.algo_utils.net_models import QNet
from algorithm_factory.algo_utils.rl_methods import print_result
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


class DDQN(BaseAgent):
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
        super(DDQN, self).__init__()
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
                action = torch.max(self.qf(state), 1)[1].item()
            else:  # random policy
                action = np.random.randint(0, self.dim_action)
        else:
            action = torch.max(self.qf(state), 1)[1].item()
        return action

    def update(self, batch: List):
        s = batch[0].to(self.device)
        action = batch[1].unsqueeze(1).to(self.device)
        s_ = batch[2].to(self.device)
        rewards = batch[3].unsqueeze(1).to(self.device)
        done = batch[4].unsqueeze(1).to(self.device)
        q_eval_value = self.qf.forward(s)
        q_next_value = self.qf_target.forward(s_)
        q_eval = q_eval_value.gather(1, action)
        q_next = q_next_value.gather(1, torch.max(q_next_value, 1)[1].unsqueeze(1))
        q_target = rewards / 100.0 + self.gamma * q_next * (1.0 - done)
        loss = self.loss_func(q_eval, q_target.detach())
        # print(torch.mean(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.train_count % 10 == 0:
            self.sync_weight()
        self.train_count += 1
        return loss.detach(), q_eval.detach().mean(), q_eval_value.detach().mean()


class LACollector:
    def __init__(self,
                 train_solus: List[IterSolution],
                 test_solus: List[IterSolution],
                 data_buffer: LABuffer,
                 mission_num: int,
                 agent: DDQN,
                 save_path: str):
        logger.info("创建data Collector")
        self.train_solus = train_solus
        self.test_solus = test_solus
        self.data_buffer = data_buffer
        self.mission_num = mission_num
        self.agent = agent
        self.best_result = [float('Inf') for _ in range(len(self.test_solus) + 1)]
        self.save_path = save_path

    def collect_rl(self, collect_epoch_num: int):
        for i in range(collect_epoch_num):
            for solu in self.train_solus:
                done = 0
                pre_makespan = 0
                state = get_state(solu.iter_env, 0)
                for step in range(self.mission_num):
                    cur_mission = solu.iter_env.mission_list[step]
                    action = self.agent.forward(state)
                    # print(action)
                    makespan = solu.step_v2(action, cur_mission, step)
                    reward = (pre_makespan - makespan)  # exp(-makespan / 10000)
                    if step == self.mission_num - 1:
                        done = 1
                        new_state = state
                        # reward = -makespan
                    else:
                        new_state = get_state(solu.iter_env, step + 1)
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
                state = get_state(solu.iter_env, 0)
                for step in range(self.mission_num):
                    cur_mission = solu.iter_env.mission_list[step]
                    action = self.agent.forward(state)
                    makespan = solu.step_v2(action, cur_mission, step)
                    if step == self.mission_num - 1:
                        new_state = state
                    else:
                        new_state = get_state(solu.iter_env, step + 1)
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
            selected = [x for x in filenames if x[0:14] == data_ls[i][0:8] + str("SA_100_")]
            solu = self.test_solus[int(data_ls[i][6]) - 1]
            for file in selected:
                print(file)
                self.get_transition(read_json_from_file(cf.OUTPUT_SOLUTION_PATH + file), solu)

    def get_transition(self, assign_dict: dict, solu: IterSolution) -> None:
        done = 0
        pre_makespan = 0
        state = get_state(solu.iter_env, 0)
        makespan = 0
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
                new_state = get_state(solu.iter_env, step + 1)
                # reward = 0

            self.data_buffer.append(state, action, new_state, reward, done)
            state = new_state
            step += 1
        print(makespan)
        solu.reset()


def l_train(train_time: int, epoch_num: int, dl_train: DataLoader, agent: DDQN, collector: LACollector,
            rl_logger: SummaryWriter) -> None:
    for epoch in range(epoch_num):
        with tqdm(dl_train, desc=f'epoch{epoch}', ncols=100) as pbar:
            total_loss = 0
            total_q_eval = 0
            total_q_eval_value = 0
            for batch in pbar:  # TODO 不需要所有都train 平均过10-100遍就够了
                loss, q_eval, q_eval_value = agent.update(batch)
                total_loss += loss.data
                total_q_eval += q_eval.data
                total_q_eval_value += q_eval_value.data
            makespan = collector.eval()
            rl_logger.add_scalar(tag=f'l_train/loss', scalar_value=total_loss / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            rl_logger.add_scalar(tag=f'l_train/q', scalar_value=total_q_eval / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            rl_logger.add_scalar(tag=f'l_train/q_all', scalar_value=total_q_eval_value / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            field_name = ['Epoch', 'loss']
            value = [epoch, total_loss / len(pbar)]
            for i in range(len(makespan)):
                rl_logger.add_scalar(tag=f'l_train/makespan' + str(i + 1), scalar_value=makespan[i],
                                     global_step=epoch + train_time * epoch_num)
                field_name.append('makespan' + str(i + 1))
                value.append(makespan[i])

            print_result(field_name=field_name, value=value)

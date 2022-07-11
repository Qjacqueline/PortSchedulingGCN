#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New
@File    ：upper_agent.py
@Author  ：JacQ
@Date    ：2022/4/25 11:28
"""
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithm_factory.algo_utils import sort_missions
from algorithm_factory.algo_utils.data_buffer import UABuffer, LABuffer
from algorithm_factory.algo_utils.machine_cal_methods import get_rnn_state_v2
from algorithm_factory.algo_utils.net_models import Actor, Critic
from algorithm_factory.algo_utils.rl_methods import print_result, process_single_state
from algorithm_factory.algo_utils.rl_methods import soft_update
from algorithm_factory.rl_algo.lower_agent import DDQN, LACollector
from common.iter_solution import IterSolution
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
        soft_update(self.target_net, self.eval_net, self.soft_target_tau)


class AC(BaseAgent):
    def __init__(self,
                 actor: Actor,
                 critic: Critic,
                 actor_lr: float,
                 critic_lr: float,
                 gamma: float,
                 device: torch.device,
                 loss_fn: Callable = nn.MSELoss(),
                 ) -> None:
        super(AC, self).__init__()
        logger.info("创建AC u_agent")
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device
        self.loss_func = loss_fn

    def forward(self, state: torch.Tensor, adjust: torch.Tensor):
        action, _ = self.actor.forward(state.to(self.device), adjust.to(self.device))
        return action.detach().cpu()

    def update(self, batch: Dict[str, Any]):
        s = batch['u_state'].to(self.device)  # batch_size*to_release_num*m_attri_dim
        adjust = batch['adjust'].to(self.device)  # batch_size*1*m_attri_dim_each_quay
        action = batch['u_action'].to(self.device)  # batch_size*1*quay_num
        s_ = batch['state_'].to(self.device)  # batch_size*to_release_num*m_attri_dim
        rewards = batch['reward'].to(self.device)  # batch_size*1

        _, log_pi = self.actor.forward(s, adjust)

        v = self.critic.forward(s)
        next_v = self.critic.forward(s_)  # prediction V(s') / with grad
        q = rewards + self.gamma * next_v

        advantage = q - v

        # A2C losses
        # true_gradient = grad[logPi(s, a) * td_error]
        actor_loss = -torch.mean(torch.sum(log_pi, 1).unsqueeze(1) * advantage.detach())
        critic_loss = F.mse_loss(v, q.detach())

        # update value network parameter
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update policy network parameter
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss, critic_loss


class UACollector:
    def __init__(self,
                 train_solus: List[IterSolution],
                 test_solus: List[IterSolution],
                 quay_num: int,
                 quay_buffer_size: int,
                 each_quay_m_num: int,
                 m_max_num: int,
                 m_attri_num: int,
                 u_data_buffer: UABuffer,
                 l_data_buffer: LABuffer,
                 u_agent: AC,
                 l_agent: DDQN,
                 save_path: str):
        logger.info("创建data Collector")
        self.train_solus = train_solus
        self.test_solus = test_solus
        self.quay_num = quay_num
        self.quay_buffer_size = quay_buffer_size
        self.each_quay_m_num = each_quay_m_num
        self.m_max_num = m_max_num
        self.m_attri_num = m_attri_num
        self.u_data_buffer = u_data_buffer
        self.l_data_buffer = l_data_buffer
        self.u_agent = u_agent
        self.l_agent = l_agent
        self.best_result = [float('Inf') for i in range(len(self.test_solus))]
        self.save_path = save_path
        self.pre_makespan = 0
        self.l_state_ls = []
        self.l_action_ls = []
        self.l_reward_ls = []

    def collect_rl(self):
        logger.info("收集RL交互数据")
        for solu in self.train_solus:
            self.l_action_ls = []
            self.l_reward_ls = []
            self.l_state_ls = []
            u_state = self.process_initial_ua(solu)
            self.pre_makespan = 0.0
            for i in range(0, self.each_quay_m_num):
                adjust = self.get_adjust(i)
                u_action = self.u_agent.forward(u_state, adjust)  # 3 to_release_job
                reward, u_new_state = self.la_process(solu, u_state, u_action, i)
                self.u_data_buffer.append(u_state, u_action, reward, u_new_state, adjust)
                u_state = u_new_state
            for i in range(len(self.l_state_ls) - 1):
                self.l_data_buffer.append(self.l_state_ls[i], self.l_action_ls[i], self.l_state_ls[i + 1],
                                          self.l_reward_ls[i], 0)
            self.l_data_buffer.append(self.l_state_ls[-1], self.l_action_ls[-1], self.l_state_ls[-1],
                                      self.l_reward_ls[-1], 1)
            solu.reset()

    def eval(self):
        with torch.no_grad():
            makespan_forall = []
            for j in range(len(self.test_solus)):
                solu = self.test_solus[j]
                u_state = self.process_initial_ua(solu)
                self.pre_makespan = 0.0
                for i in range(0, self.each_quay_m_num):
                    adjust = self.get_adjust(i)
                    u_action = self.u_agent.forward(u_state, adjust)  # 3 to_release_job
                    reward, u_new_state = self.la_process(solu, u_state, u_action, i)
                    u_state = u_new_state
                makespan_forall.append(solu.last_step_makespan)
                if solu.last_step_makespan < self.best_result[j]:
                    self.best_result[j] = solu.last_step_makespan
                    torch.save(self.u_agent.actor, self.save_path + '/actor_best.pkl')
                    torch.save(self.u_agent.critic, self.save_path + '/critic_best.pkl')
                solu.reset()
            return makespan_forall

    def process_initial_ua(self, solu: IterSolution):
        state = []
        for j in range(self.quay_num):
            for k in range(self.quay_buffer_size):
                state.append(
                    solu.attri_ls['M' + str(j * self.each_quay_m_num + k + 1)])
        state = torch.tensor(state).unsqueeze(0)
        return state

    def get_adjust(self, step: int) -> torch.Tensor:
        adjust = torch.ones(1, self.quay_buffer_size).type(torch.int64)
        if step >= self.each_quay_m_num - self.quay_buffer_size:
            adjust[:, self.each_quay_m_num - step:] = 0
        return adjust

    def la_process(self, solu: IterSolution, u_state: torch.tensor, u_action: torch.Tensor, step: int):
        u_new_state = u_state.clone()
        l_action = None
        for i in range(self.quay_num):
            # 定义当前任务，得到u vector新状态
            cur_mission, u_new_state = solu.step_v3(action=u_action.numpy()[0], step=step, i=i,
                                                    new_state=u_new_state,
                                                    quay_num=self.quay_num,
                                                    quay_buffer_size=self.quay_buffer_size,
                                                    each_quay_m_num=self.each_quay_m_num,
                                                    m_max_num=self.m_max_num,
                                                    m_attri_num=self.m_attri_num)
            # 结合当前任务，得到l rnn状态
            l_state = get_rnn_state_v2(iter_solution=solu.iter_env, step_number=int(cur_mission.idx[1:]) - 1,
                                       max_num=self.m_max_num, cur_mission=cur_mission)
            # 做出选锁站决策
            l_action = self.l_agent.forward(process_single_state(l_state))

            # 更新iter_env.mission_list排序
            getattr(sort_missions, 'RELEASE_ORDER')(solu.iter_env.mission_list)  # TODO 可加快速度
            # 根据当前任务，选的锁站更新环境
            _ = solu.step_v2(action=l_action, mission=cur_mission, step_number=step * self.quay_num + i)
            # 更新record
            self.l_state_ls.append(l_state)
            self.l_action_ls.append(l_action)
            self.l_reward_ls.append(self.pre_makespan - solu.last_step_makespan)
            self.pre_makespan = solu.last_step_makespan
        return self.pre_makespan - solu.last_step_makespan, u_new_state


def u_train(epoch_num: int, dl_train: DataLoader, agent: AC, collector: UACollector, rl_logger: SummaryWriter) -> None:
    for epoch in range(epoch_num):
        with tqdm(dl_train, desc=f'epoch{epoch}', ncols=100) as pbar:
            total_policy_loss = 0
            total_vf_loss = 0
            for batch in pbar:
                policy_loss, vf_loss = agent.update(batch)
                total_policy_loss += policy_loss.data
                total_vf_loss += vf_loss.data
            makespan = collector.eval()
            rl_logger.add_scalar(tag=f'u_train/policy_loss', scalar_value=total_policy_loss / len(pbar),
                                 global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/vf_loss', scalar_value=total_vf_loss / len(pbar), global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan1', scalar_value=makespan[0], global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan2', scalar_value=makespan[1], global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan3', scalar_value=makespan[2], global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan4', scalar_value=makespan[3], global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan5', scalar_value=makespan[4], global_step=epoch)
            print_result(
                field_name=['Epoch', 'policy_loss', 'vf_loss', 'makespan1', 'makespan2', 'makespan3', 'makespan4',
                            'makespan5'],
                value=[epoch, total_policy_loss / len(pbar), total_vf_loss / len(pbar), makespan[0], makespan[1],
                       makespan[2], makespan[3], makespan[4]])


def h_train(epoch_num: int, u_dl_train: DataLoader, l_dl_train: DataLoader, u_agent: AC, l_agent: DDQN,
            u_collector: UACollector, l_collector: LACollector, rl_logger: SummaryWriter) -> None:
    for epoch in range(epoch_num):
        with tqdm(u_dl_train, desc=f'epoch{epoch}', ncols=100) as pbar:
            total_policy_loss = 0
            total_vf_loss = 0
            for batch in pbar:
                policy_loss, vf_loss = u_agent.update(batch)
                total_policy_loss += policy_loss.data
                total_vf_loss += vf_loss.data
            makespan = u_collector.eval()
            rl_logger.add_scalar(tag=f'u_train/policy_loss', scalar_value=total_policy_loss / len(pbar),
                                 global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/vf_loss', scalar_value=total_vf_loss / len(pbar), global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan1', scalar_value=makespan[0], global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan2', scalar_value=makespan[1], global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan3', scalar_value=makespan[2], global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan4', scalar_value=makespan[3], global_step=epoch)
            rl_logger.add_scalar(tag=f'u_train/makespan5', scalar_value=makespan[4], global_step=epoch)
            print_result(
                field_name=['Epoch', 'policy_loss', 'vf_loss', 'makespan1', 'makespan2', 'makespan3', 'makespan4',
                            'makespan5'],
                value=[epoch, total_policy_loss / len(pbar), total_vf_loss / len(pbar), makespan[0], makespan[1],
                       makespan[2], makespan[3], makespan[4]])
        with tqdm(l_dl_train, desc=f'epoch{epoch}', ncols=100) as pbar:
            total_loss = 0
            for batch in pbar:
                loss = l_agent.update(batch)  # s_mission, s_station, s_cross, s_yard, l_station, l_cross, l_yard
                total_loss += loss.data
            makespan = l_collector.eval()
            rl_logger.add_scalar(tag=f'l_train/loss', scalar_value=total_loss / len(pbar), global_step=epoch)
            rl_logger.add_scalar(tag=f'l_train/makespan1', scalar_value=makespan[0], global_step=epoch)
            rl_logger.add_scalar(tag=f'l_train/makespan2', scalar_value=makespan[1], global_step=epoch)
            rl_logger.add_scalar(tag=f'l_train/makespan3', scalar_value=makespan[2], global_step=epoch)
            rl_logger.add_scalar(tag=f'l_train/makespan4', scalar_value=makespan[3], global_step=epoch)
            rl_logger.add_scalar(tag=f'l_train/makespan5', scalar_value=makespan[4], global_step=epoch)
            print_result(
                field_name=['Epoch', 'loss', 'makespan1', 'makespan2', 'makespan3', 'makespan4', 'makespan5'],
                value=[epoch, total_loss / len(pbar), makespan[0], makespan[1], makespan[2], makespan[3],
                       makespan[4]])
            # print_result(
            #     field_name=['Epoch', 'loss', 'makespan1', 'makespan2'],
            #     value=[epoch, total_loss / len(pbar), makespan[0], makespan[1]])

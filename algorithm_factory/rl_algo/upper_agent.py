#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New_Version
@File    ：upper_agent_new.py
@Author  ：JacQ
@Date    ：2022/5/11 17:31
"""

from common.mission import Mission

from abc import ABC, abstractmethod
from typing import Callable, List

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from algorithm_factory.algo_utils.data_buffer import UANewBuffer
from algorithm_factory.algo_utils.machine_cal_methods import get_rnn_state_v2, get_state
from algorithm_factory.algo_utils.net_models import ActorNew, CriticNew
from algorithm_factory.algo_utils.rl_methods import print_result
from algorithm_factory.algo_utils.rl_methods import soft_update
from algorithm_factory.rl_algo.lower_agent import DDQN
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


class ACNew(BaseAgent):
    def __init__(self,
                 actor: ActorNew,
                 critic: CriticNew,
                 actor_lr: float,
                 critic_lr: float,
                 gamma: float,
                 device: torch.device,
                 loss_fn: Callable = nn.MSELoss(),
                 ) -> None:
        super(ACNew, self).__init__()
        logger.info("创建DDPG u_agent")
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # self.optimizer = torch.optim.Adam(self.qf.parameters(), lr=1e-4)
        self.gamma = gamma
        self.device = device
        self.loss_func = loss_fn

        self.noise_lim = torch.tensor(0.3)
        self.noise_std = torch.tensor(0.1)

    def forward(self, state, eval_tag=True):
        action = self.actor.forward(state)
        if eval_tag:
            return action.detach().cpu()
        else:
            noise = (self.noise_std * torch.randn_like(action)).clamp(-self.noise_lim, self.noise_lim).to(self.device)
            action = (action + noise).clamp(0.0, 1.0)
            return action.detach().cpu()

    def update(self, batch: List):
        s = batch[0].to(self.device)
        l_action = batch[1].unsqueeze(1).to(self.device)
        s_ = batch[2].to(self.device)
        rewards = batch[3].unsqueeze(1).to(self.device)
        done = batch[4].unsqueeze(1).to(self.device)
        u_action = batch[5].unsqueeze(1).to(self.device)

        u_a = self.actor(s)
        n_u_a = self.actor(s_)

        # Compute the target Q value
        target_Q = self.critic(s_, n_u_a, l_action)
        target_Q = rewards / 100.0 + (done * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(s, u_action, l_action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(s, u_a, l_action).mean()  # Deterministic DPG 找Qmax 确定性策略
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss, critic_loss


class UANewCollector:
    def __init__(self,
                 train_solus: List[IterSolution],
                 test_solus: List[IterSolution],
                 mission_num: int,
                 m_max_num: int,
                 each_quay_m_num: int,
                 data_buffer: UANewBuffer,
                 u_agent: ACNew,
                 l_agent: DDQN,
                 save_path: str):
        logger.info("创建data Collector")
        self.train_solus = train_solus
        self.test_solus = test_solus
        self.mission_num = mission_num
        self.m_max_num = m_max_num
        self.each_quay_m_num = each_quay_m_num
        self.data_buffer = data_buffer
        self.u_agent = u_agent
        self.l_agent = l_agent
        self.best_result = [float('Inf') for _ in range(len(self.test_solus))]
        self.save_path = save_path
        self.pre_makespan = 0
        self.curr_time = [0, 0, 0]

    def collect_rl(self, collect_epoch_num: int):
        for i in range(collect_epoch_num):
            for solu in self.train_solus:
                self.curr_time = [0, 0, 0]
                pre_makespan = 0
                state = get_state(solu.iter_env, 0)
                for step in range(self.mission_num):
                    cur_mission = solu.iter_env.mission_list[step]
                    u_action = self.u_agent(state)
                    l_action = self.l_agent.forward(state, False)  # fixme
                    makespan = self.process(solu, cur_mission, u_action, l_action, step, self.each_quay_m_num)
                    reward = (pre_makespan - makespan)
                    if step != self.mission_num - 1:
                        next_mission = solu.iter_env.mission_list[step + 1]
                        self.process_release_adjust(next_mission, self.each_quay_m_num)
                        new_state = get_state(iter_solution=solu.iter_env, step_number=step + 1)
                        done = 0
                    else:
                        new_state = state
                        done = 1
                    self.data_buffer.append(state=state, u_ac=int(u_action), l_ac=l_action, new_state=new_state,
                                            r=reward,
                                            done=done)
                solu.reset()

    def eval(self, eval_flag=False):
        with torch.no_grad():
            makespan_forall = []
            for i in range(len(self.test_solus)):
                solu = self.test_solus[i]
                state = get_state(solu.iter_env, 0)
                self.curr_time = [0, 0, 0]
                for step in range(self.mission_num):
                    cur_mission = solu.iter_env.mission_list[step]
                    if eval_flag:
                        u_action = torch.tensor(90)
                    else:
                        u_action = self.u_agent.forward(state)
                    l_action = self.l_agent.forward(state, False)
                    reward = self.process(solu, cur_mission, u_action, l_action, step, self.each_quay_m_num)
                    if step != self.mission_num - 1:
                        next_mission = solu.iter_env.mission_list[step + 1]
                        self.process_release_adjust(next_mission, self.each_quay_m_num)
                        new_state = get_state(iter_solution=solu.iter_env, step_number=step + 1)
                        done = 0
                    else:
                        new_state = state
                        done = 1
                makespan_forall.append(solu.last_step_makespan)
                if solu.last_step_makespan < self.best_result[i]:
                    self.best_result[i] = solu.last_step_makespan
                    torch.save(self.u_agent.actor, self.save_path + '/actor_best.pkl')
                    torch.save(self.u_agent.critic, self.save_path + '/critic_best.pkl')
                solu.reset()
            makespan_forall.append(sum(makespan_forall))
            if sum(makespan_forall) < self.best_result[-1]:
                self.best_result[-1] = sum(makespan_forall)
                torch.save(self.u_agent.actor, self.save_path + '/actor_best_fixed.pkl')
                torch.save(self.u_agent.critic, self.save_path + '/critic_best_fixed.pkl')
            return makespan_forall

    def process(self, solu: IterSolution, mission: Mission, u_action: torch.Tensor, l_action: torch.Tensor, step: int,
                each_quay_m_num: int):
        # print(mission.idx)
        self.process_release_adjust(mission, each_quay_m_num, u_action)
        cur_makespan = solu.step_v2(action=l_action, mission=mission, step_number=step)
        return cur_makespan

    def process_release_adjust(self, mission: Mission, each_quay_m_num, u_action=torch.tensor(0)):
        adjust_time = mission.release_time
        if int(mission.idx[1:]) % each_quay_m_num == 1:
            mission.release_time = 0
            adjust_time = 0
        elif int(mission.idx[1:]) <= each_quay_m_num:
            adjust_time = self.curr_time[0] + u_action.item()
            self.curr_time[0] = adjust_time
        elif 100 < int(mission.idx[1:]) <= each_quay_m_num * 2:
            adjust_time = self.curr_time[1] + u_action.item()
            self.curr_time[1] = adjust_time
        elif 200 < int(mission.idx[1:]) <= each_quay_m_num * 3:
            adjust_time = self.curr_time[2] + u_action.item()
            self.curr_time[2] = adjust_time
        transfer_time = mission.machine_start_time[1] - mission.release_time
        mission.machine_start_time[0] = adjust_time
        mission.machine_start_time[1] = adjust_time + transfer_time
        mission.release_time = adjust_time


def u_train(epoch_num: int, dl_train: DataLoader, agent: ACNew, collector: UANewCollector,
            rl_logger: SummaryWriter) -> None:
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


def h_new_train(train_time: int, epoch_num: int, u_dl_train: DataLoader, u_agent: ACNew, l_agent: DDQN,
                u_collector: UANewCollector, rl_logger: SummaryWriter) -> None:
    for epoch in range(epoch_num):
        with tqdm(u_dl_train, desc=f'epoch{epoch}', ncols=100) as pbar:
            total_policy_loss = 0
            total_vf_loss = 0
            total_loss = 0
            total_q_eval = 0
            total_q_eval_value = 0
            for batch in pbar:
                policy_loss, vf_loss = u_agent.update(batch)
                total_policy_loss += policy_loss.data
                total_vf_loss += vf_loss.data
                loss, q_eval, q_eval_value = l_agent.update(batch)
                total_loss += loss.data
                total_q_eval += q_eval.data
                total_q_eval_value += q_eval_value.data

            makespan = u_collector.eval()
            rl_logger.add_scalar(tag=f'u_train/policy_loss', scalar_value=total_policy_loss / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            rl_logger.add_scalar(tag=f'u_train/vf_loss', scalar_value=total_vf_loss / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            rl_logger.add_scalar(tag=f'u_train/q_loss', scalar_value=total_loss / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            rl_logger.add_scalar(tag=f'u_train/q', scalar_value=total_q_eval / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            rl_logger.add_scalar(tag=f'u_train/q_all', scalar_value=total_q_eval_value / len(pbar),
                                 global_step=epoch + train_time * epoch_num)
            field_name = ['Epoch', 'policy_loss', 'vf_loss', 'q_loss']
            value = [epoch, total_policy_loss / len(pbar), total_vf_loss / len(pbar), total_loss / len(pbar)]
            for i in range(len(makespan)):
                rl_logger.add_scalar(tag=f'l_train/makespan' + str(i + 1), scalar_value=makespan[i],
                                     global_step=epoch + train_time * epoch_num)
                field_name.append('makespan' + str(i + 1))
                value.append(makespan[i])

            print_result(field_name=field_name, value=value)

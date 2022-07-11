#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New_Version
@File    ：upper_agent_new.py
@Author  ：JacQ
@Date    ：2022/5/11 17:31
"""
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
from common.mission import Mission

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

from algorithm_factory.algo_utils.data_buffer import UANewBuffer
from algorithm_factory.algo_utils.machine_cal_methods import get_rnn_state_v2
from algorithm_factory.algo_utils.net_models import ActorNew, CriticNew
from algorithm_factory.algo_utils.rl_methods import print_result, process_single_state
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
        s_mission, s_station, s_cross, s_yard = state
        action = self.actor.forward(s_mission.to(self.device), s_station, s_cross, s_yard)
        if eval_tag:
            return action.detach().cpu()
        else:
            noise = (self.noise_std * torch.randn_like(action)).clamp(-self.noise_lim, self.noise_lim).to(self.device)
            action = (action + noise).clamp(0.0, 1.0)
            return action.detach().cpu()

    def update(self, batch: Dict[str, Any]):
        s_mission = batch['state_mission'].to(self.device)
        s_station = batch['state_station']
        s_cross = batch['state_cross']
        s_yard = batch['state_yard']
        u_action = batch['u_action'].unsqueeze(1).to(self.device)
        l_action = batch['l_action'].unsqueeze(1).to(self.device)
        s_mission_ = batch['state_mission_'].to(self.device)
        s_station_ = batch['state_station_']
        s_cross_ = batch['state_cross_']
        s_yard_ = batch['state_yard_']
        rewards = batch['reward'].unsqueeze(1).to(self.device)
        done = batch['done'].unsqueeze(1).to(self.device)

        # q_eval_value = self.qf.forward(s_mission, s_station, s_cross, s_yard)
        # q_next_value = self.qf_target.forward(s_mission_, s_station_, s_cross_, s_yard_)
        # q_eval = q_eval_value.gather(1, l_action)
        # q_next = q_next_value.gather(1, torch.max(q_next_value, 1)[1].unsqueeze(1))
        # q_target = rewards / 100.0 + self.gamma * q_next * (1.0 - done)
        # loss = self.loss_func(q_eval, q_target.detach())
        # # print(torch.mean(rewards)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # if self.train_count % 1 == 0:
        #     self.sync_weight()
        # self.train_count += 1

        u_a = self.actor(s_mission, s_station, s_cross, s_yard)
        n_u_a = self.actor(s_mission_, s_station_, s_cross_, s_yard_) #TODO

        # Compute the target Q value
        target_Q = self.critic(s_mission_, s_station_, s_cross_, s_yard_, n_u_a)
        target_Q = rewards / 100.0 + (done * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(s_mission, s_station, s_cross, s_yard, u_action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(s_mission, s_station, s_cross, s_yard, u_a).mean()  # Deterministic DPG 找Qmax 确定性策略
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

    def collect_rl(self):
        logger.info("收集RL交互数据")
        for solu in self.train_solus:
            self.pre_makespan = 0.0
            state = get_rnn_state_v2(solu.iter_env, 0, self.m_max_num)
            self.curr_time = [0, 0, 0]
            pre_makespan = 0
            for step in range(self.mission_num):
                cur_mission = solu.iter_env.mission_list[step]
                u_action = self.u_agent(process_single_state(state))
                l_action = self.l_agent.forward(process_single_state(state), False)
                makespan = self.process(solu, cur_mission, u_action, l_action, step, self.each_quay_m_num)
                reward = (pre_makespan - makespan)
                if step != self.mission_num - 1:
                    next_mission = solu.iter_env.mission_list[step + 1]
                    self.process_release_adjust(next_mission, self.each_quay_m_num)
                    new_state = get_rnn_state_v2(iter_solution=solu.iter_env, step_number=step + 1,
                                                 max_num=self.m_max_num)
                    done = 0
                else:
                    new_state = state
                    done = 1
                self.data_buffer.append(state=state, u_ac=u_action, l_ac=l_action, new_state=new_state, r=reward,
                                        done=done)
            solu.reset()

    def eval(self, eval_flag=False):
        with torch.no_grad():
            makespan_forall = []
            for i in range(len(self.test_solus)):
                solu = self.test_solus[i]
                state = get_rnn_state_v2(solu.iter_env, 0, self.m_max_num)
                self.curr_time = [0, 0, 0]
                for step in range(self.mission_num):
                    cur_mission = solu.iter_env.mission_list[step]
                    if eval_flag:
                        u_action = torch.tensor(90)
                    else:
                        u_action = self.u_agent.forward(process_single_state(state))
                    l_action = self.l_agent.forward(process_single_state(state), False)
                    reward = self.process(solu, cur_mission, u_action, l_action, step, self.each_quay_m_num)
                    if step != self.mission_num - 1:
                        next_mission = solu.iter_env.mission_list[step + 1]
                        self.process_release_adjust(next_mission, self.each_quay_m_num)
                        new_state = get_rnn_state_v2(iter_solution=solu.iter_env, step_number=step + 1,
                                                     max_num=self.m_max_num)
                        done = 0
                    else:
                        new_state = state
                        done = 1
                    self.data_buffer.append(state=state, u_ac=u_action, l_ac=l_action, new_state=new_state, r=reward,
                                            done=done)
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
        mission.machine_start_time[0] -= mission.release_time - adjust_time
        mission.machine_start_time[1] -= mission.release_time - adjust_time
        mission.total_process_time -= mission.release_time - adjust_time
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


def h_new_train(epoch_num: int, u_dl_train: DataLoader, u_agent: ACNew, l_agent: DDQN,
                u_collector: UANewCollector, rl_logger: SummaryWriter) -> None:
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
        with tqdm(u_dl_train, desc=f'epoch{epoch}', ncols=100) as pbar:
            total_loss = 0
            for batch in pbar:
                loss = l_agent.update(batch)  # s_mission, s_station, s_cross, s_yard, l_station, l_cross, l_yard
                total_loss += loss.data
            makespan = u_collector.eval()
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

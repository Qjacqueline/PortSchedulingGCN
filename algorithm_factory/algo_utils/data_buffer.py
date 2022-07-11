#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling_New
@File    ：u_data_buffer.py
@Author  ：JacQ
@Date    ：2022/4/21 10:28
"""
from typing import Tuple, Any, List, Optional

import torch

from torch.utils.data import Dataset
from torch_geometric.data import Dataset

from algorithm_factory.algo_utils.rl_methods import process_multi_sequence, process_single_sequence
from utils.log import Logger

logger = Logger().get_logger()


class LABuffer(Dataset):
    def __init__(self, buffer_size: int = 30000) -> None:
        super(LABuffer, self).__init__()
        logger.info("创建data buffer")
        self.state = []
        self.action = []
        self.state_ = []
        self.reward = []
        self.done = []
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, index: int) -> Tuple:
        return (self.state[index],
                self.action[index],
                self.state_[index],
                self.reward[index],
                self.done[index]
                )

    def append(self, state, ac, new_state, r, done):
        if len(self.done) == self.buffer_size:
            self.pop()
        self.state.append(state)
        self.action.append(ac)
        self.state_.append(new_state)
        self.reward.append(r)
        self.done.append(done)

    def pop(self):
        self.state.pop(0)
        self.action.pop(0)
        self.state_.pop(0)
        self.reward.pop(0)
        self.done.pop(0)


def la_collate_fn(raw_batch: List) -> Any:
    zip_raw_batch = zip(*raw_batch)
    return {'state': next(zip_raw_batch),
            'l_action': torch.tensor(next(zip_raw_batch), dtype=torch.int64),
            'state_': next(zip_raw_batch),
            'reward': torch.tensor(next(zip_raw_batch), dtype=torch.float32),
            'done': torch.tensor(next(zip_raw_batch), dtype=torch.float32)
            }


class UABuffer(Dataset):
    def __init__(self, buffer_size: int = 30000) -> None:
        super(UABuffer, self).__init__()
        logger.info("创建data buffer")
        self.state = []
        self.adjust = []

        self.action = []

        self.state_ = []

        self.reward = []
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, index: int) -> Tuple:
        return (self.state[index],
                self.adjust[index],
                self.action[index],
                self.state_[index],
                self.reward[index]
                )

    def append(self, state, ac, r, new_state, adjust):
        if len(self.action) == self.buffer_size:
            self.pop()
        self.state.append(state)
        self.adjust.append(adjust)
        self.action.append(ac)
        self.state_.append(new_state)
        self.reward.append(r)

    def pop(self):
        self.state.pop(0)
        self.adjust.pop(0)
        self.action.pop(0)
        self.state_.pop(0)
        self.reward.pop(0)

    def clear(self):
        self.state = []
        self.adjust = []
        self.action = []
        self.state_ = []
        self.reward = []


def ua_collate_fn(raw_batch: List) -> Any:
    """

    :param raw_batch:
    :return:
    """
    zip_raw_batch = zip(*raw_batch)
    return {'u_state': torch.cat(next(zip_raw_batch), 0),
            'adjust': torch.cat(next(zip_raw_batch), 0).unsqueeze(1),
            'u_action': torch.cat(next(zip_raw_batch), 0),
            'state_': torch.cat(next(zip_raw_batch), 0),
            'reward': torch.tensor(next(zip_raw_batch), dtype=torch.float32).reshape(len(raw_batch), -1)
            }


class UANewBuffer(Dataset):
    def __init__(self, buffer_size: int = 30000) -> None:
        super(UANewBuffer, self).__init__()
        logger.info("创建data buffer")
        self.state_mission = []
        self.state_station = []
        self.state_cross = []
        self.state_yard = []

        self.u_action = []
        self.l_action = []

        self.state_mission_ = []
        self.state_station_ = []
        self.state_cross_ = []
        self.state_yard_ = []

        self.reward = []
        self.done = []
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.l_action)

    def __getitem__(self, index: int) -> Tuple:
        return (self.state_mission[index],
                self.state_station[index],
                self.state_cross[index],
                self.state_yard[index],

                self.u_action[index],
                self.l_action[index],

                self.state_mission_[index],
                self.state_station_[index],
                self.state_cross_[index],
                self.state_yard_[index],

                self.reward[index],
                self.done[index]
                )

    def append(self, state, u_ac, l_ac, new_state, r, done):
        if len(self.done) == self.buffer_size:
            self.pop()
        s_m, s_s, s_c, s_y = state
        s_m_, s_s_, s_c_, s_y_ = new_state
        self.state_mission.append(s_m)
        self.state_station.append(s_s)
        self.state_cross.append(s_c)
        self.state_yard.append(s_y)

        self.u_action.append(u_ac)
        self.l_action.append(l_ac)

        self.state_mission_.append(s_m_)
        self.state_station_.append(s_s_)
        self.state_cross_.append(s_c_)
        self.state_yard_.append(s_y_)

        self.reward.append(r)
        self.done.append(done)

    def pop(self):
        self.state_mission.pop(0)
        self.state_station.pop(0)
        self.state_cross.pop(0)
        self.state_yard.pop(0)

        self.u_action.pop(0)
        self.l_action.pop(0)

        self.state_mission_.pop(0)
        self.state_station_.pop(0)
        self.state_cross_.pop(0)
        self.state_yard_.pop(0)

        self.reward.pop(0)
        self.done.pop(0)

    def clear(self):
        self.state_mission = []
        self.state_station = []
        self.state_cross = []
        self.state_yard = []

        self.u_action = []
        self.l_action = []

        self.state_mission_ = []
        self.state_station_ = []
        self.state_cross_ = []
        self.state_yard_ = []

        self.reward = []
        self.done = []


def ua_new_collate_fn(raw_batch: List) -> Any:
    zip_raw_batch = zip(*raw_batch)
    return {'state_mission': torch.tensor(next(zip_raw_batch), dtype=torch.float32),
            'state_station': process_multi_sequence(next(zip_raw_batch)),
            'state_cross': process_single_sequence(torch.tensor(next(zip_raw_batch), dtype=torch.float32)),
            'state_yard': process_single_sequence(torch.tensor(next(zip_raw_batch), dtype=torch.float32)),
            'u_action': torch.tensor(next(zip_raw_batch), dtype=torch.int64),
            'l_action': torch.tensor(next(zip_raw_batch), dtype=torch.int64),
            'state_mission_': torch.tensor(next(zip_raw_batch), dtype=torch.float32),
            'state_station_': process_multi_sequence(next(zip_raw_batch)),
            'state_cross_': process_single_sequence(torch.tensor(next(zip_raw_batch), dtype=torch.float32)),
            'state_yard_': process_single_sequence(torch.tensor(next(zip_raw_batch), dtype=torch.float32)),
            'reward': torch.tensor(next(zip_raw_batch), dtype=torch.float32),
            'done': torch.tensor(next(zip_raw_batch), dtype=torch.float32)
            }


if __name__ == '__main__':
    a = [[[[0, 0, 0], [1, 1, 1], [0.5925, 0.5490, 0.3726]],
          [[0, 0, 0], [0.6694, 0.1828, 0.6905], [0.5925, 0.5490, 0.3726]]],
         [[[0, 0, 0], [1, 1, 1], [0.5925, 0.5490, 0.3726]],
          [[0, 0, 0], [0.6694, 0.1828, 0.6905], [0.5925, 0.5490, 0.3726]]]]
    pack, order_seq_l = process_multi_sequence(a)

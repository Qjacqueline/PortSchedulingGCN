#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：port_scheduling
@File    ：mission.py
@Author  ：JacQ
@Date    ：2021/12/1 16:53
"""


class Mission(object):

    def __init__(self, idx, quay_crane_id, yard_block_loc, yard_crane_process_time, locked,
                 release_time, vehicle_speed, station_process_time):
        """

        :param idx: 任务id
        :param quay_crane_id: 岸桥id
        :param yard_block_loc：堆存堆场的位置 ('A1',5,6) A1堆场x=5 y=6的slot
        :param yard_crane_process_time: 场桥所需处理时间
        :param locked: 是否上锁
        :param release_time: 任务释放时间（开始任务时间）
        :param machine_list: 每个工序分配的machine id ['BF2', 'a_exit', 'a_station', 'S3', 'a_crossover', 'CO2', 'a_yard', 'YCB3']
        :param machine_start_time: 每个工序开始时间
        :param machine_process_time: 每个工序处理所需时间
        :param vehicle_speed: 运载小车速度

        """

        self.idx = idx
        self.locked = locked
        self.vehicle_speed = vehicle_speed
        self.quay_crane_id = quay_crane_id
        self.release_time = release_time
        self.station_process_time = station_process_time
        self.yard_crane_process_time = yard_crane_process_time
        self.crossover_id = None
        self.yard_block_loc = yard_block_loc
        self.yard_stop_loc = []

        self.transfer_time_e2s_min = 0
        self.transfer_time_s2c_min = 0
        self.transfer_time_c2y = 0

        self.total_process_time = 0
        self.machine_list = []
        self.machine_start_time = []
        self.machine_process_time = []

        self.waiting_time = []
        self.process_time = []
        self.arriving_time = []
        self.stage = 1

    def cal_mission_attributes(self, buffer_flag=True):
        # station-crossover-yard-station_buffer
        if buffer_flag:
            self.waiting_time = [self.machine_process_time[2], self.machine_process_time[5],
                                 self.machine_process_time[7], self.machine_process_time[3]]
            self.process_time = [self.machine_process_time[4], self.machine_process_time[6],
                                 self.machine_process_time[8],
                                 self.machine_process_time[2] - self.machine_process_time[3]]
            self.arriving_time = [self.machine_start_time[2], self.machine_start_time[5], self.machine_start_time[7],
                                  self.machine_start_time[2]]

        else:
            self.waiting_time = [self.machine_process_time[2], self.machine_process_time[4],
                                 self.machine_process_time[6]]
            self.process_time = [self.machine_process_time[3], self.machine_process_time[5],
                                 self.machine_process_time[7]]
            self.arriving_time = [self.machine_start_time[2], self.machine_start_time[4], self.machine_start_time[6]]

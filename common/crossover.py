#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：port_scheduling
@File    ：crossover.py
@Author  ：JacQ
@Date    ：2021/12/1 16:53
"""
from common.machine import Machine


class Crossover(Machine):
    def __init__(self, crossover_id, location, yard_block_list, max_wait_time):
        super(Crossover, self).__init__(crossover_id, location)
        self.yard_crane_list = yard_block_list
        self.max_wait_time = max_wait_time

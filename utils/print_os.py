#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling
@File    ：print_os.py
@Author  ：JacQ
@Date    ：2022/1/5 18:48
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# a test for traverse directory

__author__ = 'AlbertS'

import os
import os.path


def dfs_showdir(path, depth):
    if depth == 0:
        print("root:[" + path + "]")

    for item in os.listdir(path):
        if '.git' not in item and item[-1] != 'c'and item[-1] != 'f':
            print("| " * depth + "|    " + item)

            newitem = path + '/' + item
            if os.path.isdir(newitem):
                dfs_showdir(newitem, depth + 1)


if __name__ == '__main__':
    dfs_showdir('..', 0)

# -*- coding: utf-8 -*-
# @Time    : 2022/7/27 11:14 AM
# @Author  : JacQ
# @File    : plot_dynamic_running.py
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from algorithm_factory.algo_utils import sort_missions
from algorithm_factory.algo_utils.machine_cal_methods import get_yard_cranes_set, process_init_solution_for_l2a, \
    get_cur_time_status
from algorithm_factory.algorithm_heuristic_rules import Random_Choice
from common import PortEnv
from data_process.input_process import read_input


def plot_dynamic_running(port_env: PortEnv):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 初始化画布
    fig = plt.figure()
    for i in range(20000):
        cur_time = 60 * i
        plt.xticks(np.arange(0, 100, 10))
        plt.yticks(np.arange(0, 380, 280))
        # plt.axis('off')
        plt.title("动态运作图")
        plt.xlabel("正在加工任务")
        # plt.axvline(0)
        qc_base = 10
        ls_base = 60
        co_base = 120
        yc_base = 170

        for j in range(len(port_env.quay_cranes)):
            plt.text(x=-5, y=qc_base + 10 * j, s="QC" + str(j + 1))
        plt.text(x=-8, y=45, s="QC2LS")

        for j in range(len(port_env.lock_stations)):
            plt.text(x=-5, y=ls_base + 10 * j, s="LS" + str(j + 1))
        plt.text(x=-8, y=105, s="LS2CO")

        for j in range(len(port_env.crossovers)):
            plt.text(x=-5, y=co_base + 10 * j, s="CO" + str(j + 1))
        plt.text(x=-8, y=155, s="CO2YC")
        yard_cranes_set = get_yard_cranes_set(port_env)

        for j in range(len(yard_cranes_set)):
            plt.text(x=-5, y=yc_base + 10 * j, s=yard_cranes_set[j])
        f_base = len(yard_cranes_set) * 10 + 5 + yc_base
        plt.text(x=-8, y=f_base, s="finish")
        # fig.canvas.mpl_connect('button_press_event', onClick)
        update(port_env, cur_time)
        plt.pause(1)
        plt.cla()


def update(port_env: PortEnv, cur_time=1000):
    qc_ls, qc_ls_ls, ls_ls, ls_co_ls, co_ls, co_yc_ls, yc_ls, f_ls = get_cur_time_status(port_env, cur_time)
    yard_cranes_set = get_yard_cranes_set(port_env)
    qc_base = 10
    ls_base = 60
    co_base = 120
    yc_base = 170
    f_base = len(yard_cranes_set) * 10 + 5 + yc_base
    plt.text(x=60, y=220, s="当前时刻为：" + str(cur_time), fontsize=9, color='red')
    # 画当前的图
    # qc
    i = 0
    for qc_mission_ls in qc_ls.values():
        tmp_qc_base = qc_base + 10 * i
        j = 0
        for mission in qc_mission_ls:
            m_base = 0 + j * 4
            plt.gca().add_patch(plt.Rectangle((m_base, tmp_qc_base), 3.5, 4, facecolor='lightgrey'))
            plt.text(m_base, tmp_qc_base, mission.idx[1:], fontsize=7)
            j = j + 1
        i = i + 1
    # qc_ls
    j = 0
    for mission in qc_ls_ls:
        m_base = 0 + j * 4
        plt.gca().add_patch(plt.Rectangle((m_base, ls_base - 15), 3.5, 4, facecolor='lightgrey'))
        plt.text(m_base, ls_base - 15, mission.idx[1:], fontsize=7)
        j = j + 1
    # ls
    i = 0
    for ls_mission_ls in ls_ls.values():
        tmp_qc_base = ls_base + 10 * i
        j = 0
        getattr(sort_missions, "A_STATION_NB")(ls_mission_ls)
        for mission in ls_mission_ls:
            m_base = 0 + j * 4
            plt.gca().add_patch(plt.Rectangle((m_base, tmp_qc_base), 3.5, 4, facecolor='lightgrey'))
            plt.text(m_base, tmp_qc_base, mission.idx[1:], fontsize=7)
            j = j + 1
        i = i + 1
    # ls_co
    j = 0
    for mission in ls_co_ls:
        m_base = 0 + j * 4
        plt.gca().add_patch(plt.Rectangle((m_base, co_base - 15), 3.5, 4, facecolor='lightgrey'))
        plt.text(m_base, co_base - 15, mission.idx[1:], fontsize=7)
        j = j + 1
    # co
    i = 0
    for co_mission_ls in co_ls.values():
        tmp_qc_base = co_base + 10 * i
        j = 0
        getattr(sort_missions, "A_CROSSOVER_NB")(co_mission_ls)
        for mission in co_mission_ls:
            m_base = 0 + j * 4
            plt.gca().add_patch(plt.Rectangle((m_base, tmp_qc_base), 3.5, 4, facecolor='lightgrey'))
            plt.text(m_base, tmp_qc_base, mission.idx[1:], fontsize=7)
            j = j + 1
        i = i + 1
    # co_yc
    j = 0
    for mission in co_yc_ls:
        m_base = 0 + j * 4
        plt.gca().add_patch(plt.Rectangle((m_base, yc_base - 15), 3.5, 4, facecolor='lightgrey'))
        plt.text(m_base, yc_base - 15, mission.idx[1:], fontsize=7)
        j = j + 1
    # yc
    i = 0
    for yc_mission_ls in yc_ls.values():
        tmp_yc_base = yc_base + 10 * i
        j = 0
        getattr(sort_missions, "A_YARD_NB")(yc_mission_ls)
        for mission in yc_mission_ls:
            m_base = 0 + j * 4
            plt.gca().add_patch(plt.Rectangle((m_base, tmp_yc_base), 3.5, 4, facecolor='lightgrey'))
            plt.text(m_base, tmp_yc_base, mission.idx[1:], fontsize=7)
            j = j + 1
        i = i + 1
    # finish
    j = 0
    if f_ls is None:
        return
    if any(f_ls):
        f_ls.reverse()
    for mission in f_ls:
        m_base = 0 + j * 4
        plt.gca().add_patch(plt.Rectangle((m_base, f_base), 3.5, 4, facecolor='lightgrey'))
        plt.text(m_base, f_base, mission.idx[1:], fontsize=7)
        j = j + 1
        if j >= 10:
            break


if __name__ == '__main__':
    random.seed(2)
    instance = read_input('train_0_')
    _, solution, _ = Random_Choice(instance.init_env)
    plot_dynamic_running(solution)

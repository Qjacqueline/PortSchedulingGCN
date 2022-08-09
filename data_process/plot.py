# -*- coding: utf-8 -*-
# @Time    : 2022/7/27 4:01 PM
# @Author  : JacQ
# @File    : plot.py
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation

from algorithm_factory.algo_utils.machine_cal_methods import get_yard_cranes_set, process_init_solution_for_l2a
from algorithm_factory.algorithm_heuristic_rules import Random_Choice
from common import PortEnv
from data_process.input_process import read_input

instance = read_input('train_0_')
_, port_env, _ = Random_Choice(instance.init_env)


def plot_running():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 初始化画布
    fig = plt.figure()
    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 380, 240))
    # plt.axis('off')
    plt.title("动态运作图")
    plt.xlabel("正在加工任务")
    # plt.axvline(0)
    qc_base = 10
    ls_base = 60
    co_base = 120
    yc_base = 170

    for i in range(len(port_env.quay_cranes)):
        plt.text(x=-5, y=qc_base + 10 * i, s="QC" + str(i + 1))
    plt.text(x=-8, y=45, s="QC2LS")

    for i in range(len(port_env.lock_stations)):
        plt.text(x=-5, y=ls_base + 10 * i, s="LS" + str(i + 1))
    plt.text(x=-8, y=105, s="LS2CO")

    for i in range(len(port_env.crossovers)):
        plt.text(x=-5, y=co_base + 10 * i, s="CO" + str(i + 1))
    plt.text(x=-8, y=155, s="CO2YC")
    yard_cranes_set = get_yard_cranes_set(port_env)

    for i in range(len(yard_cranes_set)):
        plt.text(x=-5, y=yc_base + 10 * i, s=yard_cranes_set[i])
    f_base = len(yard_cranes_set) * 10 + 5 + yc_base
    plt.text(x=-8, y=f_base, s="finish")
    ani = animation.FuncAnimation(
        fig=fig, func=func, interval=0.1, blit=False)
    plt.show()


def func(cur_time):
    qc_ls, qc_ls_ls, ls_ls, ls_co_ls, co_ls, co_yc_ls, yc_ls, f_ls = cal(cur_time)
    yard_cranes_set = get_yard_cranes_set(port_env)
    # 画当前的图
    qc_base = 10
    ls_base = 60
    co_base = 120
    yc_base = 170
    f_base = len(yard_cranes_set) * 10 + 5 + yc_base
    re_tuple = []
    # qc
    i = 0
    for qc_mission_ls in qc_ls.values():
        tmp_qc_base = qc_base + 10 * i
        j = 0
        for mission in qc_mission_ls:
            m_base = 0 + j * 4
            re_tuple.append(plt.gca().add_patch(plt.Rectangle((m_base, tmp_qc_base), 3.5, 4, facecolor='lightgrey')))
            re_tuple.append(plt.text(m_base, tmp_qc_base, mission.idx[1:], fontsize=7))
            j = j + 1
        i = i + 1
    # qc_ls
    j = 0
    for mission in qc_ls_ls:
        m_base = 0 + j * 4
        re_tuple.append(plt.gca().add_patch(plt.Rectangle((m_base, ls_base - 15), 3.5, 4, facecolor='lightgrey')))
        re_tuple.append(plt.text(m_base, ls_base - 15, mission.idx[1:], fontsize=7))
        j = j + 1
    # ls
    i = 0
    for qc_mission_ls in ls_ls.values():
        tmp_qc_base = ls_base + 10 * i
        j = 0
        for mission in qc_mission_ls:
            m_base = 0 + j * 4
            re_tuple.append(plt.gca().add_patch(plt.Rectangle((m_base, tmp_qc_base), 3.5, 4, facecolor='lightgrey')))
            re_tuple.append(plt.text(m_base, tmp_qc_base, mission.idx[1:], fontsize=7))
            j = j + 1
        i = i + 1
    # ls_co
    j = 0
    for mission in ls_co_ls:
        m_base = 0 + j * 4
        re_tuple.append(plt.gca().add_patch(plt.Rectangle((m_base, co_base - 15), 3.5, 4, facecolor='lightgrey')))
        re_tuple.append(plt.text(m_base, co_base - 15, mission.idx[1:], fontsize=7))
        j = j + 1
    # co
    i = 0
    for qc_mission_ls in co_ls.values():
        tmp_qc_base = co_base + 10 * i
        j = 0
        for mission in qc_mission_ls:
            m_base = 0 + j * 4
            re_tuple.append(plt.gca().add_patch(plt.Rectangle((m_base, tmp_qc_base), 3.5, 4, facecolor='lightgrey')))
            re_tuple.append(plt.text(m_base, tmp_qc_base, mission.idx[1:], fontsize=7))
            j = j + 1
        i = i + 1
    # co_yc
    j = 0
    for mission in co_yc_ls:
        m_base = 0 + j * 4
        re_tuple.append(plt.gca().add_patch(plt.Rectangle((m_base, yc_base - 15), 3.5, 4, facecolor='lightgrey')))
        re_tuple.append(plt.text(m_base, yc_base - 15, mission.idx[1:], fontsize=7))
        j = j + 1
    # yc
    i = 0
    for qc_mission_ls in yc_ls.values():
        tmp_qc_base = yc_base + 10 * i
        j = 0
        for mission in qc_mission_ls:
            m_base = 0 + j * 4
            re_tuple.append(plt.gca().add_patch(plt.Rectangle((m_base, tmp_qc_base), 3.5, 4, facecolor='lightgrey')))
            re_tuple.append(plt.text(m_base, tmp_qc_base, mission.idx[1:], fontsize=7))
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
        re_tuple.append(plt.gca().add_patch(plt.Rectangle((m_base, f_base), 3.5, 4, facecolor='lightgrey')))
        re_tuple.append(plt.text(m_base, f_base, mission.idx[1:], fontsize=7))
        j = j + 1
    re_tuple.append(plt.text(x=60, y=220, s="当前时刻为：" + str(cur_time), fontsize=9, color='red'))
    return re_tuple


def cal(cur_time):
    yard_cranes_set = get_yard_cranes_set(port_env)
    qc_ls = {'QC1': [], 'QC2': [], 'QC3': []}
    qc_ls_ls = []
    ls_ls = {'S1': [], 'S2': [], 'S3': [], 'S4': []}
    ls_co_ls = []
    co_ls = {'CO1': [], 'CO2': [], 'CO3': []}
    co_yc_ls = []
    yc_ls = {}
    for key in yard_cranes_set:
        yc_ls.setdefault(key, [])
    f_ls = []
    for mission in port_env.mission_list:
        if mission.idx == 'M101' and cur_time == 960:
            a = 1
        if mission.machine_start_time[0] > cur_time:
            qc_ls[mission.quay_crane_id].append(mission)
        elif mission.machine_start_time[2] > cur_time:
            qc_ls_ls.append(mission)
        elif mission.machine_start_time[2] + mission.machine_process_time[2] > cur_time:
            ls_ls[mission.machine_list[4]].append(mission)
        elif mission.machine_start_time[5] > cur_time:
            ls_co_ls.append(mission)
        elif mission.machine_start_time[5] + mission.machine_process_time[5] + mission.machine_process_time[
            6] > cur_time:
            co_ls[mission.machine_list[6]].append(mission)
        elif mission.machine_start_time[7] > cur_time:
            co_yc_ls.append(mission)
        elif mission.total_process_time + mission.release_time > cur_time:
            yc_ls[mission.machine_list[8][2:]].append(mission)
        else:
            f_ls.append(mission)
    return qc_ls, qc_ls_ls, ls_ls, ls_co_ls, co_ls, co_yc_ls, yc_ls, f_ls


plot_running()

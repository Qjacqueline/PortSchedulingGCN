#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：port_scheduling
@File    ：analyse_result.py
@Author  ：JacQ
@Date    ：2021/12/22 14:40
"""
import os
import seaborn as sns

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from algorithm_factory.algo_utils.missions_sort_rules import sort_missions
from algorithm_factory.algorithm_heuristic_rules import Random_Choice, Least_Mission_Num_Choice
from common import PortEnv
from data_process.input_process import read_input
from gurobi_solver import CongestionPortModel, solve_model
from utils.log import Logger
import conf.configs as cf

logger = Logger().get_logger()


def analyse_result(result, save_label):
    draw_gantt_graph_missions(result, save_label)
    # calculate_statistics(result)


def draw_gantt_graph_missions(port_env: PortEnv, save_label):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # add = [[14, 6, 23], [18, 7, 35], [10, 30, 21], [18, 5, 30], [10, 25, 20], [12, 35, 20], [10, 32, 17], [10, 20, 16]]
    # left = [[0, 14, 20], [14, 32, 43], [32, 42, 78], [42, 72, 99], [60, 77, 129], [70, 102, 149], [82, 137, 169],
    #         [92, 169, 189]]
    add = []
    left = []
    mission_name = []
    station = []
    crossover = []
    yardcrane = []
    getattr(sort_missions, 'RELEASE_ORDER')(port_env.mission_list)
    for mission in port_env.mission_list:
        process_time = mission.machine_process_time[0:3] + mission.machine_process_time[4:]
        start_time = mission.machine_start_time[0:3] + mission.machine_start_time[4:]
        add.append(process_time)
        left.append(start_time)
        mission_name.append(mission.idx)
        station.append(mission.machine_list[4])
        crossover.append(mission.machine_list[6])
        yardcrane.append(mission.machine_list[8])
    m = range(len(add))
    n = range(len(add[0]))
    color = ['dodgerblue', 'w', 'lightgrey', 'g', 'grey', 'r', 'grey', 'gold', 'c', 'm', 'k']

    # 画布设置，大小与分辨率
    plt.figure()  #
    # barh-柱状图换向，循坏迭代-层叠效果
    for i in m:
        for j in n:

            if j is 0:
                plt.text(left[i][j] - 18, m[i] + 0.8, mission_name[i], fontsize=9, color='r')
                plt.barh(m[i] + 1, add[i][j], left=left[i][j], color=color[j])
            elif j is 3:
                if station[i] == 'S1':
                    plt.text(left[i][j] + 6, m[i] + 0.8, station[i], fontsize=9, color='black')
                    plt.barh(m[i] + 1, add[i][j], left=left[i][j], color='green')
                if station[i] == 'S2':
                    plt.text(left[i][j] + 6, m[i] + 0.8, station[i], fontsize=9, color='black')
                    plt.barh(m[i] + 1, add[i][j], left=left[i][j], color='yellowgreen')
                if station[i] == 'S3':
                    plt.text(left[i][j] + 6, m[i] + 0.8, station[i], fontsize=9, color='black')
                    plt.barh(m[i] + 1, add[i][j], left=left[i][j], color='seagreen')
                if station[i] == 'S4':
                    plt.text(left[i][j] + 6, m[i] + 0.8, station[i], fontsize=9, color='black')
                    plt.barh(m[i] + 1, add[i][j], left=left[i][j], color='mediumseagreen')

            elif j is 5:
                plt.text(left[i][j] - 4, m[i] + 0.8, crossover[i], fontsize=9, color='black')
                plt.barh(m[i] + 1, add[i][j], left=left[i][j], color=color[j])
            elif j is 7:
                plt.text(left[i][j] + 10, m[i] + 0.8, yardcrane[i], fontsize=9, color='black')
                plt.barh(m[i] + 1, add[i][j], left=left[i][j], color='gold')
            else:
                plt.barh(m[i] + 1, add[i][j], left=left[i][j], color=color[j])

    plt.title("流水加工甘特图" + save_label)
    labels = [''] * len(add[0])
    labels = ['QC', 'a_exit', 'a_station', 'LockStation', 'travel_time', 'Crossover', 'travel_time', 'YardCrane']
    # for f in n:
    #     labels[f] = "工序%d" % (f + 1)
    # 图例绘制
    patches = [mpatches.Patch(color=color[i % 10], label="{:s}".format(labels[i])) for i in range(len(add[0]))]
    plt.legend(handles=patches, loc=4)
    # XY轴标签
    plt.xlabel("加工时间/s")
    plt.ylabel("集装箱下发顺序")
    # 网格线，此图使用不好看，注释掉
    # plt.grid(linestyle="--",alpha=0.5)
    plt.savefig(
        os.path.join(cf.OUTPUT_RESULT_PATH, save_label + '_gantt_' + str(port_env.J_num_all) + '.png'))
    plt.show()


def draw_gantt_graph_missions_exact(port_env, MLP, save_label):
    MLP_vars = MLP.getVars()
    MLP_vars_dict = {}
    for var in MLP_vars:
        if var.VarName[0] == 'r' or var.VarName[0] == 'C' or var.VarName[0] == 'u' or var.VarName[0] == 'o':
            MLP_vars_dict[var.VarName] = var.X

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签port_env,
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    add = []
    left = []
    mission_name = []
    quaycrane = []
    station = []
    crossover = []
    yardcrane = []
    getattr(sort_missions, 'RELEASE_ORDER')(port_env.mission_list)
    for mission in port_env.mission_list:
        process_time_b = mission.machine_process_time[0:3] + mission.machine_process_time[4:]
        start_time_b = mission.machine_start_time[0:3] + mission.machine_start_time[4:]
        process_time, start_time = process_time_b, start_time_b
        # qc
        start_time[0] = MLP_vars_dict['r_' + str(int(mission.idx[1:]) - 1)]
        start_time[1] = start_time_b[1] - start_time_b[0] + start_time[0]
        # ls
        start_time[2] = MLP_vars_dict['C_0_' + str(int(mission.idx[1:]) - 1)] + \
                        MLP_vars_dict['u_0_' + str(int(mission.idx[1:]) - 1)]
        start_time[3] = MLP_vars_dict['C_1_' + str(int(mission.idx[1:]) - 1)] - \
                        MLP_vars_dict['o_1_' + str(int(mission.idx[1:]) - 1)]
        process_time[2] = start_time[3] - start_time[2]
        process_time[3] = MLP_vars_dict['o_1_' + str(int(mission.idx[1:]) - 1)]
        # co
        start_time[4] = MLP_vars_dict['C_1_' + str(int(mission.idx[1:]) - 1)] + \
                        MLP_vars_dict['u_1_' + str(int(mission.idx[1:]) - 1)]
        start_time[5] = MLP_vars_dict['C_2_' + str(int(mission.idx[1:]) - 1)] - \
                        MLP_vars_dict['o_2_' + str(int(mission.idx[1:]) - 1)]
        process_time[4] = start_time[5] - start_time[4]
        process_time[5] = MLP_vars_dict['o_2_' + str(int(mission.idx[1:]) - 1)]
        # yc
        start_time[6] = MLP_vars_dict['C_2_' + str(int(mission.idx[1:]) - 1)] + \
                        MLP_vars_dict['u_2_' + str(int(mission.idx[1:]) - 1)]
        start_time[7] = MLP_vars_dict['C_3_' + str(int(mission.idx[1:]) - 1)] - \
                        MLP_vars_dict['o_3_' + str(int(mission.idx[1:]) - 1)]
        process_time[6] = start_time[7] - start_time[6]
        process_time[7] = MLP_vars_dict['o_3_' + str(int(mission.idx[1:]) - 1)]

        add.append(process_time)
        left.append(start_time)
        mission_name.append(mission.idx)
        quaycrane.append(mission.quay_crane_id)
        station.append(mission.machine_list[4])
        crossover.append(mission.machine_list[6])
        yardcrane.append(mission.machine_list[8])
    m = range(len(add))
    n = range(len(add[0]))
    color = ['dodgerblue', 'w', 'lightgrey', 'g', 'grey', 'r', 'grey', 'gold', 'c', 'm', 'k']
    sc = sns.color_palette("Spectral", 10)
    qcc = sns.color_palette("Reds", 10)
    lsc = sns.color_palette("YlOrRd", 10)
    isc = sns.color_palette("YlGn", 10)
    ycc = sns.color_palette("PuBu", 10)
    qcc.reverse(), lsc.reverse(), isc.reverse(), ycc.reverse()
    color = [sc[0], 'w', 'lightgrey', sc[3], 'silver', sc[-3], 'darkgrey', sc[-1], 'c', 'm', 'k']
    # 画布设置，大小与分辨率
    plt.figure()  #
    # barh-柱状图换向，循坏迭代-层叠效果
    for i in m:
        for j in n:
            if j is 0:
                plt.text(left[i][j] - 18, m[i] + 0.8, mission_name[i], fontsize=9, color='black')
                plt.barh(m[i] + 1, add[i][j], left=left[i][j], color=qcc[int(quaycrane[i][2:]) - 1])
            elif j is 3:
                plt.text(left[i][j] + 6, m[i] + 0.8, station[i], fontsize=9, color='black')
                plt.barh(m[i] + 1, add[i][j], left=left[i][j], color=lsc[int(station[i][1:]) + 3])
            elif j is 5:
                plt.text(left[i][j] - 4, m[i] + 0.8, 'IS' + crossover[i][-1], fontsize=9, color='black')
                plt.barh(m[i] + 1, add[i][j], left=left[i][j], color=isc[int(crossover[i][2:]) * 2 - 1])
            elif j is 7:
                plt.text(left[i][j] + 10, m[i] + 0.8, yardcrane[i], fontsize=9, color='black')
                plt.barh(m[i] + 1, add[i][j], left=left[i][j],
                         color=ycc[(4 * (ord(yardcrane[i][2]) - 65) + int(yardcrane[i][3])) % 9 + 1])
            else:
                plt.barh(m[i] + 1, add[i][j], left=left[i][j], color=color[j])

    plt.title("Gantt chart for port scheduling" + save_label)
    labels = ['QC_process', 'travel', 'LS_wait', 'LS_process', 'IS_wait', 'IS_process', 'YC_wait', 'YC_process']

    # 图例绘制
    patches = [mpatches.Patch(color=color[i % 10], label="{:s}".format(labels[i])) for i in [0, 2, 3, 4, 5, 6, 7, 1]]
    plt.legend(handles=patches, loc=4)
    # XY轴标签
    plt.xlabel("time/s")
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.xlim(0, 2600)
    # plt.ylabel("集装箱下发顺序")
    # 网格线，此图使用不好看，注释掉
    # plt.grid(linestyle="--",alpha=0.5)
    plt.savefig(
        os.path.join(cf.OUTPUT_RESULT_PATH, save_label + '_gantt_' + str(port_env.J_num_all) + '.png'))
    plt.show()


def calculate_statistics_all(ls):
    with open(os.path.join(cf.OUTPUT_RESULT_PATH, 'summary_all.txt'),
              "w", encoding="utf-8") as f:
        f.write("统计结果汇总\n")
        out_r = []
        for i in ls:
            instance = read_input('train', cf.MISSION_NUM, i)
            _, port_env, _ = Least_Mission_Num_Choice(instance.init_env)
            max_makespan = port_env.cal_finish_time()
            machines = {}
            machines.update(port_env.lock_stations)
            machines.update(port_env.crossovers)
            machines.update(port_env.yard_cranes)
            wait_time_station = [0] * len(port_env.lock_stations)
            wait_time_crossover = [0] * len(port_env.crossovers)
            wait_time_yard_crane = [0] * len(port_env.yard_cranes)
            for mission in port_env.mission_list:
                wait_time_station[int(mission.machine_list[4][-1]) - 1] += mission.machine_process_time[4]
                wait_time_crossover[int(mission.machine_list[6][-1]) - 1] += mission.machine_process_time[5]
                wait_time_yard_crane[int(mission.machine_list[8][-1]) - 1] += mission.machine_process_time[7]
            out_r.append(str(i))
            out_r.append("makespan：")
            out_r.append(str(max_makespan))
            out_r.append("LS：")
            out_r.append(str(round(sum(wait_time_station), 2)))
            out_r.append("CO：")
            out_r.append(str(round(sum(wait_time_crossover), 2)))
            out_r.append("YC：")
            out_r.append(str(round(sum(wait_time_yard_crane), 2)))
            out_r.append('\n')
        lists = [line + "\t" for line in out_r]
        f.writelines(lists)


def calculate_statistics(port_env: PortEnv):
    with open(os.path.join(cf.OUTPUT_RESULT_PATH, 'summary_' + cf.inst_type + '_' + str(port_env.J_num_all) + '.txt'),
              "w", encoding="utf-8") as f:
        f.write("统计结果汇总\n")
        machines = {}
        machines.update(port_env.lock_stations)
        machines.update(port_env.crossovers)
        machines.update(port_env.yard_cranes)
        f.write("******闲置时间汇总********\n")
        for machine_id, machine in machines.items():
            idle_time = 0.0
            if len(machine.process_time) == 0:
                f.write(machine_id + "的闲置时间为：未使用\n")
            else:
                for i in range(len(machine.process_time)):
                    if i != len(machine.process_time) - 1:
                        idle_time += machine.process_time[i + 1][0] - machine.process_time[i][2]
                f.write(machine_id + "的闲置时间为：" + str(round(idle_time, 2)) + "\n")

        f.write("******等待时间汇总********\n")
        wait_time_station = [0] * len(port_env.lock_stations)
        wait_time_crossover = [0] * len(port_env.crossovers)
        wait_time_yard_crane = [0] * len(port_env.yard_cranes)
        wait_time_station_total = 0
        wait_time_crossover_total = 0
        wait_time_yard_crane_total = 0
        for mission in port_env.mission_list:
            wait_time_station[int(mission.machine_list[4][-1]) - 1] += mission.machine_process_time[4]
            wait_time_crossover[int(mission.machine_list[6][-1]) - 1] += mission.machine_process_time[5]
            wait_time_yard_crane[int(mission.machine_list[8][-1]) - 1] += mission.machine_process_time[7]
        for i in range(len(wait_time_station)):
            wait_time_station_total += wait_time_station[i]
            f.write('S' + str(i + 1) + "的等待时间为：" + str(round(wait_time_station[i], 2)) + "\n")
        for i in range(len(wait_time_crossover)):
            wait_time_crossover_total += wait_time_crossover[i]
            f.write('CO' + str(i + 1) + "的等待时间为：" + str(round(wait_time_crossover[i], 2)) + "\n")
        for i in range(len(wait_time_yard_crane)):
            wait_time_yard_crane_total += wait_time_yard_crane[i]
            f.write('YC' + str(i + 1) + "的等待时间为：" + str(round(wait_time_yard_crane[i], 2)) + "\n")
        f.write("LockStation的总等待时间为：" + str(round(wait_time_station_total, 2)) + "\n")
        f.write("Crossover的总等待时间为：" + str(round(wait_time_crossover_total, 2)) + "\n")
        f.write("YardCrane的总等待时间为：" + str(round(wait_time_yard_crane_total, 2)) + "\n")
        max_makespan = port_env.cal_finish_time()
        f.write("最大makespan为：" + str(max_makespan) + "\n")
        logger.info("LockStation的总等待时间为：" + str(round(wait_time_station_total, 2)))
        logger.info("Crossover的总等待时间为：" + str(round(wait_time_crossover_total, 2)))
        logger.info("YardCrane的总等待时间为：" + str(round(wait_time_yard_crane_total, 2)))
        # rl_logger.info("最小makespan为：" + str(min(port_env.makespan_history)) + "\n")


if __name__ == '__main__':
    port_env = read_input('train', cf.MISSION_NUM, cf.inst_type)
    _, solu, _ = Least_Mission_Num_Choice(port_env.init_env)
    port_env.init_env, port_env.iter_env = solu, solu
    model = CongestionPortModel(port_env)
    model.construct()
    solve_model(MLP=model.MLP, inst_idx=cf.inst_type, solved_solu=port_env, tag='_fix_all')
    draw_gantt_graph_missions_exact(solu, model.MLP, 'w')
    # analyse_result(solu, '1')
    # calculate_statistics_all(['A_t', 'B_t', 'C_t', 'D_t', 'E_t', 'F_t', 'G_t', 'H_t', 'Z_t'])
    a = 1

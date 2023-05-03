# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 11:12 AM
# @Author  : JacQ
# @File    : branch_and_bound.py
import time

import torch
from gurobipy import *
import numpy as np
import copy

import queue
import conf.configs as cf
from algorithm_factory.algo_utils.machine_cal_methods import get_state_n, find_min_wait_station, \
    find_least_distance_station
from algorithm_factory.algo_utils.net_models import QNet
from algorithm_factory.rl_algo.lower_agent import DDQN
from common import PortEnv
from data_process.input_process import read_input
from gurobi_solver import RelaxedCongestionPortModel

eps = 1e-10


# Node class
class Node:
    # this class define the node
    idx = -1  # 对应job index
    model = None
    solu = None

    cnt = 0
    local_LB = 0
    local_UB = np.inf
    is_integer = False
    x_sol = {}
    x_int_sol = {}
    visited = []


def deepcopy_node(node):
    new_node = Node()
    new_node.local_LB = node.local_LB
    new_node.local_UB = node.local_UB
    new_node.model = node.model.copy()
    new_node.idx = node.idx
    new_node.visited = copy.deepcopy(node.visited)
    new_node.x_sol = copy.deepcopy(node.x_sol)
    new_node.x_int_sol = copy.deepcopy(node.x_int_sol)
    new_node.solu = copy.deepcopy(node.solu)
    return new_node


# B&B | depth first | binary branching
# def BB_depth_binary(RLP, solu, J_num, global_UB):
#     # 固定xij
#     vars = RLP.getVars()
#     ls = solu.iter_env.mission_list
#     for i in range(len(solu.iter_env.mission_list)):
#         var_idx = (int(ls[i].idx[1:]) - 1) * 22 + int(ls[i].machine_list[4][-1]) + 2
#         RLP.addConstr((vars[var_idx] == 1), "fixed_x" + str(i))
#     RLP.update()
#     RLP.optimize()
#     var_ls = [3 + i * 22 + (J_num + 2) * 22 for i in range(0, (J_num + 2) * (J_num + 2))]
#     var_ls.extend([4 + i * 22 + (J_num + 2) * 22 for i in range(0, (J_num + 2) * (J_num + 2))])
#     var_ls.extend([5 + i * 22 + (J_num + 2) * 22 for i in range(0, (J_num + 2) * (J_num + 2))])
#     var_ls.extend([6 + i * 22 + (J_num + 2) * 22 for i in range(0, (J_num + 2) * (J_num + 2))])
#
#     # 初始化参数
#     global_UB = global_UB
#     global_LB = RLP.ObjVal
#     eps = 1e-10
#     incumbent_node = None
#     branch_var_name = 'None'
#
#     # 初始化node
#     Queue = []
#     node = Node()
#     node.local_UB = float('INF')
#     node.local_LB = RLP.ObjVal
#     node.model = RLP.copy()
#     node.model.setParam("OutputFlag", 0)
#     Queue.append(node)  # start with only initial node
#     Global_UB_change, Global_LB_change = [], []
#
#     cnt = 0
#     cnt_t = 0
#     while len(Queue) > 0 and global_UB - global_LB > eps:
#         # select the current node
#         current_node = Queue.pop()
#         current_node.model.optimize()
#         '''
#         OPTIMAL = 2
#         INFEASIBLE = 3
#         UNBOUND = 5
#         '''
#
#         # check whether the current solution is integer
#         is_integer = True
#         is_Pruned = False
#
#         print('\n --------------- \n', cnt_t, '\t', current_node.cnt, '\t Solution_status = ',
#               current_node.model.Status, "\t branch_on：", branch_var_name)
#         cnt_t = cnt_t + 1
#         if current_node.cnt == 80:
#             a = 1
#         # bound step 模型有解
#         if current_node.model.Status == 2:
#             for var in current_node.model.getVars():
#                 current_node.x_sol[var.varName] = var.x
#                 current_node.x_int_sol[var.varName] = int(var.x)  # round the solution to get an integer solution
#                 if var.varName[0] is 'Y' and abs(int(var.x) - var.x) >= eps:  # 判断解是否是整数,不是整数就分支
#                     is_integer = False
#                     current_node.branch_var_list.append(var.VarName)
#                     print(var.VarName, ' = ', var.x)
#
#             '''Update the UB'''
#             if is_integer:
#                 current_node.is_integer = True
#                 current_node.local_LB = current_node.model.ObjVal
#                 current_node.local_UB = current_node.model.ObjVal
#                 if current_node.local_UB < global_UB:
#                     global_UB = current_node.local_UB
#                     incumbent_node = deepcopy_node(current_node)
#             if not is_integer:
#                 current_node.is_integer = False
#                 current_node.local_LB = current_node.model.ObjVal
#                 # for var_name in current_node.x_int_sol.keys():
#                 #     var = current_node.model.getVarByName(var_name)
#                 #     current_node.local_LB += current_node.x_int_sol[var_name] * var.Obj
#                 # if current_node.local_LB > global_LB:
#                 #     global_LB = current_node.local_LB
#                 #     incumbent_node = deepcopy_node(current_node)
#
#             '''Pruning'''
#             # prune by optimality直接解出整数解（最优）
#             if is_integer:
#                 is_Pruned = True
#
#             # prune by bound
#             if not is_integer and current_node.local_LB > global_UB:  # 解出的解的下界大于当前全局上界
#                 is_Pruned = True
#
#             Gap = (global_UB - global_LB) / global_LB
#             print(current_node.cnt, '\t Gap = ', round(100 * Gap, 2), '%', '\n --------------- \n')
#
#         else:  # 没有解出LP最优解
#             print('\n --------------- \n')
#             is_integer = False
#
#             '''Pruning'''
#             # prune by infeasible
#             is_Pruned = True
#
#             continue
#
#         # branch step
#         if not is_Pruned:
#             # select the branch variable
#             branch_var_name = current_node.branch_var_list[0]  # 优先prune第一个
#
#             # create 2 child nodes
#             left_node = deepcopy_node(current_node)
#             right_node = deepcopy_node(current_node)
#
#             temp_var = left_node.model.getVarByName(branch_var_name)
#             left_node.model.addConstr(temp_var == 0, name='branch_left__' + str(cnt))
#             left_node.model.setParam('OutputFlag', 0)
#             left_node.model.update()
#             cnt += 1
#             left_node.cnt = cnt
#
#             temp_var = right_node.model.getVarByName(branch_var_name)
#             right_node.model.addConstr(temp_var == 1, name='branch_right__' + str(cnt))
#             right_node.model.setParam('OutputFlag', 0)
#             right_node.model.update()
#             cnt += 1
#             right_node.cnt = cnt
#
#             Queue.append(left_node)
#             Queue.append(right_node)
#
#             Global_UB_change.append(global_UB)
#             Global_LB_change.append(global_LB)
#
#     # global_UB = global_LB  # 所有节点探索完 更新全局bound 保证while的退出条件
#     Gap = round(100 * (global_UB - global_LB) / global_LB, 2)
#     Global_UB_change.append(global_UB)
#     Global_LB_change.append(global_LB)
#     print('____________________________________')
#     print('       Optimal solution found       ')
#     print('____________________________________')
#     print('queue_length:' + str(len(Queue)) + '\teps:' + str(global_UB - global_LB))
#     print('Obj:', global_LB)
#     print('Obj_LB:', Global_LB_change)
#     print('Obj_UB:', Global_UB_change)
#     print('Solution:', [list(incumbent_node.x_sol)[i] for i in var_ls if
#                         incumbent_node.x_sol[list(incumbent_node.x_sol)[i]] != 0])
#     return global_UB


# B&B | priority first | wide branching
def BB_priority_wide(label, RLP, solu, g_ub, g_lb):
    f = open("output_result/BB.txt", "a")

    # 初始化参数
    global_UB = g_ub  # OTOP
    global_LB = g_lb  # max(lb1,lb3)
    # global_UB = float("inf")  # OTOP
    # global_LB = 1  # max(lb1,lb3)
    cnt = 0  # 记录节点是树中的第几个节点
    cnt_t = 0  # 记录计算了几个节点
    cnt_prune = 0  # 记录prune了几次
    Global_UB_change, Global_LB_change = [], []

    # 初始化node
    Queue = queue.PriorityQueue()
    node = Node()
    node.local_UB = float("inf")
    node.local_LB = 0
    node.model = RLP.copy()
    node.model.setParam("OutputFlag", 0)
    node.idx = -1  # job 从0开始记
    node.solu = copy.deepcopy(solu)
    node.model.optimize()
    Queue.put((node.model.ObjVal, (cnt, node)))

    st = time.time()
    # 每个机器初始节点分支
    while not Queue.empty() and global_UB - global_LB > eps:
        # select the current node
        current_po = Queue.get()  # 取下界最小的，优先探索
        current_node = current_po[1][1]

        # print('\n --------------- \n', cnt_t, '\t', current_node.cnt, "\t visited：", current_node.visited)
        cnt_t += 1

        # calculate OTOP TODO
        # node.local_UB = OT(solu, current_node.index)

        # check whether the current solution is integer
        is_integer = check_integer(current_node)
        is_Pruned = False

        # update global bound
        if is_integer:  # 如果是整数解check UB
            if current_node.local_UB < global_UB:
                global_UB = current_node.local_UB
                incumbent_node = deepcopy_node(current_node)
        global_LB = current_node.local_LB if current_node.local_LB > global_LB else global_LB

        # 二次prune 1: by optimality直接解出整数解（最优）
        if is_integer:
            is_Pruned = True

        # 二次prune 2: by bound
        if not is_integer and current_node.local_LB > global_UB:  # 解出的解的下界大于当前全局上界
            is_Pruned = True

        if is_Pruned:
            print("****************prune****************" + str(current_node.visited))
            cnt_prune = cnt_prune + 1

        # 本轮process结束
        Gap = (global_UB - global_LB) / global_LB
        # print(current_node.cnt, '\t Gap = ', round(100 * Gap, 10), '%', '\n --------------- \n')
        Global_UB_change.append(global_UB)
        Global_LB_change.append(global_LB)

        if not is_Pruned and current_node.idx < solu.iter_env.J_num_all - 1:
            for ls in range(solu.init_env.ls_num):
                tmp_node = deepcopy_node(current_node)
                tmp_node.idx = tmp_node.idx + 1
                tmp_node.visited.append([tmp_node.idx, ls])
                tmp_node.solu.step_v2(ls, tmp_node.solu.iter_env.mission_list[tmp_node.idx], tmp_node.idx)
                var_idx = 'v_' + str(tmp_node.idx) + '_' + str(ls + solu.iter_env.qc_num)
                tmp_node.model.addConstr(tmp_node.model.getVarByName(var_idx) == 1, name='branch_' + str(var_idx))
                tmp_node.model.update()
                tmp_node.model.optimize()
                if tmp_node.model.Status == 2:
                    tmp_node.local_LB = tmp_node.model.ObjVal \
                        if is_integer or (tmp_node.model.ObjVal > tmp_node.local_LB) else tmp_node.local_LB
                    tmp_node.local_UB = OT(tmp_node.solu, tmp_node.idx)
                    # tmp_node.local_UB = cal_local_LB1(tmp_node.solu.iter_env, tmp_node.idx)
                    tmp_node.local_UB = tmp_node.model.ObjVal if is_integer else tmp_node.local_UB
                    cnt += 1
                    Queue.put((tmp_node.model.ObjVal, (cnt, tmp_node)))
        Global_UB_change.append(global_UB)
        Global_LB_change.append(global_LB)
    et = time.time()
    print('____________________________________')
    print('       Optimal solution found       ')
    print('____________________________________')
    print('queue_length:' + str(Queue.maxsize) + '\teps:' + str(global_UB - global_LB))
    print('Obj:', global_LB)
    print('Obj_LB:', Global_LB_change)
    print('Obj_UB:', Global_UB_change)
    # print('Solution:', [ for var in incumbent_node.model.getVars() if
    #                     incumbent_node.x_sol[list(incumbent_node.x_sol)[i]] != 0])
    f.write(label + "\ttime:\t" + str(et - st) + "\tobj:\t" + str(global_LB) +
            "\tcnt:\t" + str(cnt_t) + "\tcnt_prune:\t" + str(cnt_prune) + "\n")
    f.close()
    return global_UB


def check_prune(node: Node, to_visit_node: [], m: int):
    tmp_node = deepcopy_node(node)
    var_idx = 'Y_' + str(tmp_node.visited[-1][-1]) + '_' + str(to_visit_node[2]) + '_' + str(m + 3)
    tmp_node.model.addConstr(tmp_node.model.getVarByName(var_idx) == 1, name='branch_' + str(var_idx))
    tmp_node.model.update()
    tmp_node.model.optimize()

    # prune 2: 模型无解 OPTIMAL = 2 / INFEASIBLE = 3 / UNBOUND = 5
    if tmp_node.model.Status != 2:
        return None
    else:
        tmp_node.to_visited['S' + str(m + 1)].remove(to_visit_node)
        tmp_node.visited.append(to_visit_node)
        tmp_node.local_LB = tmp_node.model.ObjVal  # TODO
    return tmp_node


def check_integer(node: Node):
    is_integer = True
    for var in node.model.getVars():
        node.x_sol[var.varName] = var.x
        node.x_int_sol[var.varName] = int(var.x)  # round the solution to get an integer solution
        if var.varName[0] == 'v' and abs(int(var.x) - var.x) >= eps:  # 判断解是否是整数,不是整数就分支
            is_integer = False
            break
    node.is_integer = is_integer
    return is_integer


def OT(in_solu, step):
    temp_solu = copy.deepcopy(in_solu)
    for v_step in range(step + 1, temp_solu.init_env.J_num_all):
        temp_cur_mission = temp_solu.iter_env.mission_list[v_step]
        # state = get_state_n(env=temp_solu.iter_env, step_number=v_step, max_num=5)
        # action = agent.forward(state, False)
        action = int(find_min_wait_station(solu.iter_env, temp_cur_mission).idx[-1]) - 1
        temp_solu.step_v2(action, temp_cur_mission, v_step)
    return temp_solu.last_step_makespan


def OP1(in_solu):
    tempp_solu = copy.deepcopy(in_solu)
    min_makespan = float('Inf')
    for step in range(tempp_solu.init_env.J_num_all):
        cur_mission = tempp_solu.iter_env.mission_list[step]
        min_pos = -1
        for j in range(tempp_solu.init_env.ls_num):
            temp_solu = copy.deepcopy(tempp_solu)
            temp_cur_mission = temp_solu.iter_env.mission_list[step]
            temp_makespan = temp_solu.step_v2(j, temp_cur_mission, step)
            for v_step in range(step + 1, tempp_solu.init_env.J_num_all):
                temp_cur_mission = temp_solu.iter_env.mission_list[v_step]
                state = get_state_n(env=temp_solu.iter_env, step_number=v_step, max_num=5)
                action = agent.forward(state, False)
                temp_makespan = temp_solu.step_v2(action, temp_cur_mission, v_step)
                if temp_makespan > min_makespan:
                    break
            if temp_makespan <= min_makespan:
                min_makespan = temp_makespan
                min_pos = j
        makespan = tempp_solu.step_v2(min_pos, cur_mission, step)
    return makespan


# def cal_local_LB1(port_env: PortEnv, step):
#     m_m, m_2_3_4, m_ls, m_co, m_1_2_3, m_1, m_3_4, m_1_2, m_4 = float('inf'), float('inf'), float('inf'), float(
#         'inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')
#     lb_qc, lb_ls, lb_co, lb_yc = 0, 0, 0, 0
#     c_m_t_dict = {}  # 当前机器已分配的处理时间dict
#     c_ls_t, c_co_t = 0, 0  # 当前ls已分配处理时间加和
#     for ls in list(port_env.lock_stations.values()):
#         c_m_t_dict[ls.idx] = ls.whole_occupy_time[-1][-1]
#         c_ls_t += ls.whole_occupy_time[-1][-1]
#     for co in list(port_env.crossovers.values()):
#         c_m_t_dict[co.idx] = co.process_time[-1][-1]
#         c_co_t += co
#     p_ls_t = 0  # remain job 在ls所需时间
#     for mission in port_env.mission_list[step:]:
#         p_ls_t += mission.station_process_time
#     for mission in port_env.mission_list[step:]:S
#         _, min_distance = find_least_distance_station(port_env, mission)
#         lb_ls += mission.station_process_time
#         t_m_qc = mission.station_process_time + mission.intersection_process_time + \
#                  mission.yard_crane_process_time + \
#                  abs(cf.SLOT_NUM_Y - mission.yard_block_loc[2]) * cf.SLOT_WIDTH / cf.YARDCRANE_SPEED_Y * 2 + \
#                  min_distance / mission.vehicle_speed + mission.transfer_time_c2y
#         t_m_yc = mission.release_time / 2.0 + cf.BUFFER_PROCESS_TIME + mission.station_process_time + \
#                  mission.intersection_process_time + min_distance / mission.vehicle_speed + mission.transfer_time_c2y
#         t_m_1 = mission.release_time / 2.0 + cf.BUFFER_PROCESS_TIME + \
#                 port_env.quay_cranes[mission.quay_crane_id].time_to_exit + mission.transfer_time_e2s_min
#         t_m_3_4 = mission.intersection_process_time + \
#                   mission.yard_crane_process_time + \
#                   abs(cf.SLOT_NUM_Y - mission.yard_block_loc[2]) * cf.SLOT_WIDTH / cf.YARDCRANE_SPEED_Y * 2 + \
#                   mission.transfer_time_s2c_min + mission.transfer_time_c2y
#         t_m_1_2 = mission.release_time / 2.0 + cf.BUFFER_PROCESS_TIME + mission.station_process_time + \
#                   min_distance
#         t_m_4 = mission.yard_crane_process_time + \
#                 abs(cf.SLOT_NUM_Y - mission.yard_block_loc[2]) * cf.SLOT_WIDTH / cf.YARDCRANE_SPEED_Y * 2 + \
#                 mission.transfer_time_c2y
#         if t_m_qc < m_2_3_4:
#             m_2_3_4 = t_m_qc
#         if t_m_yc < m_1_2_3:
#             m_1_2_3 = t_m_yc
#         if t_m_1 < m_1:
#             m_1 = t_m_1
#         if t_m_3_4 < m_3_4:
#             m_3_4 = t_m_3_4
#         if t_m_1_2 < m_1_2:
#             m_1_2 = t_m_1_2
#         if t_m_4 < m_4:
#             m_4 = t_m_4
#         if mission.release_time / 2.0 + cf.BUFFER_PROCESS_TIME > lb_qc:
#             lb_qc = mission.release_time / 2.0 + cf.BUFFER_PROCESS_TIME
#     lb_ls = lb_ls / len(port_env.lock_stations)
#     for co in port_env.crossovers.values():
#         t_lb_co = 0
#         for mission in co.mission_list:
#             t_lb_co += mission.intersection_process_time
#         if t_lb_co > lb_co:
#             lb_co = t_lb_co
#     for yc_idx in port_env.yard_cranes_set:
#         yc = port_env.yard_cranes['YC' + yc_idx]
#         max_x_ls = [mission.yard_block_loc[1] for mission in yc.mission_list]
#         yc_pt_ls = [mission.yard_crane_process_time + abs(cf.SLOT_NUM_Y - mission.yard_block_loc[2])
#                     * cf.SLOT_WIDTH / cf.YARDCRANE_SPEED_Y * 2 for mission in yc.mission_list]
#         t_lb_yc = sum(yc_pt_ls) + max(max_x_ls) * cf.SLOT_LENGTH / cf.YARDCRANE_SPEED_X
#         if t_lb_yc > lb_yc:
#             lb_yc = t_lb_yc
#     lb1 = max((lb_qc + m_2_3_4), (lb_ls + m_1 + m_3_4), (m_co + m_1_2 + m_4), (m_1_2_3 + lb_yc))
#     # print("lB1:" + str(lb1))
#     return lb1


if __name__ == '__main__':

    m_num_ls = [30, 10, 12, 27, 10, 15, 21, 10, 11, 21,
                10, 11, 10, 14, 15, 10, 11, 12,
                10, 17, 21, 10, 14, 18, 10, 16, 23]
    inst_type_ls = ['A2_t', 'A2_t', 'A2_t', 'A2_t', 'B2_t', 'B2_t', 'B2_t', 'C2_t', 'C2_t', 'C2_t',
                    'D2_t', 'D2_t', 'E2_t', 'E2_t', 'E2_t', 'F2_t', 'F2_t', 'F2_t',
                    'G2_t', 'G2_t', 'G2_t', 'H2_t', 'H2_t', 'H2_t', 'Z2_t', 'Z2_t', 'Z2_t']
    g_ub = [3000, 1183.97772688239, 1223.43166168913, 1048.76179632512, 1324.85107007870,
            995.63552885192, 1032.38366645490, 907.13246727407, 928.89248051475,
            808.97556365474, 932.29264235740, 834.01168677771, 871.93889842301,
            863.85221949656, 1103.45317571000, 823.93362806332, 923.38973251718,
            840.04158281189, 876.27149628864]
    g_lb = [2223.444552527978, 1105.641815762578, 1168.9612626475478, 952.294837626547,
            1245.0040282370026,
            897.0901728221861,
            968.3427914422704, 884.702347423022, 927.1774936014718, 778.5358999880411, 791.9232719834832,
            793.7779078628338, 853.6967731062788, 768.5654002614888, 908.2373235150278, 685.4308466233264,
            785.0783940902456, 732.3827487287996, 756.4921667994806]
    makespan_forall = []
    time_forall = []
    for i in range(3,4):  # len(m_num_ls)
        solu = read_input('train', str(m_num_ls[i]), inst_type_ls[i], m_num_ls[i])
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        agent = DDQN(
            eval_net=QNet(device=device, in_dim_max=5, hidden=64,
                          out_dim=solu.init_env.ls_num, ma_num=solu.init_env.machine_num),
            target_net=QNet(device=device, in_dim_max=5, hidden=64,
                            out_dim=solu.init_env.ls_num, ma_num=solu.init_env.machine_num),
            dim_action=solu.init_env.ls_num,
            device=device,
            gamma=0.9,
            epsilon=0.5,
            lr=1e-5)
        solu.l2a_init()
        RLP = RelaxedCongestionPortModel(solu)
        RLP.construct()
        BB_priority_wide(label=inst_type_ls[i] + " " + str(m_num_ls[i]), RLP=RLP.MLP, solu=solu, g_ub=g_ub[i],
                         g_lb=g_lb[i])

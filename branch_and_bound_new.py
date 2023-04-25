# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 2:14 PM
# @Author  : JacQ
# @File    : branch_and_bound_new.py

from gurobipy import *
import numpy as np
import copy

import queue

eps = 1e-10


# Node class
class Node:
    # this class define the node
    local_LB = []
    local_UB = np.inf
    x_sol = {}
    x_int_sol = {}
    branch_var_list = []
    model = None
    name = None
    is_integer = False
    cnt = 0
    visited = []
    to_visited = []
    m = -1


def deepcopy_node(node):
    new_node = Node()
    new_node.local_LB = node.local_LB
    new_node.local_UB = node.local_UB
    new_node.x_sol = copy.deepcopy(node.x_sol)
    new_node.x_int_sol = copy.deepcopy(node.x_int_sol)
    new_node.branch_var_list = []
    new_node.model = node.model.copy()
    new_node.name = node.name
    new_node.visited = copy.deepcopy(node.visited)
    new_node.to_visited = copy.deepcopy(node.to_visited)
    new_node.m = node.m
    return new_node


# B&B | depth first | binary branching
def BB_depth_binary(RLP, solu, J_num, global_UB):
    # 固定xij
    vars = RLP.getVars()
    ls = solu.iter_env.mission_list
    for i in range(len(solu.iter_env.mission_list)):
        var_idx = (int(ls[i].idx[1:]) - 1) * 22 + int(ls[i].machine_list[4][-1]) + 2
        RLP.addConstr((vars[var_idx] == 1), "fixed_x" + str(i))
    RLP.update()
    RLP.optimize()
    var_ls = [3 + i * 22 + (J_num + 2) * 22 for i in range(0, (J_num + 2) * (J_num + 2))]
    var_ls.extend([4 + i * 22 + (J_num + 2) * 22 for i in range(0, (J_num + 2) * (J_num + 2))])
    var_ls.extend([5 + i * 22 + (J_num + 2) * 22 for i in range(0, (J_num + 2) * (J_num + 2))])
    var_ls.extend([6 + i * 22 + (J_num + 2) * 22 for i in range(0, (J_num + 2) * (J_num + 2))])

    # 初始化参数
    global_UB = global_UB
    global_LB = RLP.ObjVal
    eps = 1e-10
    incumbent_node = None
    branch_var_name = 'None'

    # 初始化node
    Queue = []
    node = Node()
    node.local_UB = float('INF')
    node.local_LB = RLP.ObjVal
    node.model = RLP.copy()
    node.model.setParam("OutputFlag", 0)
    Queue.append(node)  # start with only initial node
    Global_UB_change, Global_LB_change = [], []

    cnt = 0
    cnt_t = 0
    while len(Queue) > 0 and global_UB - global_LB > eps:
        # select the current node
        current_node = Queue.pop()
        current_node.model.optimize()
        '''
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUND = 5
        '''

        # check whether the current solution is integer
        is_integer = True
        is_Pruned = False

        print('\n --------------- \n', cnt_t, '\t', current_node.cnt, '\t Solution_status = ',
              current_node.model.Status, "\t branch_on：", branch_var_name)
        cnt_t = cnt_t + 1
        if current_node.cnt == 80:
            a = 1
        # bound step 模型有解
        if current_node.model.Status == 2:
            for var in current_node.model.getVars():
                current_node.x_sol[var.varName] = var.x
                current_node.x_int_sol[var.varName] = int(var.x)  # round the solution to get an integer solution
                if var.varName[0] is 'Y' and abs(int(var.x) - var.x) >= eps:  # 判断解是否是整数,不是整数就分支
                    is_integer = False
                    current_node.branch_var_list.append(var.VarName)
                    print(var.VarName, ' = ', var.x)

            '''Update the UB'''
            if is_integer:
                current_node.is_integer = True
                current_node.local_LB = current_node.model.ObjVal
                current_node.local_UB = current_node.model.ObjVal
                if current_node.local_UB < global_UB:
                    global_UB = current_node.local_UB
                    incumbent_node = deepcopy_node(current_node)
            if not is_integer:
                current_node.is_integer = False
                current_node.local_LB = current_node.model.ObjVal
                # for var_name in current_node.x_int_sol.keys():
                #     var = current_node.model.getVarByName(var_name)
                #     current_node.local_LB += current_node.x_int_sol[var_name] * var.Obj
                # if current_node.local_LB > global_LB:
                #     global_LB = current_node.local_LB
                #     incumbent_node = deepcopy_node(current_node)

            '''Pruning'''
            # prune by optimality直接解出整数解（最优）
            if is_integer:
                is_Pruned = True

            # prune by bound
            if not is_integer and current_node.local_LB > global_UB:  # 解出的解的下界大于当前全局上界
                is_Pruned = True

            Gap = (global_UB - global_LB) / global_LB
            print(current_node.cnt, '\t Gap = ', round(100 * Gap, 2), '%', '\n --------------- \n')

        else:  # 没有解出LP最优解
            print('\n --------------- \n')
            is_integer = False

            '''Pruning'''
            # prune by infeasible
            is_Pruned = True

            continue

        # branch step
        if not is_Pruned:
            # select the branch variable
            branch_var_name = current_node.branch_var_list[0]  # 优先prune第一个

            # create 2 child nodes
            left_node = deepcopy_node(current_node)
            right_node = deepcopy_node(current_node)

            temp_var = left_node.model.getVarByName(branch_var_name)
            left_node.model.addConstr(temp_var == 0, name='branch_left__' + str(cnt))
            left_node.model.setParam('OutputFlag', 0)
            left_node.model.update()
            cnt += 1
            left_node.cnt = cnt

            temp_var = right_node.model.getVarByName(branch_var_name)
            right_node.model.addConstr(temp_var == 1, name='branch_right__' + str(cnt))
            right_node.model.setParam('OutputFlag', 0)
            right_node.model.update()
            cnt += 1
            right_node.cnt = cnt

            Queue.append(left_node)
            Queue.append(right_node)

            Global_UB_change.append(global_UB)
            Global_LB_change.append(global_LB)

    # global_UB = global_LB  # 所有节点探索完 更新全局bound 保证while的退出条件
    Gap = round(100 * (global_UB - global_LB) / global_LB, 2)
    Global_UB_change.append(global_UB)
    Global_LB_change.append(global_LB)
    print('____________________________________')
    print('       Optimal solution found       ')
    print('____________________________________')
    print('queue_length:' + str(len(Queue)) + '\teps:' + str(global_UB - global_LB))
    print('Obj:', global_LB)
    print('Obj_LB:', Global_LB_change)
    print('Obj_UB:', Global_UB_change)
    print('Solution:', [list(incumbent_node.x_sol)[i] for i in var_ls if
                        incumbent_node.x_sol[list(incumbent_node.x_sol)[i]] != 0])
    return global_UB


# B&B | priority first | wide branching
def BB_priority_wide(RLP, solu, J_num, global_UB):
    # 固定xij&记录分配信息
    vars = RLP.getVars()
    ls = solu.iter_env.mission_list
    as_ls = {'S1': [], 'S2': [], 'S3': [], 'S4': []}
    for i in range(len(solu.iter_env.mission_list)):
        var_idx = (int(ls[i].idx[1:]) - 1) * 22 + int(ls[i].machine_list[4][-1]) + 2
        RLP.addConstr((vars[var_idx] == 1), "fixed_x" + str(i))
        as_ls[ls[i].machine_list[4]].append([int(ls[i].quay_crane_id[2]) - 1, int(ls[i].idx[1:]) - 1 -
                                             int((int(ls[i].quay_crane_id[2]) - 1) * J_num / 3),
                                             int(ls[i].idx[1:]) - 1])  # (QC,order,J_index) 都从0开始编号
    var_ls = []
    for i in range(0, (J_num + 2) * (J_num + 2)):
        var_ls.extend([3 + i * 22 + (J_num + 2) * 22, 4 + i * 22 + (J_num + 2) * 22, 5 + i * 22 + (J_num + 2) * 22,
                       6 + i * 22 + (J_num + 2) * 22])
    RLP.update()
    # RLP.setParam("OutputFlag", 0)
    RLP.optimize()

    # 初始化参数
    global_UB = global_UB
    global_LB = RLP.ObjVal
    cnt = 0  # 记录节点是树中的第几个节点
    cnt_t = 0  # 记录计算了几个节点
    incumbent_node = None
    branch_var_name = 'None'
    Global_UB_change, Global_LB_change = [], []

    # 初始化node
    Queue = queue.PriorityQueue()
    node = Node()
    node.local_UB = float('INF')
    node.local_LB = RLP.ObjVal
    node.model = RLP.copy()
    node.model.setParam("OutputFlag", 0)
    node.to_visited = copy.deepcopy(as_ls)
    node.visited = [[-1, -1, J_num]]
    node.model.optimize()
    Queue.put((RLP.ObjVal, (cnt, node)))

    # 每个机器初始节点分支
    while not Queue.empty() and global_UB - global_LB > eps:
        # select the current node
        current_po = Queue.get()  # 取下界最小的，优先探索
        current_node = current_po[1][1]

        print('\n --------------- \n', cnt_t, '\t', current_node.cnt, "\t visited：", current_node.visited)
        cnt_t += 1

        # check whether the current solution is integer
        is_integer = check_integer(current_node)
        is_Pruned = False

        # update bound
        global_LB = current_po[0]  # 全局最小下界即LB
        if is_integer:  # 如果是整数解check UB
            if current_node.local_UB < global_UB:
                global_UB = current_node.local_UB
                incumbent_node = deepcopy_node(current_node)

        # 二次prune 1: by optimality直接解出整数解（最优）
        if is_integer:
            is_Pruned = True

        # 二次prune 2: by bound
        if not is_integer and current_node.local_LB > global_UB:  # 解出的解的下界大于当前全局上界
            is_Pruned = True

        # 本轮process结束
        Gap = (global_UB - global_LB) / global_LB
        print(current_node.cnt, '\t Gap = ', round(100 * Gap, 10), '%', '\n --------------- \n')
        Global_UB_change.append(global_UB)
        Global_LB_change.append(global_LB)

        # branch step：把下一步所有可能变量加进去
        if not is_Pruned:
            if cnt == 0 or len(current_node.to_visited['S' + str(current_node.m + 1)]) == 0:
                current_node.m += 1
                # 每个机器初始节点分支
                for i in range(len(as_ls['S' + str(current_node.m + 1)])):
                    if cnt != 0:
                        current_node.visited.append([-1, -1, J_num])
                    tmp_node = check_prune(current_node, as_ls['S' + str(current_node.m + 1)][i], current_node.m)
                    cnt += 1
                    if tmp_node is not None:
                        tmp_node.cnt = cnt
                        Queue.put((tmp_node.model.ObjVal, (cnt, tmp_node)))
            else:
                to_visited_ls = copy.deepcopy(current_node.to_visited['S' + str(current_node.m + 1)])
                for i in range(len(to_visited_ls)):
                    tmp_node = check_prune(current_node, to_visited_ls[i], current_node.m)
                    cnt += 1
                    if tmp_node is not None:
                        tmp_node.cnt = cnt
                        Queue.put((tmp_node.model.ObjVal, (cnt, tmp_node)))

    Gap = round(100 * (global_UB - global_LB) / global_LB, 10)
    Global_UB_change.append(global_UB)
    Global_LB_change.append(global_LB)
    print('____________________________________')
    print('       Optimal solution found       ')
    print('____________________________________')
    print('queue_length:' + str(Queue.maxsize) + '\teps:' + str(global_UB - global_LB))
    print('Obj:', global_LB)
    print('Obj_LB:', Global_LB_change)
    print('Obj_UB:', Global_UB_change)
    print('Solution:', [list(incumbent_node.x_sol)[i] for i in var_ls if
                        incumbent_node.x_sol[list(incumbent_node.x_sol)[i]] != 0])
    return global_UB


def check_prune(node: Node, to_visit_node: [], m: int):
    # prune 1: 不符合QC顺序
    for pre_node in node.to_visited['S' + str(m + 1)]:
        if pre_node[0] == to_visit_node[0] and pre_node[1] < to_visit_node[1]:
            return None

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
        if var.varName[0] is 'Y' and abs(int(var.x) - var.x) >= eps:  # 判断解是否是整数,不是整数就分支
            is_integer = False
            node.branch_var_list.append(var.VarName)  # TODO
            print(var.VarName, ' = ', var.x)
    node.is_integer = True if is_integer else False
    node.local_UB = node.model.ObjVal if is_integer else node.local_UB

    return is_integer


if __name__ == '__main__':
    pass

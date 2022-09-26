# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 10:01 PM
# @Author  : JacQ
# @File    : gurobi_solver.py
import time
from copy import deepcopy

from gurobipy import *  # 在Python中调用gurobi求解包
import conf.configs as cf
from algorithm_factory.algo_utils import sort_missions
from algorithm_factory.algo_utils.machine_cal_methods import match_mission_yard_crane_num
from data_process.input_process import read_input


class Data:
    def __init__(self,
                 inst_idx: int,
                 J_num: int):
        self.inst_idx = inst_idx  # 算例名称
        self.S_num = 4  # 阶段数
        self.J_num = J_num  # 任务数
        self.M_num = 22  # 机器数
        self.big_M = 10000  # 无穷大数
        self.S = [0, 1, 2, 3]  # 阶段编号集合
        self.M_S = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]  # 每个阶段包括的机器编号集合
        self.M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 机器编号集合
        self.J = []  # 任务编号集合
        self.J_M = []  # 机器可处理任务集合
        self.J_m = []  # 任务对应机器
        self.A = []  # 顺序工件对集合
        self.st = [[self.big_M for _ in range(self.J_num * 3 + 2)] for _ in range(self.J_num * 3 + 2)]  # 两两之间距离
        self.pt = [[0 for _ in range(self.J_num * 3 + 2)] for _ in self.S]  # 每阶段处理任务所需时间
        self.tt = [[[0 for _ in range(self.M_num)] for _ in range(self.M_num - 12)] for _ in
                   range(self.J_num * 3 + 2)]
        self.alpha = [20, 25, 33.3, 50, 100]  # CO拥堵对应时间
        self.instance = None

    def init_process(self):
        instance = read_input('train_' + str(self.inst_idx) + '_', self.J_num)
        instance.l2a_init()
        self.instance = instance
        tmp_mission_ls = deepcopy(self.instance.init_env.mission_list)
        getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
        # J
        for j in range(self.J_num * 3):
            self.J.append(j)
        self.J.append(self.J[-1] + 1)  # dummy job N+1
        self.J.append(self.J[-1] + 1)  # dummy job N+2
        # J_M
        self.J_M = [[] for _ in range(self.M_num)]
        for qc in instance.init_env.quay_cranes.values():
            for mission in qc.missions.values():
                self.J_M[int(mission.quay_crane_id[-1]) - 1].append(int(mission.idx[1:]) - 1)  # QC 0-2
                self.J_M[int(mission.crossover_id[-1]) + 6].append(int(mission.idx[1:]) - 1)  # CO 7-9
                self.J_M[match_mission_yard_crane_num(mission)].append(int(mission.idx[1:]) - 1)  # YC 10-21
                self.J_m.append(match_mission_yard_crane_num(mission))  #
                # A
                if int(mission.idx[1:]) != (int(qc.idx[-1])) * self.J_num:
                    self.A.append([int(mission.idx[1:]) - 1, int(mission.idx[1:])])
                else:
                    self.A.append([int(mission.idx[1:]) - 1, self.J[-1]])

                # pt[s][j]
                self.pt[0][int(mission.idx[1:]) - 1] = mission.machine_process_time[0]
                self.pt[1][int(mission.idx[1:]) - 1] = mission.station_process_time
                self.pt[3][int(mission.idx[1:]) - 1] = mission.yard_crane_process_time
        # self.pt[0][self.J[-1]] = 0
        # self.pt[1][self.J[-1]] = 0
        # self.pt[3][self.J[-1]] = 0
        # self.pt[0][self.J[-2]] = 0
        # self.pt[1][self.J[-2]] = 0
        # self.pt[3][self.J[-2]] = 0

        for m in self.J_M:
            if any(m):
                m.append(self.J[-2])
                m.append(self.J[-1])

        # st
        for i in range(len(self.J) - 2):
            mission_i = tmp_mission_ls[i]
            for j in range(i, len(self.J)):
                if j == len(self.J) - 2:
                    self.st[j][i] = abs(mission_i.yard_block_loc[1]) * cf.SLOT_LENGTH / cf.YARDCRANE_SPEED_X + \
                                    abs(cf.SLOT_NUM_Y - mission_i.yard_block_loc[2]) * cf.SLOT_WIDTH \
                                    / cf.YARDCRANE_SPEED_Y * 2
                    self.st[j][-1] = 0
                elif j == len(self.J) - 1:
                    self.st[i][j] = 0
                else:
                    mission_j = tmp_mission_ls[j]
                    if mission_i.yard_block_loc[0] == mission_j.yard_block_loc[0] and mission_i != mission_j:
                        self.st[i][j] = abs(mission_i.yard_block_loc[1] - mission_j.yard_block_loc[1]) * cf.SLOT_LENGTH \
                                        / cf.YARDCRANE_SPEED_X + \
                                        abs(mission_i.yard_block_loc[2] - cf.SLOT_NUM_Y) * cf.SLOT_WIDTH \
                                        / cf.YARDCRANE_SPEED_Y * 2
                        self.st[j][i] = self.st[i][j]
                    else:
                        self.st[i][j] = self.big_M
                        self.st[j][i] = self.st[i][j]
        # tt
        for j in range(len(self.J)):  # todo dummy
            if j != len(self.J) - 1 and j != len(self.J) - 2:
                mission = tmp_mission_ls[j]
                for ls in range(4):
                    self.tt[j][int(mission.quay_crane_id[-1]) - 1][ls + 3] = \
                        self.instance.init_env.quay_cranes[mission.quay_crane_id].time_to_exit + \
                        self.instance.init_env.exit_to_station_matrix[ls] / mission.vehicle_speed
                    self.tt[j][ls + 3][int(mission.crossover_id[-1]) + 6] = \
                        self.instance.init_env.station_to_crossover_matrix[ls][int(mission.crossover_id[-1]) - 1] \
                        / mission.vehicle_speed
                self.tt[j][int(mission.crossover_id[-1]) + 6][match_mission_yard_crane_num(mission)] = \
                    tmp_mission_ls[j].transfer_time_c2y
            else:
                for qc in range(3):
                    for ls in range(4):
                        self.tt[j][qc][ls + 3] = 0
                for ls in range(4):
                    for co in range(3):
                        self.tt[j][ls + 3][co + 7] = 0
                for co in range(3):
                    for yc in range(12):
                        self.tt[j][co + 7][yc + 10] = 0

# def cal():
#     MLP_COM = construct_model()
#     for i in range(0, 50):
#
# def  construct_model_com():

def construct_model(data: Data):
    # ============== 构造模型 ================
    MLP = Model("terminal operation")
    tmp_mission_ls = deepcopy(data.instance.init_env.mission_list)
    getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)

    # ============== 定义变量 ================
    # X_jm
    X = [[[] for _ in data.M] for _ in data.J]
    for j in data.J:
        for m in data.M:
            name = 'X_' + str(j) + "_" + str(m)
            X[j][m] = MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
    # Y_jj'm
    Y = [[[[] for _ in data.M] for _ in data.J] for _ in data.J]
    for j in data.J:
        for jj in data.J:
            for m in data.M:
                name = 'Y_' + str(j) + "_" + str(jj) + "_" + str(m)
                Y[j][jj][m] = MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
    # Z_jj'm
    Z = [[[[] for _ in data.M] for _ in data.J] for _ in data.J]
    for j in data.J:
        for jj in data.J:
            for m in data.M:
                name = 'Z_' + str(j) + "_" + str(jj) + "_" + str(m)
                Z[j][jj][m] = MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
    # FTsj
    FT = [[[] for _ in data.J] for _ in data.S]
    for s in data.S:
        for j in data.J:
            name = 'FT_' + str(s) + "_" + str(j)
            FT[s][j] = MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
    # OTsj
    OT = [[[] for _ in data.J] for _ in data.S]
    for s in data.S:
        for j in data.J:
            name = 'OT_' + str(s) + "_" + str(j)
            OT[s][j] = MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
    # RTj
    RT = []
    for j in data.J:
        name = 'RT_' + str(j)
        RT.append(MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name))
    # TTsj
    TT = [[[] for _ in data.J] for _ in data.S]
    for s in data.S:
        for j in data.J:
            name = 'TT_' + str(s) + "_" + str(j)
            TT[s][j] = MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
    # ZZ[i][j]
    ZZ = [[0 for _ in data.J] for _ in range(5)]
    for i in range(5):
        for j in data.J:
            name = 'ZZ_' + str(i) + "_" + str(j)
            ZZ[i][j] = MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)

    # ============== 定义约束 ================
    MLP.addConstrs((X[j][m] == 1 for m in data.M for j in data.J_M[m]), "con2")
    MLP.addConstrs((sum(X[j][m] for m in data.M_S[s]) == 1 for j in data.J[0:-2] for s in data.S), "con3")
    for pair in data.A:
        j, jj = pair[0], pair[1]
        MLP.addConstr(RT[jj] - FT[0][j] >= 0, "con4" + str(j) + str(jj))
        m = int(tmp_mission_ls[j].quay_crane_id[-1]) - 1
        MLP.addConstr(Y[j][jj][m] == 1, "con5" + str(j) + str(jj))
    MLP.addConstrs((FT[s + 1][j] >= OT[s + 1][j] + (FT[s][j] + TT[s][j]) for s in range(3) for j in data.J), "con6")
    MLP.addConstrs((FT[0][j] == RT[j] + OT[0][j] for j in data.J), "con7")  # fixme >=
    MLP.addConstrs(
        (FT[s][jj] + TT[s][jj] + data.big_M - data.big_M * Y[j][jj][m] >= FT[s][j] + TT[s][j] for j in data.J for jj
         in data.J for s in range(3) for m in data.M_S[s + 1]), "con8")
    MLP.addConstrs(
        (X[j][m] + X[jj][m] + data.big_M - data.big_M * Y[j][jj][m] >= 2 for j in data.J for jj in data.J for m in
         data.M), "con9")
    MLP.addConstrs((Y[j][j][m] == 0 for m in data.M for j in data.J), "con1")
    MLP.addConstrs((sum(Y[j][jj][m] for jj in data.J) == X[j][m] for j in data.J[0:-1] for m in data.M), "con10")
    tmp_J = data.J[0:-2].copy()
    tmp_J.append(data.J[-1])
    MLP.addConstrs(
        (sum(Y[j][jj][m] for j in data.J) == X[jj][m] for jj in tmp_J for m in data.M),
        "con10")
    MLP.addConstrs((sum(Y[j][-1][m] for j in data.J[0:-1]) == X[-1][m] for m in data.M), "con10")
    MLP.addConstrs(
        (FT[s][jj] - OT[s][jj] - FT[s][j] + data.big_M - data.big_M * Y[j][jj][m] >= 0 for s in range(1, 4) for m in
         data.M_S[s] for j in data.J for jj in data.J), "con11")
    MLP.addConstrs(
        (FT[s + 1][j] - (FT[s][jj] + TT[s][jj]) + data.big_M - data.big_M * Z[j][jj][m] >= 0 for j in data.J for jj in
         data.J for s in range(3) for m in data.M_S[s + 1]), "con12")
    MLP.addConstrs(
        (FT[s][jj] + TT[s][jj] - FT[s][j] - TT[s][j] + data.big_M - data.big_M * Z[j][jj][m] >= 0 for j in data.J for
         jj in data.J for s in range(3) for m in data.M_S[s + 1]), "con13")
    MLP.addConstrs(
        (X[j][m] + X[jj][m] + data.big_M - data.big_M * Z[j][jj][m] >= 2 for j in data.J for jj in data.J for s in
         range(3) for m in data.M_S[s + 1]), "con14")
    MLP.addConstrs((Z[j][j][m] == 0 for m in data.M for j in data.J), "con1")
    MLP.addConstrs((OT[s][j] == data.pt[s][j] for s in range(0, 2) for j in data.J), "con151")
    MLP.addConstrs(
        (OT[2][j] == 0.0001 + data.alpha[0] * ZZ[0][j] + data.alpha[1] * ZZ[1][j] + data.alpha[2] * ZZ[2][j] +
         data.alpha[3] * ZZ[3][j] + data.alpha[4] * ZZ[4][j] for j in data.J), "con152")
    MLP.addConstrs(
        (sum(Z[jj][j][m] for jj in data.J for m in data.M_S[2]) <= ZZ[0][j] + 2 * ZZ[1][j] + 3 * ZZ[2][j] + 4 * ZZ[3][
            j] + data.big_M * ZZ[4][j] for j in data.J), "con152")
    MLP.addConstrs(
        (sum(Z[jj][j][m] for jj in data.J for m in data.M_S[2]) >= 2 * ZZ[1][j] + 3 * ZZ[2][j] + 4 * ZZ[3][j] for j in
         data.J), "con162")
    MLP.addConstrs(
        (OT[3][j] == data.pt[3][j] + sum(Y[jj][j][data.J_m[j]] * data.st[jj][j] for jj in data.J) for j in
         data.J[0:-2]),
        "con163")
    # + (1 - sum(Y[jj][j][m] for jj in data.J[0:-1] for m in data.M_S[3])) * data.st[-1][j]
    MLP.addConstr(OT[3][-1] == 0, "con163")
    MLP.addConstr(OT[3][-2] == 0, "con163")
    MLP.addConstrs(
        (TT[s][j] - data.tt[j][m][mm] + data.big_M * 2 - data.big_M * X[j][m] - data.big_M * X[j][mm] >= 0 for j in
         data.J[0:-2] for s in range(0, 3) for m in data.M_S[s] for mm in data.M_S[s + 1]), "con17")
    MLP.addConstrs((TT[s][data.J[-1]] == 0 for s in range(0, 3)), "con18")

    # ============== 构造目标 ================
    q = MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='q')  # 线性化模型变量
    MLP.addConstrs((q >= FT[3][j] for j in data.J), "obj1")
    MLP.setObjective(q, GRB.MINIMIZE)

    # for var in MLP.getVars():
    #     if var.X == 1:
    #         print(var.VarName + " " + str(var.X))
    return MLP


def solve_model(MLP, inst_idx, J_num, fix_X=None):
    vars = MLP.getVars()
    # ============== 输入解 ================
    if fix_X is not None:
        if MLP.getConstrs() is not None:
            MLP.remove(MLP.getConstrs()[-J_num:-1])
        for pair in fix_X:
            MLP.addConstrs(vars[(pair[0] - 1) * 22 + pair[1] + 3] == 1, "fixed")  # station1->0

    # ============== 求解模型 ================
    MLP.write("output_result/gurobi/mod_" + str(inst_idx) + "_" + str(J_num) + ".lp")
    MLP.Params.timelimit = 3600
    T1 = time.time()
    MLP.optimize()
    print("time: " + str(time.time() - T1))
    if MLP.status == GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % MLP.status)
        # do IIS, find infeasible constraints
        MLP.computeIIS()
        for c in MLP.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        MLP.write("output_result/gurobi/sol_" + str(inst_idx) + "_" + str(J_num) + ".ilp")
    MLP.write("output_result/gurobi/sol_" + str(inst_idx) + "_" + str(J_num) + ".sol")
    return MLP


if __name__ == '__main__':
    inst_idx = 0
    J_num = 1
    print("current instance on run is:" + str(inst_idx) + "_" + str(J_num))
    data = Data(inst_idx=0, J_num=J_num)
    data.init_process()
    MLP = construct_model(data)
    solve_model(MLP=MLP, inst_idx=0, J_num=J_num)

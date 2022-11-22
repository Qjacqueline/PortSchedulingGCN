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
from common.iter_solution import IterSolution
from data_process.input_process import read_input
import conf.configs as cf


class Data:
    def __init__(self,
                 inst_idx: int,
                 J_num: int):
        self.inst_idx = inst_idx  # 算例名称
        self.J_num = J_num  # 任务数
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
        self.theta = [20, 25, 33.3, 50, 100]  # CO拥堵对应时间
        self.instance = None
        self.init_process()

    def init_process(self):
        instance = read_input('train_' + str(self.inst_idx) + '_', self.J_num)
        instance.l2a_init()
        self.instance = instance
        tmp_mission_ls = deepcopy(self.instance.init_env.mission_list)
        getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
        # J
        self.J = [j for j in range(self.J_num * 3)]
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
                # if int(mission.idx[1:]) % self.J_num == 1:
                #     self.A.append([self.J[-2], int(mission.idx[1:]) - 1])
                if int(mission.idx[1:]) != (int(qc.idx[-1])) * self.J_num:
                    self.A.append([int(mission.idx[1:]) - 1, int(mission.idx[1:])])
                # else:
                #     self.A.append([int(mission.idx[1:]) - 1, self.J[-1]])

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
                                        abs(mission_j.yard_block_loc[2] - cf.SLOT_NUM_Y) * cf.SLOT_WIDTH \
                                        / cf.YARDCRANE_SPEED_Y * 2
                        self.st[j][i] = abs(mission_i.yard_block_loc[1] - mission_j.yard_block_loc[1]) * cf.SLOT_LENGTH \
                                        / cf.YARDCRANE_SPEED_X + \
                                        abs(mission_i.yard_block_loc[2] - cf.SLOT_NUM_Y) * cf.SLOT_WIDTH \
                                        / cf.YARDCRANE_SPEED_Y * 2
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


class PortModel:
    def __init__(self, J_num: int, data: Data = None):
        self.S_num = 4  # 阶段数
        self.J_num = J_num  # 任务数
        self.M_num = 22  # 机器数
        self.big_M = 10000  # 无穷大数
        self.S = [0, 1, 2, 3]  # 阶段编号集合
        self.M_S = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]  # 每个阶段包括的机器编号集合
        self.M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 机器编号集合
        self.J = [j for j in range(self.J_num * 3)]  # 任务编号集合
        self.J.append(self.J[-1] + 1)  # dummy job N+1
        self.J.append(self.J[-1] + 1)  # dummy job N+2
        self.theta = [20, 25, 33.3, 50, 100]  # CO拥堵对应时间

        self.MLP = None
        self.X = None  # X_jm
        self.Y = None  # Y_jj'm
        self.Z = None  # Z_jj'm
        self.FT = None  # FTsj
        self.OT = None  # OTsj
        self.RT = None  # RTj
        self.TT = None  # TTsj
        self.ZZ = None  # ZZ[i][j]
        self.alpha = None
        self.beta = None

        self.data = data

    def construct(self):
        # ============== 构造模型 ================
        self.MLP = Model("port operation")

        # ============== 定义变量 ================
        self.X = [[[] for _ in self.M] for _ in self.J]
        for j in self.J:
            for m in self.M:
                name = 'X_' + str(j) + "_" + str(m)
                self.X[j][m] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        # Y_jj'm
        self.Y = [[[[] for _ in self.M] for _ in self.J] for _ in self.J]
        for j in self.J:
            for jj in self.J:
                for m in self.M:
                    name = 'Y_' + str(j) + "_" + str(jj) + "_" + str(m)
                    self.Y[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        # Z_jj'm
        self.Z = [[[[] for _ in self.M] for _ in self.J] for _ in self.J]
        for j in self.J:
            for jj in self.J:
                for m in self.M:
                    name = 'Z_' + str(j) + "_" + str(jj) + "_" + str(m)
                    self.Z[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        # FTsj
        self.FT = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'FT_' + str(s) + "_" + str(j)
                self.FT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # OTsj
        self.OT = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'OT_' + str(s) + "_" + str(j)
                self.OT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # RTj
        self.RT = []
        for j in self.J:
            name = 'RT_' + str(j)
            self.RT.append(self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name))
        # TTsj
        self.TT = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'TT_' + str(s) + "_" + str(j)
                self.TT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # ZZ[i][j]
        self.ZZ = [[0 for _ in self.J] for _ in range(5)]
        for i in range(5):
            for j in self.J:
                name = 'ZZ_' + str(i) + "_" + str(j)
                self.ZZ[i][j] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        # theta[s][j]
        self.alpha = [[0 for _ in self.J] for _ in range(4)]
        for i in range(4):
            for j in self.J:
                name = 'alpha_' + str(i) + "_" + str(j)
                self.alpha[i][j] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        self.beta = [[0 for _ in self.J] for _ in range(4)]
        for i in range(4):
            for j in self.J:
                name = 'beta_' + str(i) + "_" + str(j)
                self.beta[i][j] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)

        # ============== 定义公共约束 ================
        self.MLP.addConstrs((self.Y[j][j][m] == 0 for m in self.M for j in self.J), "con1")
        self.MLP.addConstrs((self.Z[j][j][m] == 0 for m in self.M for j in self.J), "con1")
        self.MLP.addConstrs((sum(self.X[j][m] for m in self.M_S[s]) == 1 for j in self.J[0:-2] for s in self.S), "con3")
        self.MLP.addConstrs(
            (self.FT[s + 1][j] >= self.OT[s + 1][j] + (self.FT[s][j] + self.TT[s][j]) for s in range(3) for j in
             self.J), "con6")
        self.MLP.addConstrs((self.FT[0][j] == self.RT[j] + self.OT[0][j] for j in self.J), "con7")  # fixme >=
        self.MLP.addConstrs(
            (self.FT[s][jj] + self.TT[s][jj] + self.big_M - self.big_M * self.Y[j][jj][m] >= self.FT[s][j] + self.TT[s][
                j] for j in self.J for jj in self.J for s in range(3) for m in self.M_S[s + 1]), "con8")

        tmp_J = self.J[0:-2].copy()
        tmp_J.append(self.J[-1])
        self.MLP.addConstrs(
            (sum(self.Y[jj][j][m] for jj in self.J) == self.X[j][m] for j in tmp_J for m in self.M), "con10")
        self.MLP.addConstrs(
            (sum(self.Y[j][jj][m] for jj in self.J) == self.X[j][m] for j in self.J[0:-1] for m in self.M), "con11")
        # self.MLP.addConstrs((self.Y[j][jj][m] + self.Y[jj][j][m] <= 1 for j in self.J for jj in self.J for m in self.M),
        #                     "con12")
        # self.MLP.addConstrs(
        #     (self.X[j][m] + self.X[jj][m] + self.big_M - self.big_M * self.Y[j][jj][m] >= 2 for j in self.J for jj in
        #      self.J for m in self.M), "con9")
        # self.MLP.addConstrs(
        #     (self.FT[s][jj] - self.OT[s][jj] - self.FT[s][j] + self.big_M - self.big_M * self.Y[j][jj][m] >= 0 for s in
        #      range(1, 4) for m in self.M_S[s] for j in self.J for jj in self.J), "con13")
        for s in range(1, 4):
            for m in self.M_S[s]:
                for j in self.J:
                    for jj in self.J:
                        self.MLP.addConstr(
                            self.FT[s][jj] - self.OT[s][jj] - self.FT[s][j] + 3 * self.big_M - self.big_M *
                            self.Y[j][jj][m] - self.big_M * self.X[j][m] - self.big_M * self.X[jj][m] >= 0, "add13")

        self.MLP.addConstrs(
            (self.FT[s + 1][j] - self.OT[s + 1][j] <= self.FT[s][j] + self.TT[s][j] + self.big_M - self.big_M *
             self.alpha[s + 1][j] for s in range(0, 3) for j in self.J[0:-2]), "con141")
        self.MLP.addConstrs(
            (self.FT[s][j] - self.OT[s][j] <= self.FT[s][jj] + 2 * self.big_M - self.big_M * self.beta[s][
                j] - self.big_M * self.Y[jj][j][m] for s in range(1, 4) for m in self.M_S[s] for j in self.J[0:-2]
             for jj in self.J), "con142")
        self.MLP.addConstrs((self.alpha[s][j] + self.beta[s][j] >= 1 for s in range(1, 4) for j in self.J[0:-2]),
                            "con143")
        self.MLP.addConstrs(
            (self.FT[s + 1][j] + 3 * self.big_M >= (self.FT[s][jj] + self.TT[s][jj]) + self.big_M * self.Z[j][jj][
                m] + self.big_M * self.X[j][m] + self.big_M * self.X[jj][m] for j in self.J for jj in self.J for
             s in range(3) for m in self.M_S[s + 1]), "con15")
        self.MLP.addConstrs(
            (self.FT[s][jj] + self.TT[s][jj] - self.FT[s][j] - self.TT[s][j] + 3 * self.big_M - self.big_M *
             self.Z[j][jj][m] - self.big_M * self.X[j][m] - self.big_M * self.X[jj][m] >= 0 for j in self.J
             for jj in self.J for s in range(3) for m in self.M_S[s + 1]), "con16")  # fixme
        # self.MLP.addConstrs(
        #     (self.X[j][m] + self.X[jj][m] + self.big_M - self.big_M * self.Z[j][jj][m] >= 2 for j in self.J for jj in
        #      self.J for s in range(3) for m in self.M_S[s + 1]), "con17")
        self.MLP.addConstrs(
            (self.OT[2][j] == 5 + self.theta[0] * self.ZZ[0][j] + self.theta[1] * self.ZZ[1][j] + self.theta[2] *
             self.ZZ[2][j] + self.theta[3] * self.ZZ[3][j] + self.theta[4] * self.ZZ[4][j] for j in self.J[0:-2]),
            "con23")
        self.MLP.addConstrs(
            (sum(self.Z[jj][j][m] for jj in self.J for m in self.M_S[2]) <= self.ZZ[0][j] + 2 * self.ZZ[1][j] + 3 *
             self.ZZ[2][j] + 4 * self.ZZ[3][j] + self.big_M * self.ZZ[4][j] for j in self.J[0:-2]), "con24")
        self.MLP.addConstrs(
            (sum(self.Z[jj][j][m] for jj in self.J for m in self.M_S[2]) >= 2 * self.ZZ[1][j] + 3 * self.ZZ[2][j] + 4 *
             self.ZZ[3][j] for j in self.J[0:-2]), "con25")
        self.MLP.addConstrs((self.OT[s][-1] == 0 for s in self.S), "con180")
        self.MLP.addConstrs((self.OT[s][-2] == 0 for s in self.S), "con180")
        self.MLP.addConstrs((self.TT[s][self.J[-1]] == 0 for s in range(0, 3)), "con21")
        self.MLP.addConstrs((self.TT[s][self.J[-2]] == 0 for s in range(0, 3)), "con21")

        # ============== 构造目标 ================
        q = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='q')  # 线性化模型变量
        self.MLP.addConstrs((q >= self.FT[3][j] for j in self.J), "obj1")
        self.MLP.setObjective(q,
                              GRB.MINIMIZE)  # fixme+ 0.000001 * sum(self.FT[s][j] for s in self.S for j in self.J)

        # ============== 定义差异约束 ================
        tmp_mission_ls = deepcopy(self.data.instance.init_env.mission_list)
        getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
        self.MLP.addConstrs((self.X[j][m] == 1 for m in self.M for j in self.data.J_M[m]), "con2")
        for pair in self.data.A:
            j, jj = pair[0], pair[1]
            self.MLP.addConstr(self.RT[jj] - self.FT[0][j] >= 0, "con4" + str(j) + str(jj))
            m = int(tmp_mission_ls[j].quay_crane_id[-1]) - 1
            self.MLP.addConstr(self.Y[j][jj][m] == 1, "con5" + str(j) + str(jj))
            # self.MLP.addConstr(self.RT[jj] - self.RT[j] == 120, "con00" + str(j) + str(jj))  # FIXME: match RL
        self.MLP.addConstr(self.RT[-2] == 0, "con00")
        # for i in range(3):
        #     self.MLP.addConstr(self.RT[i * self.J_num] == 0, "con00")  # FIXME: match RL
        self.MLP.addConstrs((self.OT[s][j] == self.data.pt[s][j] for s in range(0, 2) for j in self.J[0:-2]), "con181")
        self.MLP.addConstrs((self.OT[3][j] == self.data.pt[3][j] + sum(
            self.Y[jj][j][self.data.J_m[j]] * self.data.st[jj][j] for jj in self.J) for j in self.J[0:-2]), "con183")
        self.MLP.addConstrs(
            (self.TT[s][j] - self.data.tt[j][m][mm] + self.big_M * 2 - self.big_M * self.X[j][m] - self.big_M *
             self.X[j][mm] >= 0 for j in self.J[0:-2] for s in range(0, 3) for m in self.M_S[s] for mm in
             self.M_S[s + 1]), "con19")
        self.MLP.addConstrs(
            (self.TT[s][j] - self.data.tt[j][m][mm] <= self.big_M * 2 - self.big_M * self.X[j][m] - self.big_M *
             self.X[j][mm] for j in self.J[0:-2] for s in range(0, 3) for m in self.M_S[s] for mm in self.M_S[s + 1]),
            "con20")
        self.MLP.update()


class PortModelRLP:
    def __init__(self, J_num: int, data: Data = None):
        self.S_num = 4  # 阶段数
        self.J_num = J_num  # 任务数
        self.M_num = 22  # 机器数
        self.big_M = 10000  # 无穷大数
        self.S = [0, 1, 2, 3]  # 阶段编号集合
        self.M_S = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]  # 每个阶段包括的机器编号集合
        self.M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 机器编号集合
        self.J = [j for j in range(self.J_num * 3)]  # 任务编号集合
        self.J.append(self.J[-1] + 1)  # dummy job N+1
        self.J.append(self.J[-1] + 1)  # dummy job N+2
        self.theta = [20, 25, 33.3, 50, 100]  # CO拥堵对应时间

        self.MLP = None
        self.X = None  # X_jm
        self.Y = None  # Y_jj'm
        self.Z = None  # Z_jj'm
        self.FT = None  # FTsj
        self.OT = None  # OTsj
        self.RT = None  # RTj
        self.TT = None  # TTsj
        self.ZZ = None  # ZZ[i][j]
        self.alpha = None
        self.beta = None

        self.data = data

    def construct(self):
        # ============== 构造模型 ================
        self.MLP = Model("port operation")

        # ============== 定义变量 ================
        self.X = [[[] for _ in self.M] for _ in self.J]
        for j in self.J:
            for m in self.M:
                name = 'X_' + str(j) + "_" + str(m)
                self.X[j][m] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        # Y_jj'm
        self.Y = [[[[] for _ in self.M] for _ in self.J] for _ in self.J]
        for j in self.J:
            for jj in self.J:
                for m in self.M:
                    if 3 <= m <= 6:
                        name = 'Y_' + str(j) + "_" + str(jj) + "_" + str(m)
                        self.Y[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
                    else:
                        name = 'Y_' + str(j) + "_" + str(jj) + "_" + str(m)
                        self.Y[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
        # Z_jj'm
        self.Z = [[[[] for _ in self.M] for _ in self.J] for _ in self.J]
        for j in self.J:
            for jj in self.J:
                for m in self.M:
                    name = 'Z_' + str(j) + "_" + str(jj) + "_" + str(m)
                    self.Z[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
        # FTsj
        self.FT = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'FT_' + str(s) + "_" + str(j)
                self.FT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # OTsj
        self.OT = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'OT_' + str(s) + "_" + str(j)
                self.OT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # RTj
        self.RT = []
        for j in self.J:
            name = 'RT_' + str(j)
            self.RT.append(self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name))
        # TTsj
        self.TT = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'TT_' + str(s) + "_" + str(j)
                self.TT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # ZZ[i][j]
        self.ZZ = [[0 for _ in self.J] for _ in range(5)]
        for i in range(5):
            for j in self.J:
                name = 'ZZ_' + str(i) + "_" + str(j)
                self.ZZ[i][j] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
        # theta[s][j]
        self.alpha = [[0 for _ in self.J] for _ in range(4)]
        for i in range(4):
            for j in self.J:
                name = 'alpha_' + str(i) + "_" + str(j)
                self.alpha[i][j] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
        self.beta = [[0 for _ in self.J] for _ in range(4)]
        for i in range(4):
            for j in self.J:
                name = 'beta_' + str(i) + "_" + str(j)
                self.beta[i][j] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)

        # ============== 定义公共约束 ================
        self.MLP.addConstrs((self.Y[j][j][m] == 0 for m in self.M for j in self.J), "con1")
        self.MLP.addConstrs((self.Y[j][self.J[-2]][m] == 0 for m in self.M for j in self.J), "con1")
        self.MLP.addConstrs((self.Y[self.J[-1]][j][m] == 0 for m in self.M for j in self.J), "con1")
        self.MLP.addConstrs((self.Z[j][j][m] == 0 for m in self.M for j in self.J), "con1")
        self.MLP.addConstrs((sum(self.X[j][m] for m in self.M_S[s]) == 1 for j in self.J[0:-2] for s in self.S), "con3")
        self.MLP.addConstrs(
            (self.FT[s + 1][j] >= self.OT[s + 1][j] + (self.FT[s][j] + self.TT[s][j]) for s in range(3) for j in
             self.J), "con6")
        self.MLP.addConstrs((self.FT[0][j] == self.RT[j] + self.OT[0][j] for j in self.J), "con7")  # fixme >=
        self.MLP.addConstrs(
            (self.FT[s][jj] + self.TT[s][jj] + self.big_M - self.big_M * self.Y[j][jj][m] >= self.FT[s][j] + self.TT[s][
                j] for j in self.J for jj in self.J for s in range(3) for m in self.M_S[s + 1]), "con8")

        tmp_J = self.J[0:-2].copy()
        tmp_J.append(self.J[-1])
        self.MLP.addConstrs(
            (sum(self.Y[jj][j][m] for jj in self.J) == self.X[j][m] for j in tmp_J for m in self.M), "con10")
        self.MLP.addConstrs(
            (sum(self.Y[j][jj][m] for jj in self.J) == self.X[j][m] for j in self.J[0:-1] for m in self.M), "con11")
        # self.MLP.addConstrs((self.Y[j][jj][m] + self.Y[jj][j][m] <= 1 for j in self.J for jj in self.J for m in self.M),
        #                     "con12")
        # self.MLP.addConstrs(
        #     (self.X[j][m] + self.X[jj][m] + self.big_M - self.big_M * self.Y[j][jj][m] >= 2 for j in self.J for jj in
        #      self.J for m in self.M), "con9")
        # self.MLP.addConstrs(
        #     (self.FT[s][jj] - self.OT[s][jj] - self.FT[s][j] + self.big_M - self.big_M * self.Y[j][jj][m] >= 0 for s in
        #      range(1, 4) for m in self.M_S[s] for j in self.J for jj in self.J), "con13")
        for s in range(1, 4):
            for m in self.M_S[s]:
                for j in self.J:
                    for jj in self.J:
                        self.MLP.addConstr(
                            self.FT[s][jj] - self.OT[s][jj] - self.FT[s][j] + 3 * self.big_M - self.big_M *
                            self.Y[j][jj][m] - self.big_M * self.X[j][m] - self.big_M * self.X[jj][m] >= 0, "add13")

        self.MLP.addConstrs(
            (self.FT[s + 1][j] - self.OT[s + 1][j] <= self.FT[s][j] + self.TT[s][j] + self.big_M - self.big_M *
             self.alpha[s + 1][j] for s in range(0, 3) for j in self.J[0:-2]), "con141")
        self.MLP.addConstrs(
            (self.FT[s][j] - self.OT[s][j] <= self.FT[s][jj] + 2 * self.big_M - self.big_M * self.beta[s][
                j] - self.big_M * self.Y[jj][j][m] for s in range(1, 4) for m in self.M_S[s] for j in self.J[0:-2]
             for jj in self.J), "con142")
        self.MLP.addConstrs((self.alpha[s][j] + self.beta[s][j] >= 1 for s in range(1, 4) for j in self.J[0:-2]),
                            "con143")
        self.MLP.addConstrs(
            (self.FT[s + 1][j] + 3 * self.big_M >= (self.FT[s][jj] + self.TT[s][jj]) + self.big_M * self.Z[j][jj][
                m] + self.big_M * self.X[j][m] + self.big_M * self.X[jj][m] for j in self.J for jj in self.J for
             s in range(3) for m in self.M_S[s + 1]), "con15")
        self.MLP.addConstrs(
            (self.FT[s][jj] + self.TT[s][jj] - self.FT[s][j] - self.TT[s][j] + 3 * self.big_M - self.big_M *
             self.Z[j][jj][m] - self.big_M * self.X[j][m] - self.big_M * self.X[jj][m] >= 0 for j in self.J
             for jj in self.J for s in range(3) for m in self.M_S[s + 1]), "con16")  # fixme
        # self.MLP.addConstrs(
        #     (self.X[j][m] + self.X[jj][m] + self.big_M - self.big_M * self.Z[j][jj][m] >= 2 for j in self.J for jj in
        #      self.J for s in range(3) for m in self.M_S[s + 1]), "con17")
        self.MLP.addConstrs(
            (self.OT[2][j] == 5 + self.theta[0] * self.ZZ[0][j] + self.theta[1] * self.ZZ[1][j] + self.theta[2] *
             self.ZZ[2][j] + self.theta[3] * self.ZZ[3][j] + self.theta[4] * self.ZZ[4][j] for j in self.J[0:-2]),
            "con23")
        self.MLP.addConstrs(
            (sum(self.Z[jj][j][m] for jj in self.J for m in self.M_S[2]) <= self.ZZ[0][j] + 2 * self.ZZ[1][j] + 3 *
             self.ZZ[2][j] + 4 * self.ZZ[3][j] + self.big_M * self.ZZ[4][j] for j in self.J[0:-2]), "con24")
        self.MLP.addConstrs(
            (sum(self.Z[jj][j][m] for jj in self.J for m in self.M_S[2]) >= 2 * self.ZZ[1][j] + 3 * self.ZZ[2][j] + 4 *
             self.ZZ[3][j] for j in self.J[0:-2]), "con25")
        self.MLP.addConstrs((self.OT[s][-1] == 0 for s in self.S), "con180")
        self.MLP.addConstrs((self.OT[s][-2] == 0 for s in self.S), "con180")
        self.MLP.addConstrs((self.TT[s][self.J[-1]] == 0 for s in range(0, 3)), "con21")
        self.MLP.addConstrs((self.TT[s][self.J[-2]] == 0 for s in range(0, 3)), "con21")

        # ============== 构造目标 ================
        q = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='q')  # 线性化模型变量
        self.MLP.addConstrs((q >= self.FT[3][j] for j in self.J), "obj1")
        self.MLP.setObjective(q,
                              GRB.MINIMIZE)  # fixme+ 0.000001 * sum(self.FT[s][j] for s in self.S for j in self.J)

        # ============== 定义差异约束 ================
        tmp_mission_ls = deepcopy(self.data.instance.init_env.mission_list)
        getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
        self.MLP.addConstrs((self.X[j][m] == 1 for m in self.M for j in self.data.J_M[m]), "con2")
        for pair in self.data.A:
            j, jj = pair[0], pair[1]
            self.MLP.addConstr(self.RT[jj] - self.FT[0][j] >= 0, "con4" + str(j) + str(jj))
            m = int(tmp_mission_ls[j].quay_crane_id[-1]) - 1
            self.MLP.addConstr(self.Y[j][jj][m] == 1, "con5" + str(j) + str(jj))
            # self.MLP.addConstr(self.RT[jj] - self.RT[j] == 120, "con00" + str(j) + str(jj))  # FIXME: match RL
        self.MLP.addConstr(self.RT[-2] == 0, "con00")
        # for i in range(3):
        #     self.MLP.addConstr(self.RT[i * self.J_num] == 0, "con00")  # FIXME: match RL
        self.MLP.addConstrs((self.OT[s][j] == self.data.pt[s][j] for s in range(0, 2) for j in self.J[0:-2]), "con181")
        self.MLP.addConstrs((self.OT[3][j] == self.data.pt[3][j] + sum(
            self.Y[jj][j][self.data.J_m[j]] * self.data.st[jj][j] for jj in self.J) for j in self.J[0:-2]), "con183")
        self.MLP.addConstrs(
            (self.TT[s][j] - self.data.tt[j][m][mm] + self.big_M * 2 - self.big_M * self.X[j][m] - self.big_M *
             self.X[j][mm] >= 0 for j in self.J[0:-2] for s in range(0, 3) for m in self.M_S[s] for mm in
             self.M_S[s + 1]), "con19")
        self.MLP.addConstrs(
            (self.TT[s][j] - self.data.tt[j][m][mm] <= self.big_M * 2 - self.big_M * self.X[j][m] - self.big_M *
             self.X[j][mm] for j in self.J[0:-2] for s in range(0, 3) for m in self.M_S[s] for mm in self.M_S[s + 1]),
            "con20")
        self.MLP.update()


def solve_model(MLP, inst_idx, J_num, solu: IterSolution = None, tag='', X_flag=True, Z_flag=True):
    vars = MLP.getVars()
    # ============== 输入解 ================
    if solu is not None:
        # mode
        X_flag = X_flag
        Z_flag = Z_flag
        # add valid equality
        # MLP.addConstr(vars[-1] <= solu.last_step_makespan, "ub")

        ls = solu.iter_env.mission_list
        # fix X_jm
        if X_flag:
            for i in range(len(solu.iter_env.mission_list)):
                var_idx = (int(ls[i].idx[1:]) - 1) * 22 + int(ls[i].machine_list[4][-1]) + 2
                MLP.addConstr((vars[var_idx] == 1), "fixed_x" + str(i))
        # fix Z_jj'm
        if Z_flag:
            for i in range(len(solu.iter_env.lock_stations)):
                for j in range(len(solu.iter_env.lock_stations['S' + str(i + 1)].mission_list)):
                    p_mission_idx = int(solu.iter_env.lock_stations['S' + str(i + 1)].mission_list[j].idx[1:]) - 1
                    if j == 0 and j == len(solu.iter_env.lock_stations['S' + str(i + 1)].mission_list) - 1:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * J_num * 3 + 22 * p_mission_idx + 3 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls" + str(i))
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * p_mission_idx + 22 * (
                                J_num * 3 + 1) + 3 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls" + str(i))
                    elif j == 0:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * J_num * 3 + 22 * p_mission_idx + 3 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls" + str(i))
                        l_mission_idx = int(
                            solu.iter_env.lock_stations['S' + str(i + 1)].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num * 3 + 2) * 22 + (
                                J_num * 3 + 2) * 22 * p_mission_idx + 22 * l_mission_idx + 3 + i
                        # Y[p_mission_idx][l_mission_idx][3 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls" + str(i))
                    elif j == len(solu.iter_env.lock_stations['S' + str(i + 1)].mission_list) - 1:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * p_mission_idx + 22 * (
                                J_num * 3 + 1) + 3 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls" + str(i))
                    else:
                        l_mission_idx = int(
                            solu.iter_env.lock_stations['S' + str(i + 1)].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num * 3 + 2) * 22 + (
                                J_num * 3 + 2) * 22 * p_mission_idx + 22 * l_mission_idx + 3 + i
                        # Y[p_mission_idx][l_mission_idx][3 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls" + str(i))
            for i in range(len(solu.iter_env.crossovers)):
                for j in range(len(solu.iter_env.crossovers['CO' + str(i + 1)].mission_list)):
                    p_mission_idx = int(solu.iter_env.crossovers['CO' + str(i + 1)].mission_list[j].idx[1:]) - 1
                    if j == 0 and j == len(solu.iter_env.crossovers['CO' + str(i + 1)].mission_list) - 1:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * J_num * 3 + 22 * p_mission_idx + 7 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co" + str(i))
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * p_mission_idx + 22 * (
                                J_num * 3 + 1) + 7 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co" + str(i))
                    elif j == 0:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * J_num * 3 + 22 * p_mission_idx + 7 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co" + str(i))
                        l_mission_idx = int(solu.iter_env.crossovers['CO' + str(i + 1)].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num * 3 + 2) * 22 + (
                                J_num * 3 + 2) * 22 * p_mission_idx + 22 * l_mission_idx + 7 + i
                        # Y[p_mission_idx][l_mission_idx][7 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co" + str(i))
                    elif j == len(solu.iter_env.crossovers['CO' + str(i + 1)].mission_list) - 1:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * p_mission_idx + 22 * (
                                J_num * 3 + 1) + 7 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co" + str(i))
                    else:
                        l_mission_idx = int(solu.iter_env.crossovers['CO' + str(i + 1)].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num * 3 + 2) * 22 + (
                                J_num * 3 + 2) * 22 * p_mission_idx + 22 * l_mission_idx + 7 + i
                        # Y[p_mission_idx][l_mission_idx][7 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co" + str(i))
            for i in range(len(solu.iter_env.yard_cranes)):
                for j in range(len(list(solu.iter_env.yard_cranes.values())[i].mission_list)):
                    p_mission_idx = int(list(solu.iter_env.yard_cranes.values())[i].mission_list[j].idx[1:]) - 1
                    if j == 0 and j == len(list(solu.iter_env.yard_cranes.values())[i].mission_list) - 1:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * J_num * 3 + 22 * p_mission_idx + 10 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc" + str(i))
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * p_mission_idx + 22 * (
                                J_num * 3 + 1) + 10 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc" + str(i))
                    elif j == 0:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * J_num * 3 + 22 * p_mission_idx + 10 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc" + str(i))
                        l_mission_idx = int(list(solu.iter_env.yard_cranes.values())[i].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num * 3 + 2) * 22 + (
                                J_num * 3 + 2) * 22 * p_mission_idx + 22 * l_mission_idx + 10 + i
                        # Y[p_mission_idx][l_mission_idx][7 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc" + str(i))
                    elif j == len(list(solu.iter_env.yard_cranes.values())[i].mission_list) - 1:
                        var_idx = (J_num * 3 + 2) * 22 + (J_num * 3 + 2) * 22 * p_mission_idx + 22 * (
                                J_num * 3 + 1) + 10 + i
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc" + str(i))
                    else:
                        l_mission_idx = int(list(solu.iter_env.yard_cranes.values())[i].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num * 3 + 2) * 22 + (
                                J_num * 3 + 2) * 22 * p_mission_idx + 22 * l_mission_idx + 10 + i
                        # Y[p_mission_idx][l_mission_idx][7 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc" + str(i))

    MLP.update()
    MLP.setParam('OutputFlag', 0)
    # ============== 求解模型 ================
    MLP.write("output_result/gurobi/mod_" + str(inst_idx) + "_" + str(J_num) + tag + ".lp")
    MLP.Params.timelimit = 7200
    T1 = time.time()
    MLP.optimize()
    # print("time: " + str(time.time() - T1))
    if MLP.status == GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % MLP.status)
        # do IIS, find infeasible constraints
        MLP.computeIIS()
        for c in MLP.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        MLP.write("output_result/gurobi/sol_" + str(inst_idx) + "_" + str(J_num) + tag + ".ilp")
    MLP.write("output_result/gurobi/sol_" + str(inst_idx) + "_" + str(J_num) + tag + ".sol")
    # 非0变量及LS选择
    ls_ls = []
    m_ls = {}
    for var in MLP.getVars():
        if int(var.X) is not 0:
            tmp_str = var.VarName.split('_')
            # print(var.VarName + ": " + str(var.X))
            if tmp_str[0] == 'X' and 6 >= int(tmp_str[2]) >= 3 and int(tmp_str[1]) < J_num * 3:
                ls_ls.append(int(tmp_str[2]) - 3)
            if tmp_str[0] == 'Y':
                if m_ls.get(int(tmp_str[-1])) is None:
                    m_ls.setdefault(int(tmp_str[-1]), [int(tmp_str[1]), int(tmp_str[2]), -1])
                else:
                    tmp: list = m_ls.get(int(tmp_str[-1]))
                    # if tmp.count(int(tmp_str[1])) is not 0 and tmp.count(int(tmp_str[2])) is not 0:

                    if tmp.count(int(tmp_str[1])) is not 0:
                        tmp.insert(tmp.index(int(tmp_str[1])) + 1, int(tmp_str[2]))
                    elif tmp.count(int(tmp_str[2])) is not 0:
                        tmp.insert(tmp.index(int(tmp_str[2])), int(tmp_str[1]))
                    else:
                        tmp.extend([int(tmp_str[1]), int(tmp_str[2])])
                    # m_ls.update(var.VarName[-1], m_ls.get(var.VarName[-1]))
                    # m_ls.get(var.VarName[-1])
    # print(ls_ls)
    m_ls = sorted(m_ls.items(), key=lambda d: d[0], reverse=False)
    # for m in m_ls:
    # print(m)
    var_ls = [3 + i * 22 + (J_num * 3 + 2) * 22 for i in range(0, (J_num * 3 + 2) * (J_num * 3 + 2))]
    var_ls.extend([4 + i * 22 + (J_num * 3 + 2) * 22 for i in range(0, (J_num * 3 + 2) * (J_num * 3 + 2))])
    var_ls.extend([5 + i * 22 + (J_num * 3 + 2) * 22 for i in range(0, (J_num * 3 + 2) * (J_num * 3 + 2))])
    var_ls.extend([6 + i * 22 + (J_num * 3 + 2) * 22 for i in range(0, (J_num * 3 + 2) * (J_num * 3 + 2))])
    print('Solution:', [MLP.getVars()[i].VarName for i in var_ls if
                        MLP.getVars()[i].X != 0])
    return MLP


if __name__ == '__main__':
    inst_idx = 0
    J_num = cf.MISSION_NUM_ONE_QUAY_CRANE
    model = PortModel(J_num=J_num)
    model.construct()
    # print("current instance on run is:" + str(inst_idx) + "_" + str(J_num))
    data = Data(inst_idx=inst_idx, J_num=J_num)
    data.init_process()
    model.data = data
    model.construct_diff()
    s_t_g = time.time()
    MLP = solve_model(MLP=model.MLP, inst_idx=0, J_num=J_num, tag='origin')
    e_t_g = time.time()
    print("gurobi后makespan为" + str(MLP.getVars()[-1].X))
    print("gurobi算法时间" + str(e_t_g - s_t_g))

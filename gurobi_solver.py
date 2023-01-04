# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 10:01 PM
# @Author  : JacQ
# @File    : gurobi_solver.py
import time
from copy import deepcopy

from gurobipy import *  # 在Python中调用gurobi求解包
from algorithm_factory.algo_utils import sort_missions
from common import PortEnv
from common.iter_solution import IterSolution
from data_process.input_process import read_input
import conf.configs as cf


#
# class Data:
#     def __init__(self,
#                  inst_idx: int,
#                  J_num: int):
#         self.inst_idx = inst_idx  # 算例名称
#         self.J_num = J_num  # 任务数
#         self.S_num = 4  # 阶段数
#         self.J_num = J_num  # 任务数
#         self.K_num = 22  # 机器数
#         self.big_M = 10000  # 无穷大数
#         self.S = [0, 1, 2, 3]  # 阶段编号集合
#         self.K_s = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9],
#                     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]  # 每个阶段包括的机器编号集合
#         self.K = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 机器编号集合
#         self.J = []  # 任务编号集合
#         self.J_k = []  # 机器可处理任务集合
#         self.J_m = []  # 任务对应机器
#         self.A = []  # 顺序工件对集合
#         self.st = [[self.big_M for _ in range(self.J_num * 3 + 2)] for _ in range(self.J_num * 3 + 2)]  # 两两之间距离
#         self.pt = [[0 for _ in range(self.J_num * 3 + 2)] for _ in self.S]  # 每阶段处理任务所需时间
#         self.tt = [[[0 for _ in range(self.K_num)] for _ in range(self.K_num - 12)] for _ in
#                    range(self.J_num * 3 + 2)]
#         self.theta = [20, 25, 33.3, 50, 100]  # CO拥堵对应时间
#         self.instance = None
#         self.init_process()
#
#     def init_process(self):
#         instance = read_input('train', 0, 'A')
#         instance.l2a_init()
#         self.instance = instance
#         tmp_mission_ls = deepcopy(self.instance.init_env.mission_list)
#         getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
#         # J
#         self.J = [j for j in range(self.J_num * 3)]
#         self.J.append(self.J[-1] + 1)  # dummy job N+1
#         self.J.append(self.J[-1] + 1)  # dummy job N+2
#         # J_M
#         self.J_k = [[] for _ in range(self.K_num)]
#         for qc in instance.init_env.quay_cranes.values():
#             for mission in qc.missions.values():
#                 self.J_k[int(mission.quay_crane_id[-1]) - 1].append(int(mission.idx[1:]) - 1)  # QC 0-2
#                 self.J_k[int(mission.crossover_id[-1]) + 6].append(int(mission.idx[1:]) - 1)  # CO 7-9
#                 self.J_k[match_mission_yard_crane_num(mission)].append(int(mission.idx[1:]) - 1)  # YC 10-21
#                 self.J_m.append(match_mission_yard_crane_num(mission))  #
#                 # A
#                 # if int(mission.idx[1:]) % self.J_num == 1:
#                 #     self.A.append([self.J[-2], int(mission.idx[1:]) - 1])
#                 if int(mission.idx[1:]) != (int(qc.idx[-1])) * self.J_num:
#                     self.A.append([int(mission.idx[1:]) - 1, int(mission.idx[1:])])
#                 # else:
#                 #     self.A.append([int(mission.idx[1:]) - 1, self.J[-1]])
#
#                 # pt[s][j]
#                 self.pt[0][int(mission.idx[1:]) - 1] = mission.machine_process_time[0]
#                 self.pt[1][int(mission.idx[1:]) - 1] = mission.station_process_time
#                 self.pt[3][int(mission.idx[1:]) - 1] = mission.yard_crane_process_time
#         # self.pt[0][self.J[-1]] = 0
#         # self.pt[1][self.J[-1]] = 0
#         # self.pt[3][self.J[-1]] = 0
#         # self.pt[0][self.J[-2]] = 0
#         # self.pt[1][self.J[-2]] = 0
#         # self.pt[3][self.J[-2]] = 0
#
#         for m in self.J_k:
#             if any(m):
#                 m.append(self.J[-2])
#                 m.append(self.J[-1])
#
#         # st
#         for i in range(len(self.J) - 2):
#             mission_i = tmp_mission_ls[i]
#             for j in range(i, len(self.J)):
#                 if j == len(self.J) - 2:
#                     self.st[j][i] = abs(mission_i.yard_block_loc[1]) * cf.SLOT_LENGTH / cf.YARDCRANE_SPEED_X + \
#                                     abs(cf.SLOT_NUM_Y - mission_i.yard_block_loc[2]) * cf.SLOT_WIDTH \
#                                     / cf.YARDCRANE_SPEED_Y * 2
#                     self.st[j][-1] = 0
#                 elif j == len(self.J) - 1:
#                     self.st[i][j] = 0
#                 else:
#                     mission_j = tmp_mission_ls[j]
#                     if mission_i.yard_block_loc[0] == mission_j.yard_block_loc[0] and mission_i != mission_j:
#                         self.st[i][j] = abs(mission_i.yard_block_loc[1] - mission_j.yard_block_loc[1]) * cf.SLOT_LENGTH \
#                                         / cf.YARDCRANE_SPEED_X + \
#                                         abs(mission_j.yard_block_loc[2] - cf.SLOT_NUM_Y) * cf.SLOT_WIDTH \
#                                         / cf.YARDCRANE_SPEED_Y * 2
#                         self.st[j][i] = abs(mission_i.yard_block_loc[1] - mission_j.yard_block_loc[1]) * cf.SLOT_LENGTH \
#                                         / cf.YARDCRANE_SPEED_X + \
#                                         abs(mission_i.yard_block_loc[2] - cf.SLOT_NUM_Y) * cf.SLOT_WIDTH \
#                                         / cf.YARDCRANE_SPEED_Y * 2
#                     else:
#                         self.st[i][j] = self.big_M
#                         self.st[j][i] = self.st[i][j]
#         # tt
#         for j in range(len(self.J)):  # todo dummy
#             if j != len(self.J) - 1 and j != len(self.J) - 2:
#                 mission = tmp_mission_ls[j]
#                 for ls in range(4):
#                     self.tt[j][int(mission.quay_crane_id[-1]) - 1][ls + 3] = \
#                         self.instance.init_env.quay_cranes[mission.quay_crane_id].time_to_exit + \
#                         self.instance.init_env.exit_to_ls_matrix[ls] / mission.vehicle_speed
#                     self.tt[j][ls + 3][int(mission.crossover_id[-1]) + 6] = \
#                         self.instance.init_env.ls_to_co_matrix[ls][int(mission.crossover_id[-1]) - 1] \
#                         / mission.vehicle_speed
#                 self.tt[j][int(mission.crossover_id[-1]) + 6][match_mission_yard_crane_num(mission)] = \
#                     tmp_mission_ls[j].transfer_time_c2y
#             else:
#                 for qc in range(3):
#                     for ls in range(4):
#                         self.tt[j][qc][ls + 3] = 0
#                 for ls in range(4):
#                     for co in range(3):
#                         self.tt[j][ls + 3][co + 7] = 0
#                 for co in range(3):
#                     for yc in range(12):
#                         self.tt[j][co + 7][yc + 10] = 0

#
# class PortModel:
#     def __init__(self, J_num: int, data: Data = None):
#         self.S_num = 4  # 阶段数
#         self.J_num = J_num  # 任务数
#         self.M_num = 22  # 机器数
#         self.big_M = 10000  # 无穷大数
#         self.S = [0, 1, 2, 3]  # 阶段编号集合
#         self.M_S = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9],
#                     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]  # 每个阶段包括的机器编号集合
#         self.M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 机器编号集合
#         self.J = [j for j in range(self.J_num * 3)]  # 任务编号集合
#         self.J.append(self.J[-1] + 1)  # dummy job N+1
#         self.J.append(self.J[-1] + 1)  # dummy job N+2
#         self.theta = [20, 25, 33.3, 50, 100]  # CO拥堵对应时间
#
#         self.MLP = None
#         self.X = None  # X_jm
#         self.Y = None  # Y_jj'm
#         self.Z = None  # Z_jj'm
#         self.FT = None  # FTsj
#         self.OT = None  # OTsj
#         self.RT = None  # RTj
#         self.TT = None  # TTsj
#         self.ZZ = None  # ZZ[i][j]
#         self.alpha = None
#         self.beta = None
#
#         self.data = data
#
#     def construct(self):
#         # ============== 构造模型 ================
#         self.MLP = Model("port operation")
#
#         # ============== 定义变量 ================
#         self.X = [[[] for _ in self.M] for _ in self.J]
#         for j in self.J:
#             for m in self.M:
#                 name = 'X_' + str(j) + "_" + str(m)
#                 self.X[j][m] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
#         # Y_jj'm
#         self.Y = [[[[] for _ in self.M] for _ in self.J] for _ in self.J]
#         for j in self.J:
#             for jj in self.J:
#                 for m in self.M:
#                     name = 'Y_' + str(j) + "_" + str(jj) + "_" + str(m)
#                     self.Y[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
#         # Z_jj'm
#         self.Z = [[[[] for _ in self.M] for _ in self.J] for _ in self.J]
#         for j in self.J:
#             for jj in self.J:
#                 for m in self.M:
#                     name = 'Z_' + str(j) + "_" + str(jj) + "_" + str(m)
#                     self.Z[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
#         # FTsj
#         self.FT = [[[] for _ in self.J] for _ in self.S]
#         for s in self.S:
#             for j in self.J:
#                 name = 'FT_' + str(s) + "_" + str(j)
#                 self.FT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
#         # OTsj
#         self.OT = [[[] for _ in self.J] for _ in self.S]
#         for s in self.S:
#             for j in self.J:
#                 name = 'OT_' + str(s) + "_" + str(j)
#                 self.OT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
#         # RTj
#         self.RT = []
#         for j in self.J:
#             name = 'RT_' + str(j)
#             self.RT.append(self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name))
#         # TTsj
#         self.TT = [[[] for _ in self.J] for _ in self.S]
#         for s in self.S:
#             for j in self.J:
#                 name = 'TT_' + str(s) + "_" + str(j)
#                 self.TT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
#         # ZZ[i][j]
#         self.ZZ = [[0 for _ in self.J] for _ in range(5)]
#         for i in range(5):
#             for j in self.J:
#                 name = 'ZZ_' + str(i) + "_" + str(j)
#                 self.ZZ[i][j] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
#         # theta[s][j]
#         self.alpha = [[0 for _ in self.J] for _ in range(4)]
#         for i in range(4):
#             for j in self.J:
#                 name = 'alpha_' + str(i) + "_" + str(j)
#                 self.alpha[i][j] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
#         self.beta = [[0 for _ in self.J] for _ in range(4)]
#         for i in range(4):
#             for j in self.J:
#                 name = 'beta_' + str(i) + "_" + str(j)
#                 self.beta[i][j] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
#
#         # ============== 定义公共约束 ================
#         self.MLP.addConstrs((self.Y[j][j][m] == 0 for m in self.M for j in self.J), "con1")
#         self.MLP.addConstrs((self.Z[j][j][m] == 0 for m in self.M for j in self.J), "con1")
#         self.MLP.addConstrs((sum(self.X[j][m] for m in self.M_S[s]) == 1 for j in self.J[0:-2] for s in self.S), "con3")
#         self.MLP.addConstrs(
#             (self.FT[s + 1][j] >= self.OT[s + 1][j] + (self.FT[s][j] + self.TT[s][j]) for s in range(3) for j in
#              self.J), "con6")
#         self.MLP.addConstrs((self.FT[0][j] == self.RT[j] + self.OT[0][j] for j in self.J), "con7")  # fixme >=
#         self.MLP.addConstrs(
#             (self.FT[s][jj] + self.TT[s][jj] + self.big_M - self.big_M * self.Y[j][jj][m] >= self.FT[s][j] + self.TT[s][
#                 j] for j in self.J for jj in self.J for s in range(3) for m in self.M_S[s + 1]), "con8")
#
#         tmp_J = self.J[0:-2].copy()
#         tmp_J.append(self.J[-1])
#         self.MLP.addConstrs(
#             (sum(self.Y[jj][j][m] for jj in self.J) == self.X[j][m] for j in tmp_J for m in self.M), "con10")
#         self.MLP.addConstrs(
#             (sum(self.Y[j][jj][m] for jj in self.J) == self.X[j][m] for j in self.J[0:-1] for m in self.M), "con11")
#         # self.MLP.addConstrs((self.Y[j][jj][m] + self.Y[jj][j][m] <= 1 for j in self.J for jj in self.J for m in self.M),
#         #                     "con12")
#         # self.MLP.addConstrs(
#         #     (self.X[j][m] + self.X[jj][m] + self.big_M - self.big_M * self.Y[j][jj][m] >= 2 for j in self.J for jj in
#         #      self.J for m in self.M), "con9")
#         # self.MLP.addConstrs(
#         #     (self.FT[s][jj] - self.OT[s][jj] - self.FT[s][j] + self.big_M - self.big_M * self.Y[j][jj][m] >= 0 for s in
#         #      range(1, 4) for m in self.M_S[s] for j in self.J for jj in self.J), "con13")
#         for s in range(1, 4):
#             for m in self.M_S[s]:
#                 for j in self.J:
#                     for jj in self.J:
#                         self.MLP.addConstr(
#                             self.FT[s][jj] - self.OT[s][jj] - self.FT[s][j] + self.big_M - self.big_M *
#                             self.Y[j][jj][m] >= 0, "add13")
#
#         self.MLP.addConstrs(
#             (self.FT[s + 1][j] - self.OT[s + 1][j] <= self.FT[s][j] + self.TT[s][j] + self.big_M - self.big_M *
#              self.alpha[s + 1][j] for s in range(0, 3) for j in self.J[0:-2]), "con141")
#         self.MLP.addConstrs(
#             (self.FT[s][j] - self.OT[s][j] <= self.FT[s][jj] + 2 * self.big_M - self.big_M * self.beta[s][
#                 j] - self.big_M * self.Y[jj][j][m] for s in range(1, 4) for m in self.M_S[s] for j in self.J[0:-2]
#              for jj in self.J), "con142")
#         self.MLP.addConstrs((self.alpha[s][j] + self.beta[s][j] >= 1 for s in range(1, 4) for j in self.J[0:-2]),
#                             "con143")
#         # self.MLP.addConstrs(
#         #     (self.FT[s][j] - self.OT[s][j] <= self.FT[s][jj] + 1 * self.big_M + self.big_M * self.alpha[s][
#         #         j] - self.big_M * self.Y[jj][j][m] for s in range(1, 4) for m in self.M_S[s] for j in self.J[0:-2]
#         #      for jj in self.J), "con142")
#         self.MLP.addConstrs(
#             (self.FT[s + 1][j] + 3 * self.big_M >= (self.FT[s][jj] + self.TT[s][jj]) + self.big_M * self.Z[j][jj][
#                 m] + self.big_M * self.X[j][m] + self.big_M * self.X[jj][m] for j in self.J for jj in self.J for
#              s in range(3) for m in self.M_S[s + 1]), "con15")
#         self.MLP.addConstrs(
#             (self.FT[s][jj] + self.TT[s][jj] - self.FT[s][j] - self.TT[s][j] + 3 * self.big_M - self.big_M *
#              self.Z[j][jj][m] - self.big_M * self.X[j][m] - self.big_M * self.X[jj][m] >= 0 for j in self.J
#              for jj in self.J for s in range(3) for m in self.M_S[s + 1]), "con16")  # fixme
#         # self.MLP.addConstrs(
#         #     (self.X[j][m] + self.X[jj][m] + self.big_M - self.big_M * self.Z[j][jj][m] >= 2 for j in self.J for jj in
#         #      self.J for s in range(3) for m in self.M_S[s + 1]), "con17")
#         self.MLP.addConstrs(
#             (self.OT[2][j] == 5 + self.theta[0] * self.ZZ[0][j] + self.theta[1] * self.ZZ[1][j] + self.theta[2] *
#              self.ZZ[2][j] + self.theta[3] * self.ZZ[3][j] + self.theta[4] * self.ZZ[4][j] for j in self.J[0:-2]),
#             "con23")
#         self.MLP.addConstrs(
#             (sum(self.Z[jj][j][m] for jj in self.J for m in self.M_S[2]) <= self.ZZ[0][j] + 2 * self.ZZ[1][j] + 3 *
#              self.ZZ[2][j] + 4 * self.ZZ[3][j] + self.big_M * self.ZZ[4][j] for j in self.J[0:-2]), "con24")
#         self.MLP.addConstrs(
#             (sum(self.Z[jj][j][m] for jj in self.J for m in self.M_S[2]) >= 2 * self.ZZ[1][j] + 3 * self.ZZ[2][j] + 4 *
#              self.ZZ[3][j] for j in self.J[0:-2]), "con25")
#         self.MLP.addConstrs((self.OT[s][-1] == 0 for s in self.S), "con180")
#         self.MLP.addConstrs((self.OT[s][-2] == 0 for s in self.S), "con180")
#         self.MLP.addConstrs((self.TT[s][self.J[-1]] == 0 for s in range(0, 3)), "con21")
#         self.MLP.addConstrs((self.TT[s][self.J[-2]] == 0 for s in range(0, 3)), "con21")
#
#         # ============== 构造目标 ================
#         q = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='q')  # 线性化模型变量
#         self.MLP.addConstrs((q >= self.FT[3][j] for j in self.J), "obj1")
#         self.MLP.setObjective(q,
#                               GRB.MINIMIZE)  # fixme+ 0.000001 * sum(self.FT[s][j] for s in self.S for j in self.J)
#
#         # ============== 定义差异约束 ================
#         tmp_mission_ls = deepcopy(self.data.instance.init_env.mission_list)
#         getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
#         self.MLP.addConstrs((self.X[j][m] == 1 for m in self.M for j in self.data.J_k[m]), "con2")
#         for pair in self.data.A:
#             j, jj = pair[0], pair[1]
#             self.MLP.addConstr(self.RT[jj] - self.FT[0][j] >= 0, "con4" + str(j) + str(jj))
#             m = int(tmp_mission_ls[j].quay_crane_id[-1]) - 1
#             self.MLP.addConstr(self.Y[j][jj][m] == 1, "con5" + str(j) + str(jj))
#             # self.MLP.addConstr(self.RT[jj] - self.RT[j] == 60, "con00" + str(j) + str(jj))  # FIXME: match RL
#         self.MLP.addConstr(self.RT[-2] == 0, "con00")
#         # for i in range(3):
#         #     self.MLP.addConstr(self.RT[i * self.J_num] == 0, "con00")  # FIXME: match RL
#         self.MLP.addConstrs((self.OT[s][j] == self.data.pt[s][j] for s in range(0, 2) for j in self.J[0:-2]), "con181")
#         self.MLP.addConstrs((self.OT[3][j] == self.data.pt[3][j] + sum(
#             self.Y[jj][j][self.data.J_m[j]] * self.data.st[jj][j] for jj in self.J) for j in self.J[0:-2]), "con183")
#         self.MLP.addConstrs(
#             (self.TT[s][j] - self.data.tt[j][m][mm] + self.big_M * 2 - self.big_M * self.X[j][m] - self.big_M *
#              self.X[j][mm] >= 0 for j in self.J[0:-2] for s in range(0, 3) for m in self.M_S[s] for mm in
#              self.M_S[s + 1]), "con19")
#         self.MLP.addConstrs(
#             (self.TT[s][j] - self.data.tt[j][m][mm] <= self.big_M * 2 - self.big_M * self.X[j][m] - self.big_M *
#              self.X[j][mm] for j in self.J[0:-2] for s in range(0, 3) for m in self.M_S[s] for mm in self.M_S[s + 1]),
#             "con20")
#         self.MLP.update()
#
#
# class PortModelRLP:
#     def __init__(self, J_num: int, data: Data = None):
#         self.S_num = 4  # 阶段数
#         self.J_num = J_num  # 任务数
#         self.M_num = 22  # 机器数
#         self.big_M = 10000  # 无穷大数
#         self.S = [0, 1, 2, 3]  # 阶段编号集合
#         self.M_S = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9],
#                     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]  # 每个阶段包括的机器编号集合
#         self.M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 机器编号集合
#         self.J = [j for j in range(self.J_num * 3)]  # 任务编号集合
#         self.J.append(self.J[-1] + 1)  # dummy job N+1
#         self.J.append(self.J[-1] + 1)  # dummy job N+2
#         self.theta = [20, 25, 33.3, 50, 100]  # CO拥堵对应时间
#
#         self.MLP = None
#         self.X = None  # X_jm
#         self.Y = None  # Y_jj'm
#         self.Z = None  # Z_jj'm
#         self.FT = None  # FTsj
#         self.OT = None  # OTsj
#         self.RT = None  # RTj
#         self.TT = None  # TTsj
#         self.ZZ = None  # ZZ[i][j]
#         self.alpha = None
#         self.beta = None
#
#         self.data = data
#
#     def construct(self):
#         # ============== 构造模型 ================
#         self.MLP = Model("port operation")
#
#         # ============== 定义变量 ================
#         self.X = [[[] for _ in self.M] for _ in self.J]
#         for j in self.J:
#             for m in self.M:
#                 name = 'X_' + str(j) + "_" + str(m)
#                 self.X[j][m] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
#         # Y_jj'm
#         self.Y = [[[[] for _ in self.M] for _ in self.J] for _ in self.J]
#         for j in self.J:
#             for jj in self.J:
#                 for m in self.M:
#                     if 3 <= m <= 6:
#                         name = 'Y_' + str(j) + "_" + str(jj) + "_" + str(m)
#                         self.Y[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
#                     else:
#                         name = 'Y_' + str(j) + "_" + str(jj) + "_" + str(m)
#                         self.Y[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
#         # Z_jj'm
#         self.Z = [[[[] for _ in self.M] for _ in self.J] for _ in self.J]
#         for j in self.J:
#             for jj in self.J:
#                 for m in self.M:
#                     name = 'Z_' + str(j) + "_" + str(jj) + "_" + str(m)
#                     self.Z[j][jj][m] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
#         # FTsj
#         self.FT = [[[] for _ in self.J] for _ in self.S]
#         for s in self.S:
#             for j in self.J:
#                 name = 'FT_' + str(s) + "_" + str(j)
#                 self.FT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
#         # OTsj
#         self.OT = [[[] for _ in self.J] for _ in self.S]
#         for s in self.S:
#             for j in self.J:
#                 name = 'OT_' + str(s) + "_" + str(j)
#                 self.OT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
#         # RTj
#         self.RT = []
#         for j in self.J:
#             name = 'RT_' + str(j)
#             self.RT.append(self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name))
#         # TTsj
#         self.TT = [[[] for _ in self.J] for _ in self.S]
#         for s in self.S:
#             for j in self.J:
#                 name = 'TT_' + str(s) + "_" + str(j)
#                 self.TT[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
#         # ZZ[i][j]
#         self.ZZ = [[0 for _ in self.J] for _ in range(5)]
#         for i in range(5):
#             for j in self.J:
#                 name = 'ZZ_' + str(i) + "_" + str(j)
#                 self.ZZ[i][j] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
#         # theta[s][j]
#         self.alpha = [[0 for _ in self.J] for _ in range(4)]
#         for i in range(4):
#             for j in self.J:
#                 name = 'alpha_' + str(i) + "_" + str(j)
#                 self.alpha[i][j] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
#         self.beta = [[0 for _ in self.J] for _ in range(4)]
#         for i in range(4):
#             for j in self.J:
#                 name = 'beta_' + str(i) + "_" + str(j)
#                 self.beta[i][j] = self.MLP.addVar(0, 1, vtype=GRB.CONTINUOUS, name=name)
#
#         # ============== 定义公共约束 ================
#         self.MLP.addConstrs((self.Y[j][j][m] == 0 for m in self.M for j in self.J), "con1")
#         self.MLP.addConstrs((self.Y[j][self.J[-2]][m] == 0 for m in self.M for j in self.J), "con1")
#         self.MLP.addConstrs((self.Y[self.J[-1]][j][m] == 0 for m in self.M for j in self.J), "con1")
#         self.MLP.addConstrs((self.Z[j][j][m] == 0 for m in self.M for j in self.J), "con1")
#         self.MLP.addConstrs((sum(self.X[j][m] for m in self.M_S[s]) == 1 for j in self.J[0:-2] for s in self.S), "con3")
#         self.MLP.addConstrs(
#             (self.FT[s + 1][j] >= self.OT[s + 1][j] + (self.FT[s][j] + self.TT[s][j]) for s in range(3) for j in
#              self.J), "con6")
#         self.MLP.addConstrs((self.FT[0][j] == self.RT[j] + self.OT[0][j] for j in self.J), "con7")  # fixme >=
#         self.MLP.addConstrs(
#             (self.FT[s][jj] + self.TT[s][jj] + self.big_M - self.big_M * self.Y[j][jj][m] >= self.FT[s][j] + self.TT[s][
#                 j] for j in self.J for jj in self.J for s in range(3) for m in self.M_S[s + 1]), "con8")
#
#         tmp_J = self.J[0:-2].copy()
#         tmp_J.append(self.J[-1])
#         self.MLP.addConstrs(
#             (sum(self.Y[jj][j][m] for jj in self.J) == self.X[j][m] for j in tmp_J for m in self.M), "con10")
#         self.MLP.addConstrs(
#             (sum(self.Y[j][jj][m] for jj in self.J) == self.X[j][m] for j in self.J[0:-1] for m in self.M), "con11")
#         # self.MLP.addConstrs((self.Y[j][jj][m] + self.Y[jj][j][m] <= 1 for j in self.J for jj in self.J for m in self.M),
#         #                     "con12")
#         # self.MLP.addConstrs(
#         #     (self.X[j][m] + self.X[jj][m] + self.big_M - self.big_M * self.Y[j][jj][m] >= 2 for j in self.J for jj in
#         #      self.J for m in self.M), "con9")
#         # self.MLP.addConstrs(
#         #     (self.FT[s][jj] - self.OT[s][jj] - self.FT[s][j] + self.big_M - self.big_M * self.Y[j][jj][m] >= 0 for s in
#         #      range(1, 4) for m in self.M_S[s] for j in self.J for jj in self.J), "con13")
#         for s in range(1, 4):
#             for m in self.M_S[s]:
#                 for j in self.J:
#                     for jj in self.J:
#                         self.MLP.addConstr(
#                             self.FT[s][jj] - self.OT[s][jj] - self.FT[s][j] + self.big_M - self.big_M *
#                             self.Y[j][jj][m] >= 0, "add13")
#
#         self.MLP.addConstrs(
#             (self.FT[s + 1][j] - self.OT[s + 1][j] <= self.FT[s][j] + self.TT[s][j] + self.big_M - self.big_M *
#              self.alpha[s + 1][j] for s in range(0, 3) for j in self.J[0:-2]), "con141")
#         self.MLP.addConstrs(
#             (self.FT[s][j] - self.OT[s][j] <= self.FT[s][jj] + 2 * self.big_M - self.big_M * self.beta[s][
#                 j] - self.big_M * self.Y[jj][j][m] for s in range(1, 4) for m in self.M_S[s] for j in self.J[0:-2]
#              for jj in self.J), "con142")
#         self.MLP.addConstrs((self.alpha[s][j] + self.beta[s][j] >= 1 for s in range(1, 4) for j in self.J[0:-2]),
#                             "con143")
#         self.MLP.addConstrs(
#             (self.FT[s + 1][j] + 3 * self.big_M >= (self.FT[s][jj] + self.TT[s][jj]) + self.big_M * self.Z[j][jj][
#                 m] + self.big_M * self.X[j][m] + self.big_M * self.X[jj][m] for j in self.J for jj in self.J for
#              s in range(3) for m in self.M_S[s + 1]), "con15")
#         self.MLP.addConstrs(
#             (self.FT[s][jj] + self.TT[s][jj] - self.FT[s][j] - self.TT[s][j] + 3 * self.big_M - self.big_M *
#              self.Z[j][jj][m] - self.big_M * self.X[j][m] - self.big_M * self.X[jj][m] >= 0 for j in self.J
#              for jj in self.J for s in range(3) for m in self.M_S[s + 1]), "con16")  # fixme
#         # self.MLP.addConstrs(
#         #     (self.X[j][m] + self.X[jj][m] + self.big_M - self.big_M * self.Z[j][jj][m] >= 2 for j in self.J for jj in
#         #      self.J for s in range(3) for m in self.M_S[s + 1]), "con17")
#         self.MLP.addConstrs(
#             (self.OT[2][j] == 5 + self.theta[0] * self.ZZ[0][j] + self.theta[1] * self.ZZ[1][j] + self.theta[2] *
#              self.ZZ[2][j] + self.theta[3] * self.ZZ[3][j] + self.theta[4] * self.ZZ[4][j] for j in self.J[0:-2]),
#             "con23")
#         self.MLP.addConstrs(
#             (sum(self.Z[jj][j][m] for jj in self.J for m in self.M_S[2]) <= self.ZZ[0][j] + 2 * self.ZZ[1][j] + 3 *
#              self.ZZ[2][j] + 4 * self.ZZ[3][j] + self.big_M * self.ZZ[4][j] for j in self.J[0:-2]), "con24")
#         self.MLP.addConstrs(
#             (sum(self.Z[jj][j][m] for jj in self.J for m in self.M_S[2]) >= 2 * self.ZZ[1][j] + 3 * self.ZZ[2][j] + 4 *
#              self.ZZ[3][j] for j in self.J[0:-2]), "con25")
#         self.MLP.addConstrs((self.OT[s][-1] == 0 for s in self.S), "con180")
#         self.MLP.addConstrs((self.OT[s][-2] == 0 for s in self.S), "con180")
#         self.MLP.addConstrs((self.TT[s][self.J[-1]] == 0 for s in range(0, 3)), "con21")
#         self.MLP.addConstrs((self.TT[s][self.J[-2]] == 0 for s in range(0, 3)), "con21")
#
#         # ============== 构造目标 ================
#         q = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='q')  # 线性化模型变量
#         self.MLP.addConstrs((q >= self.FT[3][j] for j in self.J), "obj1")
#         self.MLP.setObjective(q,
#                               GRB.MINIMIZE)  # fixme+ 0.000001 * sum(self.FT[s][j] for s in self.S for j in self.J)
#
#         # ============== 定义差异约束 ================
#         tmp_mission_ls = deepcopy(self.data.instance.init_env.mission_list)
#         getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
#         self.MLP.addConstrs((self.X[j][m] == 1 for m in self.M for j in self.data.J_k[m]), "con2")
#         for pair in self.data.A:
#             j, jj = pair[0], pair[1]
#             self.MLP.addConstr(self.RT[jj] - self.FT[0][j] >= 0, "con4" + str(j) + str(jj))
#             m = int(tmp_mission_ls[j].quay_crane_id[-1]) - 1
#             self.MLP.addConstr(self.Y[j][jj][m] == 1, "con5" + str(j) + str(jj))
#             # self.MLP.addConstr(self.RT[jj] - self.RT[j] == 120, "con00" + str(j) + str(jj))  # FIXME: match RL
#         self.MLP.addConstr(self.RT[-2] == 0, "con00")
#         # for i in range(3):
#         #     self.MLP.addConstr(self.RT[i * self.J_num] == 0, "con00")  # FIXME: match RL
#         self.MLP.addConstrs((self.OT[s][j] == self.data.pt[s][j] for s in range(0, 2) for j in self.J[0:-2]), "con181")
#         self.MLP.addConstrs((self.OT[3][j] == self.data.pt[3][j] + sum(
#             self.Y[jj][j][self.data.J_m[j]] * self.data.st[jj][j] for jj in self.J) for j in self.J[0:-2]), "con183")
#         self.MLP.addConstrs(
#             (self.TT[s][j] - self.data.tt[j][m][mm] + self.big_M * 2 - self.big_M * self.X[j][m] - self.big_M *
#              self.X[j][mm] >= 0 for j in self.J[0:-2] for s in range(0, 3) for m in self.M_S[s] for mm in
#              self.M_S[s + 1]), "con19")
#         self.MLP.addConstrs(
#             (self.TT[s][j] - self.data.tt[j][m][mm] <= self.big_M * 2 - self.big_M * self.X[j][m] - self.big_M *
#              self.X[j][mm] for j in self.J[0:-2] for s in range(0, 3) for m in self.M_S[s] for mm in self.M_S[s + 1]),
#             "con20")
#         self.MLP.update()
#

class CongestionPortModel:
    def __init__(self, solu: IterSolution):
        self.env = solu.init_env
        self.solu = solu
        self.S_num = 4  # 阶段数
        self.J_num = self.env.J_num  # 任务数
        self.J_num_all = self.env.J_num_all
        self.K_num = -1  # 机器数
        self.big_M = 10000  # 无穷大数
        self.S = [0, 1, 2, 3]  # 阶段编号集合
        self.K_s = []  # 每个阶段包括的机器编号集合
        self.K = []  # 机器编号集合
        self.J = []  # 任务编号集合
        self.J_k = []  # 机器可处理任务集合
        self.A = []  # 顺序工件对集合
        self.st = [[self.big_M for _ in range(self.J_num_all + 2)] for _ in range(self.J_num_all + 2)]  # 两两之间距离
        self.pt = [[0 for _ in range(self.J_num_all + 2)] for _ in self.S]  # 每阶段处理任务所需时间
        self.tt = [[[0 for _ in range(self.env.machine_num)] for _ in range(self.env.machine_num - self.env.qc_num)] for
                   _ in range(self.J_num_all + 2)]
        self.init_model()

        self.MLP = None
        self.v = None  # v_jk
        self.x = None  # x_ijk
        self.C = None  # C^s_j
        self.o = None  # o^s_j
        self.r = None  # r_j
        self.u = None  # u^s_j
        self.alpha = None
        self.beta = None

    def init_model(self):
        self.K_num = self.env.machine_num
        self.big_M = 100000  # 无穷大数
        self.K_s = [[i for i in range(self.env.qc_num)],
                    [i + self.env.qc_num for i in range(self.env.ls_num)],
                    [i + self.env.qc_num + self.env.ls_num for i in range(self.env.is_num)],
                    [i + self.env.qc_num + self.env.ls_num + self.env.is_num for i in
                     range(self.env.yc_num)]]  # 每个阶段包括的机器编号集合
        self.K = [i for i in range(self.env.machine_num)]  # 机器编号集合
        self.J = [j for j in range(self.J_num_all)]  # 任务编号集合
        self.J.append(self.J[-1] + 1)  # dummy job N+1
        self.J.append(self.J[-1] + 1)  # dummy job N+2

        # self.solu.l2a_init()
        tmp_mission_ls = deepcopy(self.env.mission_list)
        getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
        # J_K
        self.J_k = [[] for _ in range(self.K_num)]  # QC 0-2,CO 7-9, YC 10-21
        for qc in self.env.quay_cranes.values():
            for mission in qc.missions.values():
                mission_idx = int(mission.idx[1:]) - 1
                qc_idx = self.env.machine_name_to_idx[mission.quay_crane_id]
                is_idx = self.env.machine_name_to_idx[mission.crossover_id]
                yc_idx = self.env.machine_name_to_idx['YC' + mission.yard_block_loc[0]]
                self.J_k[qc_idx].append(mission_idx)
                self.J_k[is_idx].append(mission_idx)
                self.J_k[yc_idx].append(mission_idx)
                # A
                if (mission_idx + 1) != sum(self.J_num[0:qc_idx + 1]):
                    self.A.append([mission_idx, mission_idx + 1])
                # pt[s][j]
                self.pt[0][mission_idx] = mission.machine_process_time[0]
                self.pt[1][mission_idx] = mission.station_process_time
                self.pt[2][mission_idx] = mission.intersection_process_time
                self.pt[3][mission_idx] = mission.yard_crane_process_time
        for k in self.J_k:
            if len(k) > 0:
                k.append(self.J[-2])
                k.append(self.J[-1])
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
                for ls in range(self.env.ls_num):
                    self.tt[j][self.env.machine_name_to_idx[mission.quay_crane_id]][
                        self.env.machine_name_to_idx['S' + str(ls + 1)]] = \
                        self.env.quay_cranes[mission.quay_crane_id].time_to_exit + \
                        self.env.exit_to_ls_matrix[ls] / mission.vehicle_speed
                    self.tt[j][self.env.machine_name_to_idx['S' + str(ls + 1)]][
                        self.env.machine_name_to_idx[mission.crossover_id]] = \
                        self.env.ls_to_co_matrix[ls][int(mission.crossover_id[-1]) - 1] / mission.vehicle_speed
                self.tt[j][self.env.machine_name_to_idx[mission.crossover_id]][
                    self.env.machine_name_to_idx['YC' + mission.yard_block_loc[0]]] = tmp_mission_ls[
                    j].transfer_time_c2y
            else:
                for qc in range(self.env.qc_num):
                    for ls in range(self.env.ls_num):
                        self.tt[j][qc][ls + self.env.qc_num] = 0
                for ls in range(self.env.ls_num):
                    for co in range(self.env.is_num):
                        self.tt[j][ls + self.env.qc_num][co + self.env.qc_num + self.env.ls_num] = 0
                for co in range(self.env.is_num):
                    for yc in range(self.env.yc_num):
                        self.tt[j][co + self.env.qc_num + self.env.ls_num][
                            yc + self.env.qc_num + self.env.ls_num + self.env.is_num] = 0

    def construct(self):
        # ============== 构造模型 ================
        self.MLP = Model("port operation")

        # ============== 定义变量 ================
        # v_jk
        self.v = [[[] for _ in self.K] for _ in self.J]
        for j in self.J:
            for k in self.K:
                name = 'v_' + str(j) + "_" + str(k)
                self.v[j][k] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        # x_ijk
        self.x = [[[[] for _ in self.K] for _ in self.J] for _ in self.J]
        for j in self.J:
            for jj in self.J:
                for k in self.K:
                    name = 'x_' + str(j) + "_" + str(jj) + "_" + str(k)
                    self.x[j][jj][k] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        # C^s_j
        self.C = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'C_' + str(s) + "_" + str(j)
                self.C[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # o^s_j
        self.o = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'o_' + str(s) + "_" + str(j)
                self.o[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # r_j
        self.r = []
        for j in self.J:
            name = 'r_' + str(j)
            self.r.append(self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name))
        # u^s_j
        self.u = [[[] for _ in self.J] for _ in self.S]
        for s in self.S:
            for j in self.J:
                name = 'u_' + str(s) + "_" + str(j)
                self.u[s][j] = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
        # alpha
        self.alpha = [[0 for _ in self.J] for _ in range(4)]
        for i in range(4):
            for j in self.J:
                name = 'alpha_' + str(i) + "_" + str(j)
                self.alpha[i][j] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)
        # beta
        self.beta = [[0 for _ in self.J] for _ in range(4)]
        for i in range(4):
            for j in self.J:
                name = 'beta_' + str(i) + "_" + str(j)
                self.beta[i][j] = self.MLP.addVar(0, 1, vtype=GRB.BINARY, name=name)

        # ============== 定义公共约束 ================
        self.MLP.addConstrs((self.x[j][j][k] == 0 for k in self.K for j in self.J), "con0")
        self.MLP.addConstrs((sum(self.v[j][k] for k in self.K_s[s]) == 1 for j in self.J[0:-2] for s in self.S), "con3")
        self.MLP.addConstrs((self.C[s + 1][j] >=
                             self.o[s + 1][j] + self.C[s][j] + self.u[s][j] for s in range(3) for j in self.J), "con4")
        for s in range(1, 4):
            for k in self.K_s[s]:
                for j in self.J:
                    for jj in self.J:
                        self.MLP.addConstr(
                            self.C[s][jj] - self.o[s][jj] - self.C[s][j] + self.big_M - self.big_M *
                            self.x[j][jj][k] >= 0, "con5")
        self.MLP.addConstrs((self.C[0][j] == self.r[j] + self.o[0][j] for j in self.J), "con9")
        tmp_J = self.J[0:-2].copy()
        tmp_J.append(self.J[-1])
        self.MLP.addConstrs(
            (sum(self.x[jj][j][k] for jj in self.J) == self.v[j][k] for j in tmp_J for k in self.K), "con10")
        self.MLP.addConstrs(
            (sum(self.x[j][jj][k] for jj in self.J) == self.v[j][k] for j in self.J[0:-1] for k in self.K), "con11")
        self.MLP.addConstrs(
            (self.C[s][jj] + self.u[s][jj] + self.big_M - self.big_M * self.x[j][jj][k] >= self.C[s][j] + self.u[s][
                j] for j in self.J for jj in self.J for s in range(3) for k in self.K_s[s + 1]), "con12")
        self.MLP.addConstrs(
            (self.C[s + 1][j] - self.o[s + 1][j] <= self.C[s][j] + self.u[s][j] + self.big_M - self.big_M *
             self.alpha[s + 1][j] for s in range(0, 3) for j in self.J[0:-2]), "con14")
        self.MLP.addConstrs(
            (self.C[s][j] - self.o[s][j] <= self.C[s][jj] + 2 * self.big_M - self.big_M * self.beta[s][
                j] - self.big_M * self.x[jj][j][k] for s in range(1, 4) for k in self.K_s[s] for j in self.J[0:-2]
             for jj in self.J), "con15")
        self.MLP.addConstrs((self.alpha[s][j] + self.beta[s][j] >= 1 for s in range(1, 4) for j in self.J[0:-2]),
                            "con16")
        self.MLP.addConstrs((self.o[s][-1] == 0 for s in self.S), "con202")
        self.MLP.addConstrs((self.o[s][-2] == 0 for s in self.S), "con203")
        self.MLP.addConstrs((self.u[s][self.J[-1]] == 0 for s in range(0, 3)), "con190")
        self.MLP.addConstrs((self.u[s][self.J[-2]] == 0 for s in range(0, 3)), "con191")

        # ============== 定义差异约束 ================
        tmp_mission_ls = deepcopy(self.env.mission_list)
        getattr(sort_missions, 'CHAR_ORDER')(tmp_mission_ls)
        self.MLP.addConstrs((self.v[j][k] == 1 for k in self.K for j in self.J_k[k]), "con8")
        for pair in self.A:
            j, jj = pair[0], pair[1]
            self.MLP.addConstr(self.r[jj] - self.C[0][j] >= 0, "con10" + str(j) + str(jj))
            k = int(tmp_mission_ls[j].quay_crane_id[-1]) - 1
            self.MLP.addConstr(self.x[j][jj][k] == 1, "con11" + str(j) + str(jj))
            # self.MLP.addConstr(self.r[jj] - self.r[j] == 120, "con00" + str(j) + str(jj))  # FIXME: match RL
        self.MLP.addConstr(self.r[-2] == 0, "con00")
        # for i in range(self.env.qc_num):
            # self.MLP.addConstr(self.r[sum(self.J_num[0:i])] == 0, "con00")  # FIXME: match RL
        self.MLP.addConstrs((self.o[s][j] == self.pt[s][j] for s in range(0, 3) for j in self.J[0:-2]), "con200")
        self.MLP.addConstrs((self.o[3][j] == self.pt[3][j] + sum(
            self.x[jj][j][k] * self.st[jj][j] for jj in self.J for k in self.K_s[3]) for j in self.J[0:-2]), "con201")
        self.MLP.addConstrs(
            (self.u[s][j] - self.tt[j][k][kk] + self.big_M * 2 - self.big_M * self.v[j][k] - self.big_M *
             self.v[j][kk] >= 0 for j in self.J[0:-2] for s in range(0, 3) for k in self.K_s[s] for kk in
             self.K_s[s + 1]), "con17")
        self.MLP.addConstrs(
            (self.u[s][j] - self.tt[j][k][kk] <= self.big_M * 2 - self.big_M * self.v[j][k] - self.big_M *
             self.v[j][kk] for j in self.J[0:-2] for s in range(0, 3) for k in self.K_s[s] for kk in self.K_s[s + 1]),
            "con18")

        # ============== 构造目标 ================
        q_1 = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='q_1')  # 线性化模型变量
        q_2 = self.MLP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='q_2')  # 线性化模型变量
        self.MLP.addConstrs((q_1 >= self.C[3][j] for j in self.J), "obj1")
        self.MLP.addConstr((q_2 == sum(self.C[s + 1][j] - self.o[s + 1][j]
                                       - self.C[s][j] - self.u[s][j] for j in tmp_J for s in range(0, 3))), "obj2")
        self.MLP.setObjective(q_1 + 0.000001 * sum(self.C[s][j] for j in tmp_J for s in self.S),
                              GRB.MINIMIZE)  # fixme+ 0.000001 * sum(self.FT[s][j] for s in self.S for j in self.J)
        self.MLP.update()


def solve_model(MLP, inst_idx, solved_solu: IterSolution = None, tag='', X_flag=True, Y_flag=True, epsilon=0.9):
    vars = MLP.getVars()
    machine_num = solved_solu.iter_env.machine_num
    J_num_all = solved_solu.iter_env.J_num_all
    # ============== 输入解 ================
    if solved_solu is not None:
        ls = solved_solu.iter_env.mission_list
        # fix v_jk
        if X_flag:
            for j in range(len(solved_solu.iter_env.mission_list)):
                var_idx = (int(ls[j].idx[1:]) - 1) * machine_num + int(
                    ls[j].machine_list[4][-1]) + solved_solu.iter_env.qc_num - 1
                MLP.addConstr((vars[var_idx] == 1), "fixed_x" + str(j))
        # fix Y_ijk s=2
        if Y_flag:
            for i in range(solved_solu.iter_env.ls_num):
                ls_idx = solved_solu.init_env.machine_name_to_idx['S' + str(i + 1)]
                for j in range(len(solved_solu.iter_env.lock_stations['S' + str(i + 1)].mission_list)):
                    p_mission_idx = int(
                        solved_solu.iter_env.lock_stations['S' + str(i + 1)].mission_list[j].idx[1:]) - 1
                    if j == 0 and j == len(solved_solu.iter_env.lock_stations['S' + str(i + 1)].mission_list) - 1:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * J_num_all + \
                                  machine_num * p_mission_idx + ls_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls_" + str(i))
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * (J_num_all + 1) + ls_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls_" + str(i))
                    elif j == 0:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * J_num_all + \
                                  machine_num * p_mission_idx + ls_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls_" + str(i))
                        l_mission_idx = \
                            int(solved_solu.iter_env.lock_stations['S' + str(i + 1)].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * l_mission_idx + ls_idx
                        # Y[p_mission_idx][l_mission_idx][3 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls_" + str(i))
                    elif j == len(solved_solu.iter_env.lock_stations['S' + str(i + 1)].mission_list) - 1:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * (J_num_all + 1) + ls_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls_" + str(i))
                    else:
                        l_mission_idx = int(
                            solved_solu.iter_env.lock_stations['S' + str(i + 1)].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * l_mission_idx + ls_idx
                        # Y[p_mission_idx][l_mission_idx][3 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_ls_" + str(i))
                    # print(vars[var_idx])
            for i in range(len(solved_solu.iter_env.crossovers)):
                co_idx = solved_solu.init_env.machine_name_to_idx['CO' + str(i + 1)]
                for j in range(len(solved_solu.iter_env.crossovers['CO' + str(i + 1)].mission_list)):
                    p_mission_idx = int(solved_solu.iter_env.crossovers['CO' + str(i + 1)].mission_list[j].idx[1:]) - 1
                    if j == 0 and j == len(solved_solu.iter_env.crossovers['CO' + str(i + 1)].mission_list) - 1:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * J_num_all + \
                                  machine_num * p_mission_idx + co_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co_" + str(i))
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * (J_num_all + 1) + co_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co_" + str(i))
                    elif j == 0:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * J_num_all + \
                                  machine_num * p_mission_idx + co_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co_" + str(i))
                        l_mission_idx = int(
                            solved_solu.iter_env.crossovers['CO' + str(i + 1)].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * l_mission_idx + co_idx
                        # Y[p_mission_idx][l_mission_idx][7 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co_" + str(i))
                    elif j == len(solved_solu.iter_env.crossovers['CO' + str(i + 1)].mission_list) - 1:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * (J_num_all + 1) + co_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co_" + str(i))
                    else:
                        l_mission_idx = int(
                            solved_solu.iter_env.crossovers['CO' + str(i + 1)].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * l_mission_idx + co_idx
                        # Y[p_mission_idx][l_mission_idx][7 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_co_" + str(i))
                    # print(vars[var_idx])
            for yc in solved_solu.init_env.yard_cranes_set:
                yc_idx = solved_solu.init_env.machine_name_to_idx['YC' + yc]
                for j in range(len(solved_solu.iter_env.yard_cranes['YC' + yc].mission_list)):
                    p_mission_idx = int(solved_solu.iter_env.yard_cranes['YC' + yc].mission_list[j].idx[1:]) - 1
                    if j == 0 and j == len(solved_solu.iter_env.yard_cranes['YC' + yc].mission_list) - 1:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * J_num_all + \
                                  machine_num * p_mission_idx + yc_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc_" + yc)
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * (J_num_all + 1) + yc_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc_" + yc)
                    elif j == 0:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * J_num_all + \
                                  machine_num * p_mission_idx + yc_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc_" + yc)
                        l_mission_idx = int(solved_solu.iter_env.yard_cranes['YC' + yc].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * l_mission_idx + yc_idx
                        # Y[p_mission_idx][l_mission_idx][7 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc_" + yc)
                    elif j == len(solved_solu.iter_env.yard_cranes['YC' + yc].mission_list) - 1:
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * (J_num_all + 1) + yc_idx
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc_" + yc)
                    else:
                        l_mission_idx = int(solved_solu.iter_env.yard_cranes['YC' + yc].mission_list[j + 1].idx[1:]) - 1
                        var_idx = (J_num_all + 2) * machine_num + (J_num_all + 2) * machine_num * p_mission_idx + \
                                  machine_num * l_mission_idx + yc_idx
                        # Y[p_mission_idx][l_mission_idx][7 + i]
                        MLP.addConstr(vars[var_idx] == 1, "fixed_qc_" + yc)
                    # print(vars[var_idx])
    # max_q_2 = 53348.4599303192
    # min_q_2 = 2401.81184668972
    # MLP.addConstr((vars[-1] <= epsilon * (max_q_2 - min_q_2) + min_q_2), "multi-objectives")
    MLP.update()
    # MLP.setParam('OutputFlag', 0)
    # ============== 求解模型 ================
    MLP.write("output_result/gurobi/mod_" + str(inst_idx) + "_" + tag + ".lp")
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
        MLP.write("output_result/gurobi/sol_" + str(inst_idx) + "_" + tag + ".ilp")
    MLP.write("output_result/gurobi/sol_" + str(inst_idx) + "_" + tag + ".sol")
    # 非0变量及LS选择
    ls_ls = []
    m_ls = {}
    f_2 = 0
    for var in MLP.getVars():
        if int(var.x) is not 0:
            tmp_str = var.VarName.split('_')
            # print(var.VarName + ": " + str(var.X))
            if tmp_str[0] == 'v' and solved_solu.iter_env.ls_num + solved_solu.iter_env.qc_num - 1 >= int(
                    tmp_str[2]) >= solved_solu.iter_env.qc_num and int(tmp_str[1]) < J_num_all:
                ls_ls.append(int(tmp_str[2]) - solved_solu.iter_env.qc_num)
            if tmp_str[0] == 'x':
                if m_ls.get(int(tmp_str[-1])) is None:
                    m_ls.setdefault(int(tmp_str[-1]), [int(tmp_str[1]), int(tmp_str[2]), -1])
                else:
                    tmp: list = m_ls.get(int(tmp_str[-1]))
                    if tmp.count(int(tmp_str[1])) is not 0:
                        tmp.insert(tmp.index(int(tmp_str[1])) + 1, int(tmp_str[2]))
                    elif tmp.count(int(tmp_str[2])) is not 0:
                        tmp.insert(tmp.index(int(tmp_str[2])), int(tmp_str[1]))
                    else:
                        tmp.extend([int(tmp_str[1]), int(tmp_str[2])])

    print(ls_ls)
    m_ls = sorted(m_ls.items(), key=lambda d: d[0], reverse=False)
    for m in m_ls:
        print(m)
    var_ls = [solved_solu.iter_env.qc_num + i * machine_num + (J_num_all + 2) * machine_num
              for i in range(0, (J_num_all + 2) * (J_num_all + 2))]
    var_ls.extend([solved_solu.iter_env.qc_num + 1 + i * machine_num + (J_num_all + 2) * machine_num
                   for i in range(0, (J_num_all + 2) * (J_num_all + 2))])
    var_ls.extend([solved_solu.iter_env.qc_num + 2 + i * machine_num + (J_num_all + 2) * machine_num
                   for i in range(0, (J_num_all + 2) * (J_num_all + 2))])
    var_ls.extend([solved_solu.iter_env.qc_num + 3 + i * machine_num + (J_num_all + 2) * machine_num
                   for i in range(0, (J_num_all + 2) * (J_num_all + 2))])
    print('Solution:', [MLP.getVars()[i].VarName for i in var_ls if
                        MLP.getVars()[i].x != 0])
    print('Makespan: ', MLP.getVars()[-2].x)
    print('Congestion time: ', MLP.getVars()[-1].x)
    return MLP


if __name__ == '__main__':
    pass

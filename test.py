# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 2:07 PM
# @Author  : JacQ
# @File    : test.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class Subproblem:
    def __init__(self, N, M) -> None:
        self.N, self.M = N, M
        self.m = gp.Model("subproblem")
        self.u = self.m.addVars(N + M, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')

    def add_constrs(self, A, c):
        self.m.addConstrs(
            (gp.quicksum(A[j, i] * self.u[j] for j in range(A.shape[0])) >= c[i]) for i in range(A.shape[1]))

    def set_objective(self, B, b, y):
        self.p = (b - np.dot(B, y)).reshape(self.N + self.M)
        self.m.setObjective(gp.quicksum(self.p[i] * self.u[i] for i in range(self.N + self.M)), sense=GRB.MINIMIZE)

    def solve(self, flag=0):
        self.m.Params.OutputFlag = flag
        self.m.Params.InfUnbdInfo = 1
        self.m.optimize()

    def get_status(self):
        if self.m.Status == GRB.Status.UNBOUNDED or self.m.Status == GRB.Status.INF_OR_UNBD:
            return np.array([x.getAttr('UnbdRay') for x in self.m.getVars()]), self.m.Status
        elif self.m.Status == GRB.Status.OPTIMAL:
            return self.get_solution(), self.m.Status
        else:
            return None

    def get_solution(self):
        return np.array([self.m.getVars()[i].x for i in range(self.M + self.N)])

    def get_objval(self):
        return self.m.ObjVal if self.m.Status == GRB.Status.OPTIMAL else -np.inf

    def write(self):
        self.m.write("sub_model.lp")


class Master:
    def __init__(self, N, M, d) -> None:
        self.m = gp.Model("Master")
    #
    # def build_model(self):
    #


    def add_cut1(self, B, b, u):
        self.p = np.dot(u.T, B)
        self.q = np.dot(u.T, b)
        self.m.addConstr(self.p[0] * self.y <= self.q[0])

    def add_cut2(self, B, b, u):
        self.p = np.dot(u.T, B)
        self.q = np.dot(u.T, b)
        self.m.addConstr(self.z <= self.q[0] - self.p[0] * self.y)

    def set_objective(self):
        self.m.setObjective(self.z + self.d * self.y, sense=GRB.MAXIMIZE)

    def solve(self, flag=0):
        self.m.Params.OutputFlag = flag
        self.m.optimize()

    def get_solution(self):
        return self.m.getVars()[0].x

    def get_objval(self):
        return self.m.ObjVal if self.m.Status == GRB.Status.OPTIMAL else np.inf

    def write(self):
        self.m.write("master_model.lp")
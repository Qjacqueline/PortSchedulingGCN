# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 9:48 PM
# @Author  : JacQ
# @File    : calculate_LB.py
from copy import deepcopy

from algorithm_factory.algo_utils.machine_cal_methods import cal_LB1, cal_LB2, cal_LB4
from algorithm_factory.algorithm_heuristic_rules import Least_Mission_Num_Choice
from data_process.input_process import read_input
from gurobi_solver import RelaxedCongestionPortModel, solve_model

if __name__ == '__main__':
    m_num_ls = [i for i in range(50)]  # , 500, 1000
    inst_type_ls = ['A2' for _ in range(len(m_num_ls))]
    lb1s, lb2s, lb3s, ms = [], [], [], []
    for i in range(len(m_num_ls)):
        solu = read_input('train', str(m_num_ls[i]), inst_type_ls[i], 100)
        # 计算下界
        _, lb_env, _ = Least_Mission_Num_Choice(deepcopy(solu.init_env))
        lb1 = cal_LB1(lb_env)
        lb2, r_lb2 = cal_LB2(lb_env)
        lb3 = cal_LB4(lb_env, r_lb2)
        solu.l2a_init()
        model = RelaxedCongestionPortModel(solu)
        model.construct()
        solve_model(MLP=model.MLP, inst_idx=inst_type_ls[i] + '_' + str(m_num_ls[i]), solved_env=solu, tag='_relax',
                    X_flag=False, Y_flag=False)
        lb1s.append(lb1)
        lb2s.append(lb2)
        lb3s.append(lb3)
        ms.append(model.MLP.getVars()[-2].x)

    for i in range(len(m_num_ls)):
        print("算例为\t" + inst_type_ls[i] + str(m_num_ls[i]) +
              "\t松弛解为\t" + str(ms[i]) + "\tlb1为\t" + str(lb1s[i]) +
              "\tlb2为\t" + str(lb2s[i]) + "\tlb3为\t" + str(lb3s[i]) + '\tmin\t' + str(max(lb1s[i], lb3s[i])))

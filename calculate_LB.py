# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 9:48 PM
# @Author  : JacQ
# @File    : calculate_LB.py
from copy import deepcopy

from algorithm_factory.algo_utils.machine_cal_methods import cal_LB1, cal_LB2, cal_LB4
from algorithm_factory.algorithm_heuristic_rules import Least_Mission_Num_Choice
from data_process.input_process import read_input

if __name__ == '__main__':
    m_num_ls = [10, 12, 27, 10, 15, 21, 10, 11, 21,
                10, 11, 10, 14, 15, 10, 11, 12,
                10, 17, 21, 10, 14, 18, 10, 16, 23]
    inst_type_ls = ['A2_t', 'A2_t', 'A2_t', 'B2_t', 'B2_t', 'B2_t', 'C2_t', 'C2_t', 'C2_t',
                    'D2_t', 'D2_t', 'E2_t', 'E2_t', 'E2_t', 'F2_t', 'F2_t', 'F2_t',
                    'G2_t', 'G2_t', 'G2_t', 'H2_t', 'H2_t', 'H2_t', 'Z2_t', 'Z2_t', 'Z2_t']
    for i in range(len(m_num_ls)):
        solu = read_input('train', str(m_num_ls[i]), inst_type_ls[i], m_num_ls[i])
        # 计算下界
        _, lb_env, _ = Least_Mission_Num_Choice(deepcopy(solu.init_env))
        lb1 = cal_LB1(lb_env)
        lb2, r_lb2 = cal_LB2(lb_env)
        lb3 = cal_LB4(lb_env, r_lb2)
        print("算例为\t" + inst_type_ls[i] + str(m_num_ls[i]) + "\tlb1为\t" + str(lb1) +
              "\tlb2为\t" + str(lb2) + "\tlb3为\t" + str(lb3) + '\tmin\t' + str(max(lb1, lb3)))

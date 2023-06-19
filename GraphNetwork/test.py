# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         test.py
# Description:
# Author:       Lv
# Date:         2023/6/16
# -------------------------------------------------------------------------------
import pandas as pd

# FEATURE_NUM = 38
#
# machine_txt = open('./gat/data/machine-1-1.txt')
# machine_feature_list = []
# for i in range(FEATURE_NUM):
#     machine_feature_list.append([])
# print(machine_feature_list.__len__())
# for i, line in enumerate(machine_txt):
#     # temp_line =
#     for j in range(FEATURE_NUM):
#         machine_feature_list[j].append((float))

# print(machine_feature_list)

machine_1_1_df = pd.read_csv('./gat/data/machine-1-1.csv')
machine_data = machine_1_1_df.values
print(machine_data)

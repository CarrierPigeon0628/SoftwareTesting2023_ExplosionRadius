import os

from TGCN import draw
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import time
import re

import brief
import gat
import gcn

# <editor-fold desc="Img & Data">
brief_image = Image.open('./brief/img/brief.PNG')
lstm_image = Image.open('./brief/img/LSTM.png')

gat_image = Image.open('./gat/img/GAT.png')  # GAT结构图
mtad_gat_image = Image.open('./gat/img/MTAD-GAT.png')  # MTAD-GAT结构图
regularization_image = Image.open('./gat/img/regularization.PNG')

gcn_image = Image.open('./gcn/img/GCN.png')  # GCN结构图

machine_1_1_df = pd.read_csv('./gat/data/machine-1-1.csv')
machine_1_1_label_df = pd.read_csv('./gat/data/machine-1-1-label.csv')
machine_data = machine_1_1_df.values  # machine-1-1的全部检测数据
inference_score_df = pd.read_csv('./gat/data/inference_score.csv')
Anomaly_df = pd.read_csv('./gat/data/Anomaly.csv')

# </editor-fold>

st.sidebar.title('爆炸半径建模')
option = st.sidebar.selectbox(
    'Select Function Items To Test',
    ["简介(Brief)", "MTAD-GAT", 'T-GCN+GSL', "总结(Summary)"])

st.title(option)  # 设置页面主标题

if option == "简介(Brief)":
    # <editor-fold desc="导航栏内容">
    st.sidebar.markdown(brief.background)
    st.sidebar.image(brief_image, use_column_width=True)
    # </editor-fold>
    # <editor-fold desc="主页内容">
    st.markdown(brief.background)
    st.image(brief_image, use_column_width=True)
    st.subheader('问题分析')
    st.markdown('- 需要考虑不同服务器之间、同一台服务器的不同KPI之间的关联关系，以更好的对故障产生的相关原因进行定位')
    st.markdown('- 需要考虑到图网络的动态变化问题，因为在实际运维环境中，很可能由于部分服务器产生故障而停止运行从而导致图的网络结构'
                '发生变化，因此在建模过程中需要考虑到图的动态建模')

    st.subheader('LSTM —— 长短期记忆递归神经网络')
    st.markdown('长短期记忆递归神经网络（Long Short-Term Memory Recurrent Neural Network，LSTM）是一种特殊类型的递归神经网络，它在处理序列数据时具有优秀的记忆能力。'
                '传统的递归神经网络在处理长序列时往往会面临梯度消失或梯度爆炸的问题，导致难以捕捉到序列中的长距离依赖关系。'
                'LSTM通过引入门控机制解决了这个问题，使得网络能够更好地处理长序列数据。')
    st.image(lstm_image, use_column_width=True)
    st.subheader('GCN —— 图卷积神经网络')
    st.markdown('图卷积神经网络（Graph Convolutional Neural Network，GCN）是一种用于处理图结构数据的神经网络模型。'
                '传统的神经网络主要用于处理规则的、结构固定的数据，而GCN则能够有效地处理非欧几里得结构的数据，如社交网络、推荐系统、分子化学等。')
    st.markdown('GCN的核心思想是将图的节点和边作为输入，通过学习节点之间的关系来进行节点的表示学习和特征提取。'
                '与传统的卷积神经网络不同，GCN利用邻居节点的信息来更新每个节点的特征表示。')
    st.image(gcn_image, use_column_width=True)
    st.subheader('GAT —— 图注意力网络')
    st.markdown('图注意力网络（Graph Attention Network，GAT）是一种基于注意力机制的图卷积神经网络（GCN）扩展模型。'
                '它在处理图结构数据时能够自适应地学习节点之间的关系权重，从而更好地捕捉节点之间的重要性和相关性。')
    st.markdown('与传统的GCN使用固定的权重矩阵进行邻居节点特征的聚合不同，GAT引入了注意力机制来计算每个节点与其邻居节点之间的重要性权重。'
                '这样，每个节点可以自适应地调整邻居节点对其特征表示的贡献程度。')
    st.image(gat_image, use_column_width=True)
    # </editor-fold>
elif option == "MTAD-GAT":
    # <editor-fold desc="导航栏内容">
    st.sidebar.markdown(gat.abstract)
    st.sidebar.image(gat_image, use_column_width=True)
    option2 = st.sidebar.selectbox(
        'Select Different Items For Details',
        ['原理(Principle)', '性能测试(Performance Test)', '后续工作(Future Work)']
    )
    # </editor-fold>
    # <editor-fold desc="主页内容">
    st.header(option2)  # 设置页面副标题
    if option2 == '原理(Principle)':
        # <editor-fold desc="Principle部分主体内容">
        st.image(mtad_gat_image, use_column_width=True)
        st.markdown(gat.mtad_gat_abstract)
        st.markdown('- 使用基于自监督(self-supervised)的多变量时间序列异常检测框架，在训练过程中捕获了不同时间序列之间的关系。')
        st.markdown('- 利用两个并行图注意(GAT)网络动态学习不同时间序列和时间戳之间的关系。在没有任何先验知识的情况下成功地捕获了不同时间序列之间的相关性。')
        st.markdown('- 通过引入联合优化目标，将基于预测模型(forecasting-based)和基于重建模型(reconstruction-based)的优点结合起来。')
        st.subheader('Procedure')
        st.markdown('1. 输入')
        st.markdown(r'$x\in R^{n\times k}$，其中$n$代表时间序列的长度，$k$表示特征的数量')
        st.line_chart(machine_1_1_df['Feature1'])
        st.line_chart(machine_1_1_df['Feature8'])
        st.line_chart(machine_1_1_df['Feature14'])
        st.markdown('2. 推理')
        st.markdown('- 数据正则化')
        st.image(regularization_image, use_column_width=True)
        st.markdown('- 数据清洗')
        st.markdown(r'对每个独立的时间序列使用单变量异常检测算法SR（Spectral Residual），来检测异常时间戳。'
                    r'之后，设置阈值为3来生成异常检测结果。并将这些检测到的异常时间戳用该时间戳周围的正常值替换。')
        st.markdown('3. 输出')
        st.markdown(r'$y\in R^n \quad y_i\in\{0,1\}$，每个y值对应当前时间戳是否发生故障')
        st.line_chart(inference_score_df)
        st.bar_chart(Anomaly_df)
        # </editor-fold>
        if st.button("Check ᕙ(• ॒ ູ•)ᕘ"):
            print(len(machine_1_1_label_df['Label']))
            print(len(Anomaly_df['Anomaly']))
            match_num = 100
            for i in range(min(len(machine_1_1_label_df['Label']), len(Anomaly_df['Anomaly']))):
                if machine_1_1_label_df['Label'][i] == Anomaly_df['Anomaly'][i]:
                    match_num += 1
            st.markdown('$precision$:{}'.format(match_num / len(machine_1_1_label_df['Label'])))
            # print(comparision)
    elif option2 == '性能测试(Performance Test)':
        uploaded_file = st.file_uploader("", type="tfrecords")
        # if uploaded_file is not None:
        #     chart_data = pd.read_csv(uploaded_file)
        # if st.checkbox('Show test samples'):
        #     st.write(chart_data)
        if st.button("Diagnose ᕙ(• ॒ ູ•)ᕘ"):
            print('hello world')
    elif option2 == '后续工作(Future Work)':
        st.sidebar.image(gat_image, use_column_width=True)
        st.markdown('### 1. 对预测网络进行调整，使用其他例如LSTM、Transformer等模型进行预测(Forcasting)')
        st.markdown('### 2. 优化阈值的选择方法，使用遗传算法、蚁群算法等智能群体算法进行估计')
    # </editor-fold>
elif option == "T-GCN+GSL":
    # <editor-fold desc="导航栏内容">
    st.sidebar.markdown(gcn.gcn_abstract)
    st.sidebar.image(gcn_image, use_column_width=True)
    option2 = st.sidebar.selectbox(
        'Select Different Items For Details',
        ['原理(Principle)', '性能测试(Performance Test)', '后续工作(Future Work)']
    )
    # </editor-fold>
    # <editor-fold desc="主页内容">
    st.header(option2)  # 设置页面副标题
    if option2 == '原理(Principle)':
        pass
    elif option2 == '性能测试(Performance Test)':
        uploaded_file = st.file_uploader("", type="csv")
        if st.button("Diagnose ᕙ(• ॒ ູ•)ᕘ"):
            fig, normal_label, normal_output, abnormal_label, abnormal_output = draw.test()
            st.pyplot(fig)
            normal = np.vstack((normal_label, normal_output)).T
            abnormal = np.vstack((abnormal_label, abnormal_output)).T
            normal = pd.DataFrame(normal)
            abnormal = pd.DataFrame(abnormal)
            st.line_chart(normal)
            st.line_chart(abnormal)
    elif option2 == 'Future Work':
        pass
    # </editor-fold>
elif option == "总结(Summary)":
    st.markdown('## 我们的工作')
    st.markdown('### 1. 使用LSTM、GCN、GAT模型对爆炸半径求解问题建模，并建立了问题求解的全流程')
    st.markdown('### 2. 在SMD公开数据集上进行性能测试，在GCN和GAT模型的测试中普遍达到90%的故障检测准确率')
    st.markdown('### 3. 提出了将服务器各KPI之间的影响关系作为先验知识初始化模型参数的可行方法')
    st.markdown('## 不足之处')
    st.markdown('### 1. 目前仅在SMD数据集上进行了测试，后续应当在更多的公开数据集上测试模型性能')
    st.markdown('### 2. 目前模型仅针对单一机器进行建模，后续需要拓展到机群当中')

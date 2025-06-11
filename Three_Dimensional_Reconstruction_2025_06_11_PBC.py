# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:15:54 2023
@author: Dongyuan Ge
"""
import numpy as np
import math
import random
import string
import matplotlib as mpl
import matplotlib.pyplot as plt



# 设置随机数种子（可选，但有助于结果的可重复性）
np.random.seed(0)  # 设置种子为0，以便每次运行代码时都能得到相同的结果

# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a


# 生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)
    return a


# 函数sigmoid(),这里采用tanh，因为看起来要比标准的sigmoid函数好看
def sigmoid(x):
    return 1*(math.tanh(x))


# 函数sigmoid的派生函数
def derived_sigmoid(x):
    return 1*(1.0 - x ** 2)


# 构造三层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层，输出层的节点数
        self.num_in = num_in   # 增加一个偏置结点
        self.num_hidden = num_hidden   # 增加一个偏置结点
        self.num_out = num_out

        # 激活神经网络的所有节点（向量）
        self.active_in = [1.0] * self.num_in
        self.active_hidden = [1.0] * self.num_hidden
        self.active_out = [1.0] * self.num_out   #可以用2

        # 创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden)
        self.wight_out = makematrix(self.num_hidden, self.num_out)

        #对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.wight_in[i][j] = random_number(-0.9, 0.9)   #原来的是random_number(-0.2, 0.2)
        for j in range(self.num_hidden):
            for k in range(self.num_out):
                self.wight_out[j][k] = random_number(-0.9, 0.9)

        self.wight_in=[

           #
            [5.437694944452474, -7.806223283283583, 3.4010231484516193, 0.034119074785716776, 1.2228116168128274,  -0.30933986987046547,  4.904352121016822, -7.414289798059994, 3.1015855611738656, -0.04710478970670941, 0.5735930261628404, 0.3596677834410535],
            [0.12740669054821877, -5.014850113538859, -0.8477613831748844, 1.0170493195084434, -3.9092428052281813, -0.8337627663998691 ,  0.11600005201293703, -4.383986083039213, -0.878548675564849, 0.5426955763115496, -3.0077358823464193, -1.008895067560161],
            [- 0.0911658657340839, 4.287049084075241, -26.025559495688388, 15.61574137245841, 9.910987445701943,    6.332991843877077,    -0.350967249617043, 4.693163452197321, -25.57072835210723, 15.64010607988969, 9.09444403376687, 6.504187149219303],
            [-0.3329881770198686, 0.5570343535578456, -1.891982037571118, -0.4828169874986403, -0.9438029484839108,   -0.26154659350860215,  -0.3312100312908608, 0.5586069476235483, -1.8900185930455378, -0.4787734956038044, -0.9491890356647626, -0.2583882796923472]
        ]
        self.wight_out = [
            # 以下为左摄像机的
            [1.4796827479033727, 1.7741334632632455,   0,  0],
            [0.5854511070940049, 0.6058183553069706,   0, 0],
            [2.289149627931907, -0.0079517354652275,   0, 0],
            [3.9661153628492634, 0.5987316896140886,   0, 0],
            [0.35411880211121516, -4.04911374454641,   0, 0],
            [- 3.531453462241525, 4.493704320375342,  0, 0],
#以下为右摄像机的
            [0,  0,   1.526365001481235, 0.1720004723222518],
            [0,  0,   0.31461897044945647, 0.07770851147168532],
            [0,  0,   1.56842736593774, 0.14304530252314748],
            [0,  0,   2.784647056772489, 0.7507135312560574],
            [0,  0,   0.8140684942298372, -3.645050120033926],
            [0,  0,  - 3.30320976779897, 3.7892961857912]
            # 2024.10.05      前两次运行得到的权值的组合
        ]
        # 最后建立动量因子（矩阵）
        self.ci = makematrix(self.num_in, self.num_hidden)
        self.co = makematrix(self.num_hidden, self.num_out)

        # 信号正向传播

    def update(self, inputs):
        if len(inputs) != self.num_in:
            raise ValueError('与输入层节点数不符')

        # 数据输入输入层
        for i in range(self.num_in ):
                 self.active_in[i] = inputs[i]  # active_in[]是输入数据的矩阵

        # 数据在隐藏层的处理
        for j in range(self.num_hidden ):
            sum = 0.0
            for i in range(self.num_in):
                sum = sum + self.active_in[i] * self.wight_in[i][j]
            self.active_hidden[j] = sigmoid(sum)  # active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据

        # 数据在输出层的处理
        for k in range(self.num_out):
            sum = 0.0
            for j in range(self.num_hidden):
                sum = sum + self.active_hidden[j] * self.wight_out[j][k]
            self.active_out[k] = sigmoid(sum)  # 与上同理

        return self.active_out[:]

    # 误差反向传播
    def errorbackpropagate(self, targets, lr, m):  # lr是学习率， m是动量因子
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')

        # 首先计算输出层的误差
        out_deltas = [0.0] * self.num_out
        for k in range(self.num_out):
            error = targets[k] - self.active_out[k]
            out_deltas[k] = derived_sigmoid(self.active_out[k]) * error

        # 然后计算隐藏层误差
        hidden_deltas = [0.0] * self.num_hidden
        for j in range(self.num_hidden):
            error = 0.0
            for k in range(self.num_out):
                error = error + out_deltas[k] * self.wight_out[j][k]
            hidden_deltas[j] = derived_sigmoid(self.active_hidden[j]) * error

        # 然后更新输入的信息
        for i in range(self.num_in-1):
            change=[0, 0, 0]
            for j in range(self.num_hidden):
                change[i] =change[i]+ hidden_deltas[j] * self.wight_in[i][j]

            self.active_in[i] = self.active_in[i] + lr * change[i]
                #self.ci[i][j] = change

        # 计算总误差
        error = 0.0
        for i in range(len(targets)):
            error = error + 0.5 * (targets[i] - self.active_out[i]) ** 2
        return error

    # 测试
    def test(self, patterns):
        for i in patterns:
            print(i[0], '->', self.update(i[0]))

    # 权重
    def weights(self):
        print("输入层权重")
        for i in range(self.num_in):
            print(self.wight_in[i])
        print("输出层权重")
        for i in range(self.num_hidden):
            print(self.wight_out[i])


    def train(self, pattern, itera=300001, lr=0.00200, m=0.00):

        #for j in pattern:
        for k, j in enumerate( pattern):
            for i in range(itera):  # 9500000
                error = 0.0
                if i == 0:
                   inputs = j[0]
                #   print('initial1=',inputs)
                   # inputs=[0.9698416053274535, 0.9492286353372284, 0.7490594680050898, 0.0001]
                   # print('initial2=', inputs)
                   targets = j[1]
                   self.update(inputs)
                   error = error + self.errorbackpropagate(targets, lr, m)
                else:
            # for j in pattern:
                   inputs =[self.active_in[0],self.active_in[1],self.active_in[2],0.0001]
                   # print('gdy_1030=', inputs)
                   targets = j[1]
                   self.update(inputs)
                   error = error + self.errorbackpropagate(targets, lr, m)
                if i % 300000 == 0:
                   # print('误差gdy %-.25f'  error)
                   # print('gdy_inputs=', inputs)
                   # print(  error)
                   # print( inputs)
                   print(k+1,error, inputs)
                 #  print(inputs)

# 实例
def demo():
    gdy = np.random.uniform(0, 1, 3);
    gdy3=np.concatenate((np.random.uniform(0, 1, 3),[0.0001]),0)
    patt  =  [
         #np.concatenate(
         #gdy3=np.concatenate((np.random.uniform(0, 1, 3),[0.0001]),0)
        #  [gdy3,
        #  [0.172735170, 0.341553079, 0.109960373, 0.269760623]],

        #以下为PBC的MCS51芯片选定引脚的中心点的三维重建    2025_06_10     00:27
        [gdy3, [0.023459104, 0.181094805, 0.001177879, 0.143681527]],
        [gdy3, [0.025325625, 0.181087434, 0.002669862, 0.143765315]],
        [gdy3, [0.027192651, 0.181079856, 0.004161965, 0.143849137]],
        [gdy3, [0.029060168, 0.181072002, 0.005654178, 0.143932936]],
        [gdy3, [0.030926177, 0.181064018, 0.007146498, 0.144016828]],
        [gdy3, [0.032792738, 0.181055825, 0.008638977, 0.144100753]],
        [gdy3, [0.034659687, 0.181047368, 0.010131485, 0.144184658]],
        [gdy3, [0.036527099, 0.181038786, 0.011624081, 0.144268657]],
        [gdy3, [0.045696254, 0.205405337, 0.017916202, 0.164035623]],
        [gdy3, [0.047571516, 0.205393870, 0.019409069, 0.164118294]],
        [gdy3, [0.049447162, 0.205382295, 0.020901949, 0.164201014]],
        [gdy3, [0.051322848, 0.205370535, 0.022394892, 0.164283723]],
        [gdy3, [0.053197777, 0.205358736, 0.023887894, 0.164366538]],
        [gdy3, [0.055072727, 0.205346754, 0.025380945, 0.164449344]],
        [gdy3, [0.056948173, 0.205334729, 0.026874100, 0.164532255]],
        [gdy3, [0.058825950, 0.205322604, 0.028367232, 0.164615217]],
        # 以下为PBC的MCS51芯片选定引脚的中心点的三维重建    2025_06_10     00:28
    ]

    # 创建神经网络，4个输入节点，12个隐藏层节点，4个输出层节点
    n = BPNN(4, 12, 4)
    # 训练神经网络
    n.train(patt)
    # 测试神经网络
    # n.test(patt)
    # 查阅权重值
    n.weights()


if __name__ == '__main__':
    demo()
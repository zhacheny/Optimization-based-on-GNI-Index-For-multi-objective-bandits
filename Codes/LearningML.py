#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from Arm import arm
import math
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import os
from cvxopt import *

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#Calculating the optimal solution
def change_X_opt_1(K,Arms,D,W_pre):
    W =[0.0]*len(W_pre);
    for i in range(len(W_pre)-1):
        W[i] = W_pre[i]-W_pre[i+1]
    W[D-1]= W_pre[D-1]
    row = (D+D*D+K)
    c = []
    for i in range(0,row):
        if(i >=0 and i <D):
            c.append(W[i]*(i+1))
        elif(i>=D and i<row-K):
            c.append(W[(i-D)//D])
        else:
            c.append(0)
    C = matrix([c[i] for i in range(0,len(c))],(row,1))
    g = []
    LIST = -1
    for i in range(0,row*(D*D+K+D*D)):
        Row = i // row
        Col = i % row

        if(Row>=0 and Row<D*D):
            if(Row%D==Col):
                g.append(1.0)
                if(Row%D ==0):
                    LIST +=1
            elif( Col == D + Row%D*D+LIST):
                g.append(1.0)
            elif(Col>=D+D*D):
                g.append(-1*Arms[(Col-D+D*D)%K].P[LIST])
            else:
                g.append(0.0)
        elif(Row>=D*D):
            if(Row-D*D == Col-D and Col>=D ):
                g.append(1.0)
            else:
                g.append(0.0)

    G = matrix([g[i] for i in range(0,len(g))],(row,(D*D+K+D*D)))
    G = -1.0*(G.trans())
    h = []
    z=-1
    for i in range(0,(D*D*2+K)):
        h.append(0.0)
    H = matrix([h[i] for i in range(0,len(h))],(D*D*2+K,1))
    H = -1*H
    ones = []
    for i in range(0, K):
        ones.append(1.0)
    X = []
    for i in range(0,row):
        if (i>=D+D*D and i < row):
            X.append(1.0)
        else:
            X.append(0.0)
    A = matrix([X[i] for i in range(0,len(X))], (1,row))
    b = matrix(1.0)
    sol = solvers.lp(C, G, H, A, b)
    qp = np.array(sol['x']).flatten()
    return qp[-K:]


#Updating function of Baseline
def change_X_1(a,K,t,Arms,D,W_pre):
#Calculate eta
    W =[0.0]*len(W_pre);
    for i in range(len(W_pre)-1):
        W[i] = W_pre[i]-W_pre[i+1]
    W[D-1]= W_pre[D-1]
    row = (D+D*D+K)
    beta = (math.sqrt(2.0)*math.sqrt(math.log(2.0/0.95)/(t+K)))/(1.0-1.0/math.sqrt(K))
    beta /= K
#Create C
    c = []
    for i in range(0,row):
        if(i >=0 and i <D):
            c.append(W[i]*(i+1))
        elif(i>=D and i<row-K):
            c.append(W[(i-D)//D])
        else:
            c.append(0)
    C = matrix([c[i] for i in range(0,len(c))],(row,1))
#Create G
    g = []
    LIST = -1
    for i in range(0,row*(D*D+K+D*D)):
        Row = i // row
        Col = i % row

        if(Row>=0 and Row<D*D):
            if(Row%D==Col):
                g.append(1.0)
                if(Row%D ==0):
                    LIST +=1
            elif( Col == D + Row%D*D+LIST):
                g.append(1.0)
            elif(Col>=D+D*D):
                g.append(-1*Arms[(Col-D+D*D)%K].E[LIST])
            else:
                g.append(0.0)
        elif(Row>=D*D):
            if(Row-D*D == Col-D and Col>=D ):
                g.append(1.0)
            else:
                g.append(0.0)
    G = matrix([float(g[i]) for i in range(0,len(g))],(row,(D*D+K+D*D)))
    G = -1.0*(G.trans())
#Create h
    h = []
    z=-1
    for i in range(0,(D*D*2+K)):
         if(i >=0 and i <D*D):
             h.append(0.0)
         elif(i<2*D*D+K and i >= 2*D*D):
             h.append(beta)
         else:
             h.append(0.0)
    H = matrix([h[i] for i in range(0,len(h))],(D*D*2+K,1))
    H = -1*H
#Create A,b
	ones = []
    for i in range(0, K):
        ones.append(1.0)
    X = []
    for i in range(0,row):
        if (i>=D+D*D and i < row):
            X.append(1.0)
        else:
            X.append(0.0)
    A = matrix([X[i] for i in range(0,len(X))], (1,row))
    b = matrix(1.0)
#linear optimal
    sol = solvers.lp(C, G, H, A, b)
    qp = np.array(sol['x']).flatten()
    return qp[-K:]


#Get arms 
def init(K,D):
    a = []
    Arms = []
    Arms1 = []
    W = []
    for i in range(0,K):
        a.append(1/K)
        Arm = arm(D)
        Arm1 = arm(D)
        Arm1.P = deepcopy(Arm.P);
        Arm1.ini_exceptb()
        Arms.append(Arm)
        Arms1.append(Arm1)
    for i in range(0, D):
        W.append(1.0/(2**(i)))

    return a,Arms,Arms1,W

#Select arms according to strategy a
def get_select(a):
    x = random.randint(0,359)
    low =0
    up=0
    for i in range(0,len(a)):
        up = low + 360.0*a[i]
        if(x<up and x >=low):
            return i
        low = up

#Update a via online gradient descent 
def update_a(a,Arms,W,learn_rate,D):
    for i in range(0,len(a)):
        gradint = 0
        for j in range(0,D):
            gradint = gradint+W[j]*Arms[i].sortE[j]
        a[i] = a[i]-learn_rate*gradint
    return a

#Project updated a into the correct one
def change_A(a,K,t):
    beta = (math.sqrt(2.0)*math.sqrt(math.log(2/0.95)/(t+K)))/(1-1/math.sqrt(K))
    beta /= K
    if(beta >1/K):
        beta = 1/K
    M_A = matrix([a[i] for i in range(0,K)],(K,1))
    # create P
    p=[]
    for i in range(0,K):
        p.append(1.0)
    ones = matrix([p[i] for i in range(0,K)],(K,1))
    P =(2.0)*spdiag(p)

    #create q
    q = -2.0*M_A
    #create G
    G = -1.0*((1.0)/(2.0))*P
    #create h
    h=-1*beta*ones
    # create A
    A =  matrix([p[i] for i in range(0,K)],(1,K))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)
    return np.array(sol['x']).flatten()

#Figure GGI value of the mixed strategy a 
def expectP(a, Arms, K, W):
    sumvector = [0.0] * len(Arms[0].P);

    for j in range(len(Arms[0].P)):
        for i in range(len(a)):
            sumvector[j] += a[i] * Arms[i].P[j];
    sumvector = sorted(sumvector, reverse=True)
    ginivalue = 0.0;
    for i in range(len(W)):
        ginivalue += W[i] * sumvector[i];
    return ginivalue;

#Rank the element in a and return the ranked index
def figureRank(Arms,a,W):
    sumvector=[0.0]*len(Arms[0].E);
    
    for j in range(len(Arms[0].E)):
        for i in range(len(a)):
            sumvector[j]+=a[i]*Arms[i].E[j];
            
    return [i[0] for i in sorted(enumerate(sumvector), key=lambda x:x[1],reverse=True)]

Allregret=[]
Allregretorg = []
regretMax = []
regretMin = []
regretorgMax = []
regretorgMin = []
regretorgMiu = []
regretMiu = []
K = 5
D = 5
#100 repetitions
for m in range(0,100):
    #Get new Arms 
    #In order to avoid potential negtive effect of shallow copy, use deepcopy
    a,Arms,orgArms,W = init(K,D)
    orgA = deepcopy(a)
    opta = change_X_opt_1(K,Arms,D,W);
    #Learning rate
    learn_Rate =0.01
    listregret = []
    listregretorg = []
    
    cost_Con = expectP(opta, orgArms, K, W)
    avg_a=[0.0]*K;
    avg_orga=[0.0]*K;
    #The number of iterations of online gradient descent 
    Iteration_num = 100
    for i in range(1, Iteration_num):
        #Select arm
        j2 = get_select(a)
        j1 = get_select(orgA)
        #update the reward of selected arm
        orgArms[j1].update(figureRank(orgArms,orgA,W))
        orgA = change_X_1(orgA,K,i,orgArms,D,W);
        #Baseline (linear programming)
        Arms[j2].update(figureRank(Arms,a,W))
        a = update_a(a, Arms, W, learn_Rate, D)
        #online Gradient Descent 
        a = change_A(a, K, i).tolist()
        #Caculate the average strategy
        avg_a = [float(avg_a[j]*(i-1)+a[j])/i for j in range(K)];
        avg_orga = [float(avg_orga[j]*(i-1)+orgA[j])/(i) for j in range(K)];
        #caculate the GGI of average strategy 
        listregret.append(expectP(avg_a, Arms, K, W))
        listregretorg.append(expectP(avg_orga, orgArms, K, W))
    #caculate regret
    listregret = [(listregret[i]-cost_Con) for i in range(len(listregret))]
    listregretorg = [(listregretorg[i]-cost_Con) for i in range(len(listregretorg))]
    #Save the result
    Allregret.append(listregret)
    Allregretorg.append(listregretorg)

#Calculate the upper bound, lower bound and average regret in 100 repetive experiments
for j in range(len(Allregret[0])):
    d = [Allregret[i][j] for i in range(len(Allregret))]
    d.sort()
    regretMin.append(d[0])
    regretMiu.append(sum(d) / len(d))
    regretMax.append(d[len(Allregretorg)-1])
for j in range(len(Allregretorg[0])):
    d = [Allregretorg[i][j] for i in range(len(Allregretorg))]
    d.sort()
    regretorgMin.append(d[0])
    regretorgMiu.append(sum(d) / len(d))
    regretorgMax.append(d[len(Allregretorg)-1])

#Draw figure
x=list(range(0,Iteration_num-1))
plt.plot(regretMiu, linewidth=5,c="r")
plt.plot(regretMin ,c='r')
plt.plot(regretMax, c='r')
plt.xlabel("Rounds")
plt.ylabel("Regret")
plt.title("K = 10,D = 10")
plt.fill_between(x,regretMin,regretMax, color='red', alpha='0.5')

x1=list(range(0,Iteration_num-1))
plt.plot(regretorgMiu, linewidth=5,c="b")
plt.plot(regretorgMin ,c='b')
plt.plot(regretorgMax, c='b')
plt.xlabel("Rounds")
plt.ylabel("Regret")
plt.title("K = 10,D = 10")
plt.fill_between(x1,regretorgMin,regretorgMax, color='blue', alpha='0.5')
plt.figure()

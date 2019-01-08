import random
import numpy as np
from scipy import stats
import os
from copy import deepcopy
'''
Class arm
D is the demension of vector
E is reward of this arm which is a vector
P is the mean of reward
sortE is sorted E
sortP is sorted P
T is the time we choose this arm.
init() is to init the arm
update() is used when we use to choose this arm.
'''
class arm:
    D =0.0
    E= []
    P=[]
    T = 0
    sortE = []
    sortP=[]

    def __init__(self,d):
        self.D = d
        self.E=[]
        self.P=[]
        for i in range(0,d):
            p = round(random.random(),2)
            self.P .append(p)
            e = stats.binom.rvs(1,p,size=1)
            self.E.append(float(e[0]))
        self.T =0
        self.sortE = sorted(self.E,reverse=True)
        self.sortP = sorted(self.P,reverse=True)

    def ini_exceptb(self):
        self.E=[]
        for i in range(0,self.D):
            e = stats.binom.rvs(1,self.P[i],size=1)
            self.E.append(e[0])
        self.T =0
        self.sortE = sorted(self.E,reverse=True)

    def update(self,rank):
        self.T = self.T + 1
        old_E = self.E
        self.E = []
        for i in range(0, len(old_E)):
            new_e = float(stats.binom.rvs(1, self.P[i], size=1)[0])
            summm = (old_E[i]*self.T + new_e)/(self.T+1)
            self.E.append(summm)
        for i in range(len(self.E)):
            self.sortE[i]=self.E[rank[i]];
        #self.sortE = sorted(self.E, reverse=True)
        #self.sortP = sorted(self.P, reverse=True)




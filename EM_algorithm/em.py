import numpy as np
from random import uniform 
from scipy.optimize import LinearConstraint, minimize

MAX_ITER = 1000

class DataGenerator:
    '''
    - probs: a list of probs in order of [p(aa), p(a), p(b), p(stop)]
    '''

    def __init__(self, probs = [0.5, 0.125, 0.25, 0.125]):
        if sum(probs) != 1:
            assert(False)
        self.data_ = [] 
        names = ["aa", "a", "b", "stop"]
        self.probs_ = {names[i]:probs[i] for i in range(len(probs))}

    def getData(self):
        return self.data_

    def getProbs(self):
        return self.probs_

    def generate(self, num):
        self.data_ = [] # reset data if generated before 
        for i in range(num):
            s = ""
            while True:
                p = uniform(0,1)
                if p < self.probs_["aa"]:
                    s += "aa"
                elif self.probs_["aa"] <= p < self.probs_["aa"] + self.probs_["a"]:
                    s += "a"
                elif self.probs_["aa"] + self.probs_["a"] <= p < 1 - self.probs_["stop"]:
                    s += "b"
                else:
                    self.data_.append(s)
                    break 
        return self.data_

def Estep_helper(data):
    # - data is a list of strings generated from DataGenerator
    data = sorted(data, key = len)
    mem = dict()
    # a dict of string:list[(#aa,#a,#b,prob)] where the list contains all permutations

    def findAllPerm(s):
        # returns a list of tuple (# of "aa", # of "a", # of "b")
        if s == "":
            return [(0, 0, 0)] # (#"aa", #"a", #"b")
        if s in mem:
            return mem[s]
        elif len(s) > 1 and s[:2] == "aa":
            first = [(i[0], i[1]+1, i[2]) for i in findAllPerm(s[1:])]
            second = [(i[0]+1, i[1], i[2]) for i in findAllPerm(s[2:])]
            result = first + second
            mem[s] = result
            return result 
        else: 
            if s[0] == "a":
                result = [(i[0], i[1]+1, i[2]) for i in findAllPerm(s[1:])]
            else:
                result = [(i[0], i[1], i[2]+1) for i in findAllPerm(s[1:])]
            mem[s] = result
            return result 

    result = []
    for d in data:
        perm = findAllPerm(d)
        result.append(perm)
    return result 

def Estep(perms, theta):
    # - perms is a list of permutation returned by Estep_helper
    # - theta is a list of probs in order [p(aa), p(a), p(b), p(stop)]

    result = [0, 0, 0]
    for perm in perms:
        constant = 0
        Eaa = 0 
        Ea = 0 
        Eb = 0
        for p in perm:
            prob = (theta[0]**p[0])*(theta[1]**p[1])*(theta[2]**p[2])*theta[3]
            constant += prob
            Eaa += p[0] * prob
            Ea += p[1] * prob
            Eb += p[2] * prob
        result[0] += Eaa/constant
        result[1] += Ea/constant
        result[2] += Eb/constant

    return result + [len(perms)]

def M_create(expectation):
    def func(x):
        return -sum(np.multiply(np.log(x), np.array(expectation)))
    return func 
     
def M_step(permutations, max_iter, theta0):
    theta = theta0
    lc = LinearConstraint([[1,1,1,1],
                           [1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1]], 
                           [1,0,0,0,0], [1,1,1,1,1])
    for it in range(max_iter):
        res = minimize(M_create(Estep(permutations, theta)), np.array(theta0), 
                       method = 'trust-constr', 
                       constraints = [lc])
        if sum((np.array(theta) - res.x)**2) < 0.0001:
            return res.x.tolist()
        theta = res.x.tolist()
    print("Not converging\n")
    return res.x.tolist()

def EMalgorithm(numofdata, probs = [0.5, 0.125, 0.25, 0.125]):
    d = DataGenerator(probs)
    data = d.generate(numofdata) 
    permutations = Estep_helper(data)
    result = M_step(permutations, MAX_ITER, [0.25,0.25,0.25,0.25])
    return result 



probs = [0.2, 0.1, 0.4, 0.3]
r = EMalgorithm(10000, probs)
print(r)



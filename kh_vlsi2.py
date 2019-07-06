import numpy as np
import random
import math
import time

def func1(ind):
    with open('tfa_leak_25,0.8.sp','r') as file:
        data=file.readlines()
        #ind=str(ind)
        #print(len(data))
        #print(data[39])
        #print("lol",len(data))
        #data[27]= '.param wp='+str(ind[0])+'n'+'\n'
        #data[28]= '.param wn='+str(ind[1])+'n'+'\n'
    #data[29]= '.param lp='+str(ind[2])+'n'+'\n'
    #data[30]= '.param ln='+str(ind[3])+'n'+'\n'

        data[34]='Mp1    1   nodea   vdd   vdd   pmos  l=' + str(ind[1]) + 'n ' + 'w=' + str(ind[0]) + 'n\n'
        data[35]='Mp2    1   nodeb   vdd   vdd   pmos  l=' + str(ind[3]) + 'n ' + 'w=' + str(ind[2]) + 'n\n'
        data[36]='Mp3    nodecon nodec   1   vdd   pmos l=' + str(ind[5]) + 'n ' + 'w=' + str(ind[4]) + 'n\n'
        data[37]='Mn1    5   nodea   gnd   gnd   nmos  l=' + str(ind[7]) + 'n ' + 'w=' + str(ind[6]) + 'n\n'
        data[38]='Mn2    5   nodeb   gnd   gnd   nmos  l=' + str(ind[9]) + 'n ' + 'w=' + str(ind[8]) + 'n\n'
        data[39]='Mn3    nodecon nodec   5   gnd   nmos l=' + str(ind[11]) + 'n ' + 'w=' + str(ind[10]) + 'n\n'
        data[40]='Mp4    4   nodea   vdd   vdd   pmos  l=' + str(ind[13]) + 'n ' + 'w=' + str(ind[12]) + 'n\n'
        data[41]='Mp5    nodecon nodeb   4   vdd   pmos  l=' + str(ind[15]) + 'n ' + 'w=' + str(ind[14]) + 'n\n'
        data[42]='Mn4    nodecon nodeb   node4   gnd   nmos l=' + str(ind[17]) + 'n ' + 'w=' + str(ind[16]) + 'n\n'
        data[43]='Mn5    node4   nodea   gnd   gnd   nmos l=' + str(ind[19]) + 'n ' + 'w=' + str(ind[18]) + 'n\n'
        data[44]='Mp6    2   nodea   vdd   vdd   pmos  l=' + str(ind[21]) + 'n ' + 'w=' + str(ind[20]) + 'n\n'
        data[45]='Mp7    2   nodeb   vdd   vdd   pmos l=' + str(ind[23]) + 'n ' + 'w=' + str(ind[22]) + 'n\n'
        data[46]='Mp8    2   nodec   vdd   vdd   pmos l=' + str(ind[25]) + 'n ' + 'w=' + str(ind[24]) + 'n\n'
        data[47]='Mp9    nodes0n nodecon 2   vdd   pmos l=' + str(ind[27]) + 'n ' + 'w=' + str(ind[26]) + 'n\n'
        data[48]='Mn6    3   nodea   gnd   gnd   nmos l=' + str(ind[29]) + 'n ' + 'w=' + str(ind[28]) + 'n\n'
        data[49]='Mn7    3   nodeb   gnd   gnd   nmos l=' + str(ind[31]) + 'n ' + 'w=' + str(ind[30]) + 'n\n'
        data[50]='Mn8    3   nodec   gnd   gnd   nmos l=' + str(ind[33]) + 'n ' + 'w=' + str(ind[32]) + 'n\n'
        data[51]='Mn9    nodes0n nodecon 3   gnd   nmos l=' + str(ind[35]) + 'n ' + 'w=' + str(ind[34]) + 'n\n'
        data[52]='Mp10   9   nodea   vdd   vdd   pmos  l=' + str(ind[37]) + 'n ' + 'w=' + str(ind[36]) + 'n\n'
        data[53]='Mp11   8   nodeb   9   vdd   pmos    l=' + str(ind[39]) + 'n ' + 'w=' + str(ind[38]) + 'n\n'
        data[54]='Mp12   nodes0n nodec   8   vdd   pmos l=' + str(ind[41]) + 'n ' + 'w=' + str(ind[40]) + 'n\n'
        data[55]='Mn10   7   nodea   gnd   gnd   nmos  l=' + str(ind[43]) + 'n ' + 'w=' + str(ind[42]) + 'n\n'
        data[56]='Mn11   6   nodeb   7   gnd   nmos    l=' + str(ind[45]) + 'n ' + 'w=' + str(ind[44]) + 'n\n'
        data[57]='Mn12   nodes0n nodec   6   gnd   nmos  l=' + str(ind[47]) + 'n ' + 'w=' + str(ind[46]) + 'n\n'
        data[58]='Mp13   nodeco  nodecon vdd   vdd   pmos l=' + str(ind[49]) + 'n ' + 'w=' + str(ind[48]) + 'n\n'
        data[59]='Mn13   nodeco  nodecon gnd   gnd   nmos l=' + str(ind[51]) + 'n ' + 'w=' + str(ind[50]) + 'n\n'
        data[60]='Mp14   nodes0  nodes0n vdd   vdd   pmos l=' + str(ind[53]) + 'n ' + 'w=' + str(ind[52]) + 'n\n'
        data[61]='Mn14   nodes0  nodes0n gnd   gnd   nmos  l=' + str(ind[55]) + 'n ' + 'w=' + str(ind[54]) + 'n\n'


    with open('tfa_leak_25,0.8.sp','w') as file:
         file.writelines(data)

    from subprocess import call
    call(["hspice64", "tfa_leak_25,0.8.sp"])

    with open('tfa_leak_25,0.8.ms0','r') as file:
         data=file.readlines()

#    list_of_elements=list()
#    final_list=list()
#    list_of_elements.extend([float(x) for x in data[8].split()])
#    final_list.append(list_of_elements[1])

    final_list=list()
    list_of_elements=list()
    list_of_elements.extend([float(x) for x in data[14].split()])
    final_list.append(list_of_elements[0])
    list_of_elements=list()
    list_of_elements.extend([float(x) for x in data[21].split()])
    final_list.append(list_of_elements[0])
    list_of_elements=list()
    list_of_elements.extend([float(x) for x in data[28].split()])
    final_list.append(list_of_elements[0])
    list_of_elements=list()
    list_of_elements.extend([float(x) for x in data[35].split()])
    final_list.append(list_of_elements[0])
    list_of_elements=list()
    list_of_elements.extend([float(x) for x in data[42].split()])
    final_list.append(list_of_elements[0])
    list_of_elements=list()
    list_of_elements.extend([float(x) for x in data[49].split()])
    final_list.append(list_of_elements[0])
    list_of_elements=list()
    list_of_elements.extend([float(x) for x in data[56].split()])
    final_list.append(list_of_elements[0])
    list_of_elements=list()
    list_of_elements.extend([float(x) for x in data[63].split()])
    final_list.append(list_of_elements[0])
    return final_list

def func2(ind):
    with open('tfa_del_25,0.8.sp','r') as file:
        data=file.readlines()
   #     data[57]='Mp1   nodez   nodea   vdd!   vdd!   pmos  w=' + str(ind[0]) + 'n' + ' l=' + str(ind[2]) + 'n \n'
   #     data[58]='Mn1   nodez   nodea   gndd!   gndd!   nmos  w='+  str(ind[1]) + 'n' + ' l=' + str(ind[3]) + 'n \n'


    data[33]='Mp1    1   nodea   vdd   vdd   pmos  l=' + str(ind[1]) + 'n ' + 'w=' + str(ind[0]) + 'n\n'
    data[34]='Mp2    1   nodeb   vdd   vdd   pmos  l=' + str(ind[3]) + 'n ' + 'w=' + str(ind[2]) + 'n\n'
    data[35]='Mp3    nodecon nodec   1   vdd   pmos l=' + str(ind[5]) + 'n ' + 'w=' + str(ind[4]) + 'n\n'
    data[36]='Mn1    5   nodea   gnd   gnd   nmos  l=' + str(ind[7]) + 'n ' + 'w=' + str(ind[6]) + 'n\n'
    data[37]='Mn2    5   nodeb   gnd   gnd   nmos  l=' + str(ind[9]) + 'n ' + 'w=' + str(ind[8]) + 'n\n'
    data[38]='Mn3    nodecon nodec   5   gnd   nmos l=' + str(ind[11]) + 'n ' + 'w=' + str(ind[10]) + 'n\n'
    data[39]='Mp4    4   nodea   vdd   vdd   pmos  l=' + str(ind[13]) + 'n ' + 'w=' + str(ind[12]) + 'n\n'
    data[40]='Mp5    nodecon nodeb   4   vdd   pmos  l=' + str(ind[15]) + 'n ' + 'w=' + str(ind[14]) + 'n\n'
    data[41]='Mn4    nodecon nodeb   node4   gnd   nmos l=' + str(ind[17]) + 'n ' + 'w=' + str(ind[16]) + 'n\n'
    data[42]='Mn5    node4   nodea   gnd   gnd   nmos l=' + str(ind[19]) + 'n ' + 'w=' + str(ind[18]) + 'n\n'
    data[43]='Mp6    2   nodea   vdd   vdd   pmos  l=' + str(ind[21]) + 'n ' + 'w=' + str(ind[20]) + 'n\n'
    data[44]='Mp7    2   nodeb   vdd   vdd   pmos l=' + str(ind[23]) + 'n ' + 'w=' + str(ind[22]) + 'n\n'
    data[45]='Mp8    2   nodec   vdd   vdd   pmos l=' + str(ind[25]) + 'n ' + 'w=' + str(ind[24]) + 'n\n'
    data[46]='Mp9    nodes0n nodecon 2   vdd   pmos l=' + str(ind[27]) + 'n ' + 'w=' + str(ind[26]) + 'n\n'
    data[47]='Mn6    3   nodea   gnd   gnd   nmos l=' + str(ind[29]) + 'n ' + 'w=' + str(ind[28]) + 'n\n'
    data[48]='Mn7    3   nodeb   gnd   gnd   nmos l=' + str(ind[31]) + 'n ' + 'w=' + str(ind[30]) + 'n\n'
    data[49]='Mn8    3   nodec   gnd   gnd   nmos l=' + str(ind[33]) + 'n ' + 'w=' + str(ind[32]) + 'n\n'
    data[50]='Mn9    nodes0n nodecon 3   gnd   nmos l=' + str(ind[35]) + 'n ' + 'w=' + str(ind[34]) + 'n\n'
    data[51]='Mp10   9   nodea   vdd   vdd   pmos  l=' + str(ind[37]) + 'n ' + 'w=' + str(ind[36]) + 'n\n'
    data[52]='Mp11   8   nodeb   9   vdd   pmos    l=' + str(ind[39]) + 'n ' + 'w=' + str(ind[38]) + 'n\n'
    data[53]='Mp12   nodes0n nodec   8   vdd   pmos l=' + str(ind[41]) + 'n ' + 'w=' + str(ind[40]) + 'n\n'
    data[54]='Mn10   7   nodea   gnd   gnd   nmos  l=' + str(ind[43]) + 'n ' + 'w=' + str(ind[42]) + 'n\n'
    data[55]='Mn11   6   nodeb   7   gnd   nmos    l=' + str(ind[45]) + 'n ' + 'w=' + str(ind[44]) + 'n\n'
    data[56]='Mn12   nodes0n nodec   6   gnd   nmos  l=' + str(ind[47]) + 'n ' + 'w=' + str(ind[46]) + 'n\n'
    data[57]='Mp13   nodeco  nodecon vdd   vdd   pmos l=' + str(ind[49]) + 'n ' + 'w=' + str(ind[48]) + 'n\n'
    data[58]='Mn13   nodeco  nodecon gnd   gnd   nmos l=' + str(ind[51]) + 'n ' + 'w=' + str(ind[50]) + 'n\n'
    data[59]='Mp14   nodes0  nodes0n vdd   vdd   pmos l=' + str(ind[53]) + 'n ' + 'w=' + str(ind[52]) + 'n\n'
    data[60]='Mn14   nodes0  nodes0n gnd   gnd   nmos  l=' + str(ind[55]) + 'n ' + 'w=' + str(ind[54]) + 'n\n'

    print("PRINTING DATA", data)

    with open('tfa_del_25,0.8.sp','w')as file:
        file.writelines(data)

    from subprocess import call
    call(["hspice64","tfa_del_25,0.8.sp"])

    with open('tfa_del_25,0.8.mt0','r') as file:
        data=file.readlines()
    final_list=list()

#    strings=data[4].split()
#    final_list.append(float(strings[2]))

    final_list.extend([float(x) for x in data[4].split()])
    tp = data[5].split()
    final_list.append(float(tp[0]))
    final_list.append(float(tp[1]))
#    s = 0
#    for i in final_list:
#   s += i
#   if i > DELAY_MAX :
#       return 1
#    print("UUUUUUUUUUUUUUUUUUUUUUUUUUU", s)
#    return(s/6)
    return final_list

def func3(ind):
    cost, delay, leakage = list(), list(), list()
    leakage = func1(ind)
    delay = func2(ind)
    cost.append(sum(leakage) / len(leakage))
    cost.extend(delay)
    return cost

def func4(ind):
    cost, delay, leakage = list(), list(), list()
    leakage = func1(ind)
    delay = func2(ind)
    cost.extend(leakage)
    cost.extend(delay)
    return cost


def print_krills(krills):
    for i in krills:
        print(i.X)

def l2_norm(X):
    return np.linalg.norm(X)

res = []

class Krill:
    def __init__(self, n, obj_func, ub, lb):
        self.X = [random.random() * (ub[i] - lb[i]) for i in range(n)]
        self.X = [self.X[i] + lb[i] for i in range(n)]
        self.X = np.array(self.X)
        self.N = np.zeros((n,))
        self.F = np.zeros((n,))
        self.X_best = self.X[:]
        self.cost = obj_func(self.X_best)
        self.K_best = self.cost[0]
        self.K = self.K_best


class KrillHerd():

    def pos_effect(self, X_i, X_j):
        return (X_j - X_i) / (l2_norm(X_j - X_i) + self.eps)

    def fitness_effect(self, K_i, K_j):
        print(type(K_i), type(K_j), type(self.K_ibest), type(self.K_iworst))
        return (K_i - K_j) / (self.K_ibest - self.K_iworst + self.eps)

    def collect_neighbors(self, krill_i):

        d_s = 0

        for j in range(self.nk):
            d_s += l2_norm(np.array(self.krills[j].X) - np.array(krill_i.X))

        d_s /= (5 * self.nk)

        neighbors = list()

        for j in range(self.nk):
            if l2_norm(self.krills[j].X - krill_i.X) <= d_s:
                neighbors.append(self.krills[j])

        return neighbors


    def neighbors_motion(self, krill_i):

        neighbors = self.collect_neighbors(krill_i)

        alpha_loc = 0

        for i in neighbors:
            alpha_loc += self.fitness_effect(krill_i.K, i.K) * self.pos_effect(krill_i.X, i.X)

        return alpha_loc

    def target_motion(self, krill_i, it):

        return 2 * (random.random() + it/self.iter) * self.fitness_effect(krill_i.K, self.K_ibest) * self.pos_effect(krill_i.X, self.X_ibest)

    def foraging_motion(self, krill_i, it):

        X_food = 0

        tmp = 0

        for i in range(self.nk):

            X_food += (self.krills[i].X/self.krills[i].K)

            tmp += (1/self.krills[i].K)

        X_food /= tmp

        return 2 * (1 - it/self.iter) * self.pos_effect(krill_i.X, X_food) * self.fitness_effect(krill_i.K , self.obj_func(X_food)[0])

    def target_foraging(self, krill_i):
        return self.fitness_effect(krill_i.K, krill_i.K_best) * self.pos_effect(krill_i.X, krill_i.X_best)

    def crossover(self, krill_i):
        
        C_r = 0.2 * self.fitness_effect(krill_i.K, self.K_ibest)

        X_new = list(krill_i.X)

        r = int(round(self.nk * random.random()))

        for m in range(self.n_dim):
            if (random.uniform(0, 1) < C_r):
                X_new[m] = self.krills[r].X[m]

        return X_new




    def __init__(self, n_dim = 1, obj_func = None, wn = 0.42, V_f = 0.02, wf = 0.38, eps = 1e-30, iter = 100, nk = 50,
    N_max=0.01, D_max=0.005, ub = None, lb = None, delay_max = 1e30):
        global res
        self.n_dim, self.obj_func, self.wn, self.V_f, self.wf, self.eps, self.iter, self.nk, self.N_max, self.D_max = n_dim, obj_func, wn, V_f, wf, eps, iter, nk, N_max, D_max
        self.ub = ub
        self.lb = lb
        
        self.krills = list()
        for i in range(nk):
            self.krills.append(Krill(n_dim, obj_func, self.ub, self.lb))

        # print_krills(self.krills)


        self.K_ibest = self.krills[0].K_best
        self.K_iworst = self.K_ibest
        self.X_ibest = list(self.krills[0].X_best)


        for i in range(nk - 1):
            if (self.krills[i + 1].K_best < self.K_ibest):
                self.K_ibest = self.krills[i + 1].K_best
                self.X_ibest = list(self.krills[i + 1].X_best)
            self.K_iworst = max(self.K_iworst, self.krills[i + 1].K_best)

        self.krills[0].X = [644.584317  ,   23.61469777,  376.19453913,   23.31980614, 314.72019919, 24.00193171, 197.08318777, 23.94738964, 249.94698551,   23.79239723,  249.26212746,   22.82867611, 309.57756591, 23.3390415,  432.29375918,   24.62248823, 69.86302452,   24.97343848,   74.05475382,   24.17444322, 343.48842755, 23.32701916,  438.88500902,   22.19928252, 213.74956825,   23.59555851,  895.45197254,   24.27548629, 282.12162799,   22.75535012,  226.95525205, 22.03573876, 141.93920618,   23.49121848,  569.6262448 , 22.89554662, 376.17952665,   24.35810227, 436.12129873,   23.73065596, 120.38467802, 23.39828366, 79.75771782,   24.27139221, 778.68322422, 22.67653586, 68.15509386, 23.50315675, 888.91323128, 22.10812629, 880.105946  , 23.21088792, 531.49491888, 23.8938607, 93.66091828, 23.75944689]
        

        self.krills[0].X = np.array(self.krills[0].X)
        self.X_best = list(self.krills[0].X)
        self.krills[0].cost = obj_func(self.krills[0].X_best)
        self.krills[0].K_best = self.krills[0].cost[0]
	
	i = 0

        for it in range(self.iter):
	
	    flag = 0

            while(i < nk):

            # Delay in bound check

                for j in range(len(self.krills[i].cost) - 1):
                    if self.krills[i].cost[j + 1] >= delay_max:
                        self.krills[i].X = [random.random() * (self.ub[k] - self.lb[k]) for k in range(len(self.ub))]
                        self.krills[i].X = [self.krills[i].X[k] + self.lb[k] for k in range(len(self.lb))]
                        self.krills[i].X = np.array(self.krills[i].X)
                        self.krills[i].N = np.zeros((self.n_dim,))
                        self.krills[i].F = np.zeros((self.n_dim,))
                        self.krills[i].X_best = self.krills[i].X[:]
                        self.krills[i].cost = obj_func(self.krills[i].X_best)
                        self.krills[i].K_best = self.krills[i].cost[0]
                        self.krills[i].K = self.krills[i].K_best
                        flag = 1
                        break
                if(flag):
		    continue

            # Other krills induced motion

                alpha_loc = self.neighbors_motion(self.krills[i])

                alpha_target = self.target_motion(self.krills[i], it)

                self.krills[i].N = alpha_loc * self.N_max + alpha_target * self.krills[i].N

            # Foraging Motion

                beta_food = self.foraging_motion(self.krills[i], it)

                beta_best = self.target_foraging(self.krills[i])

                self.krills[i].F = self.V_f * (beta_best + beta_food) + self.wf * self.krills[i].F

            # Physical Diffusion

                delta = ([2 * random.random() for k in range(self.n_dim)]) - np.ones((self.n_dim, ))

                D_i = self.D_max * (1 - it/self.iter) * delta

            # Time scaling factor delta_t

                delta_t = 1.5 * np.sum(np.array(ub) - np.array(lb))

            # Crossover

                self.krills[i].X = list(self.crossover(self.krills[i]))

            # Updates

                self.krills[i].X += delta_t * (self.krills[i].N + self.krills[i].F + D_i)

                for x in range(self.n_dim):
                    if (self.krills[i].X[x] > self.ub[x]):
                        self.krills[i].X[x] = self.lb[x] + (self.ub[x] - self.lb[x]) * random.random()
                    if (self.krills[i].X[x] < self.lb[x]):
                        self.krills[i].X[x] = self.lb[x] + (self.ub[x] - self.lb[x]) * random.random()
                    # self.krills[i].X[x] = min(self.krills[i].X[x], self.ub[0])
                    # self.krills[i].X[x] = max(self.krills[i].X[x], self.lb[0])

                self.krills[i].cost = self.obj_func(self.krills[i].X)
                self.krills[i].K = self.krills[i].cost[0]

                if (self.krills[i].K < self.krills[i].K_best):
                    self.krills[i].K_best = self.krills[i].K
                    self.krills[i].X_best = list(self.krills[i].X)

                # if (self.krills[i].K < self.K_ibest):
                #   self.K_ibest = self.krills[i].K
                #   self.X_ibest = list(self.krills[i].X)
                #   res = i
                #   print("HERD", i, self.krills[i].X, res, self.K_ibest)
                # elif (self.krills[i].K > self.K_iworst):
                #   self.K_iworst = self.krills[i].K


            for i in range(nk - 1):
                if (self.krills[i + 1].K_best < self.K_ibest):
                    self.K_ibest = self.krills[i + 1].K_best
                    self.X_ibest = list(self.krills[i + 1].X_best)
                    print("HERD", i, self.krills[i + 1].X_best, res, self.K_ibest)
                self.K_iworst = max(self.K_iworst, self.krills[i + 1].K_best)

                
                # if i == 0:

                #   print("KRILL", i)

                #   print("X", self.krills[i].X)

                #   print("K", self.krills[i].K)

                #   print("N", self.krills[i].N)

                #   print("F", self.krills[i].F)

                #   print("D", D_i)

                # time.sleep(5)

        print("FINAL", self.X_ibest, self.K_ibest)

            # time.sleep(5)




def peak(X):
    return X[0] * math.exp(-(X[0]**2 + X[1]**2))

def sphere(X):
    res = 0
    for i in X:
        res += i ** 2
    return res

def main():
    l1 = [1000, 25] * 28 
    l2 = [44, 22] * 28
    #print(ub, lb)
    KrillHerd(n_dim=56, obj_func=func3, ub=l1, lb=l2, iter=100, nk=20, delay_max=9.85784e-12)

if __name__ == "__main__":
    main()


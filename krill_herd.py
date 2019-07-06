import numpy as np
import random
import math
import time


def print_krills(krills):
	for i in krills:
		print(i.X)

def l2_norm(X):
	return np.linalg.norm(X)

res = []

class Krill:
	def __init__(self, n, obj_func, ub, lb):
		self.X = lb + [random.random() for i in range(n)] * (ub - lb)
		self.N = np.zeros((n,))
		self.F = np.zeros((n,))
		self.X_best = self.X[:]
		self.K_best = obj_func(self.X_best)
		self.K = self.K_best


class KrillHerd():

	def pos_effect(self, X_i, X_j):
		return (X_j - X_i) / (l2_norm(X_j - X_i) + self.eps)

	def fitness_effect(self, K_i, K_j):
		return (K_i - K_j) / (self.K_ibest - self.K_iworst + self.eps)

	def collect_neighbors(self, krill_i):

		d_s = 0

		for j in range(self.nk):
			d_s += l2_norm(self.krills[j].X - krill_i.X)

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

		return 2 * (1 - it/self.iter) * self.pos_effect(krill_i.X, X_food) * self.fitness_effect(krill_i.K , self.obj_func(X_food))

	def target_foraging(self, krill_i):
		return self.fitness_effect(krill_i.K, krill_i.K_best) * self.pos_effect(krill_i.X, krill_i.X_best)



	def __init__(self, n_dim = 1, obj_func = None, wn = 0.42, V_f = 0.02, wf = 0.38, eps = 1e-30, iter = 100, nk = 50, N_max=0.01, D_max=0.005, ub=1e10, lb=-1e10):
		global res
		self.n_dim, self.obj_func, self.wn, self.V_f, self.wf, self.eps, self.iter, self.nk, self.N_max, self.D_max = n_dim, obj_func, wn, V_f, wf, eps, iter, nk, N_max, D_max
		self.ub = ub * np.ones((n_dim, ))
		self.lb = lb * np.ones((n_dim, ))
		
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

		for it in range(self.iter):

			for i in range(nk):
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

				delta_t = 1.5 * np.sum(ub - lb)

			# Updates

				self.krills[i].X += delta_t * (self.krills[i].N + self.krills[i].F + D_i)

				for x in range(self.n_dim):
					if (self.krills[i].X[x] > self.ub[x]):
						self.krills[i].X[x] = self.lb[x] + (self.ub[x] - self.lb[x]) * random.random()
					if (self.krills[i].X[x] < self.lb[x]):
						self.krills[i].X[x] = self.lb[x] + (self.ub[x] - self.lb[x]) * random.random()
					# self.krills[i].X[x] = min(self.krills[i].X[x], self.ub[0])
					# self.krills[i].X[x] = max(self.krills[i].X[x], self.lb[0])

				self.krills[i].K = self.obj_func(self.krills[i].X)

				if (self.krills[i].K < self.krills[i].K_best):
					self.krills[i].K_best = self.krills[i].K
					self.krills[i].X_best = self.krills[i].X[:]

				# if (self.krills[i].K < self.K_ibest):
				# 	self.K_ibest = self.krills[i].K
				# 	self.X_ibest = list(self.krills[i].X)
				# 	res = i
				# 	print("HERD", i, self.krills[i].X, res, self.K_ibest)
				# elif (self.krills[i].K > self.K_iworst):
				# 	self.K_iworst = self.krills[i].K


			for i in range(nk - 1):
				if (self.krills[i + 1].K_best < self.K_ibest):
					self.K_ibest = self.krills[i + 1].K_best
					self.X_ibest = list(self.krills[i + 1].X_best)
					print("HERD", i, self.krills[i + 1].X_best, res, self.K_ibest)
				self.K_iworst = max(self.K_iworst, self.krills[i + 1].K_best)

				
				# if i == 0:

				# 	print("KRILL", i)

				# 	print("X", self.krills[i].X)

				# 	print("K", self.krills[i].K)

				# 	print("N", self.krills[i].N)

				# 	print("F", self.krills[i].F)

				# 	print("D", D_i)

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
	KrillHerd(n_dim=3, obj_func=sphere, ub=10, lb=-10, iter=100, nk=50)
	# print(res)
	# X = [0.17248473654989216, -0.3643013151473621, -1.883153284994198, -2.5369518187105893, -2.4602427456130727]
	# print(X)
	# print(sphere(X))

if __name__ == "__main__":
	main()
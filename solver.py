class SolutionInfo():
	def __init__(self):
		self.obj = float()
		self.rho = float()
		self.upper_bound_p = float()
		self.upper_bound_n = float()
		self.r = float()

	@property
	def obj(self):
		return self.obj

	@propety
	def rho(self):
		return self.rho

	@property
	def upper_bound_p:
		return self.upper_bound_p
	
	@property
	def upper_bound_n:
		return self.upper_bound_n

	@proper
	def r:
		return self.r


class solver:
	def get_C(self, i):
		if self.y[i] > 0:
			return Cp
		else:
			return Cn

	def update_alpha_status(self, i)
		if self.alpha[i] >= self.UPPER_BOUND:
			self.alpha_status[i] = self.UPPER_BOUND
		elif self.alpha[i] < 0:
			self.alpha_status = self.LOWER_BOUND
		else:
			alpha_status[i] = self.FREE

	def is_upper_bound(self, i):
		return self.alpha_status[i] == self.UPPER_BOUND

	def is_lower_bound(self, i):
		return self.alpha_status[i] == self.LOWER_BOUND

	def is_free(self, i):
		return self.alpha_status[i] == self.FREE

	def swap_index(self, i, j):
		print "foo"

	def reconstruct_gradient(self):
 		if self.active_size != self.l:
			nr_free = int()

			for j in range(self.active_size,self.l):
				self.G[j] = self.G_bar[j] + self.p[j]

			for j in range(self.active_size):
				if self.is_free(j):
					nr_free += 1

			if 2 * nr_free < active_size:
				print "Warning: using -h 0 may be faster"

			if nr_free * l > 2 * active_size * (l-active_size):
				for i in range(active_size, l):
					Q_i = Q.get_Q(i, self.active_size)
					for j in range(self.active_size):
						if self.is_free(j):
							G[j] += alpha[j] * Q_i[j]

			else:
				for i in range(self.active_size):
					if self.is_free(j):
						Q_i = Q.get_Q(i, self.active_size)
						alpha_i = alpha[i]
						for j in range(active_size):
							self.G[j] += self.alpha_i * self.Q_i[j]
	
	def solve(self, l, Q, p_, y_, alpha_, Cp, Cn, eps, si, shrinking):
		self.l = l
		self.Q = Q
		QD=Q.get_QD()
		p = p_
		y = y_
		alpha = alpha_
        self.Cp = Cp
		self.Cn = CN
		self.eps = eps
		unshrink = false

	def select_working_set(self, out_i, out_j):
		print "Foo "

	def be_shrunk(self, i , Gmax1, Gmax2):
		if self.is_upper_bound(i):
			if self.y[i] == 1:
				return -self.G[i] > self.Gmax1
			else:
				return -self.G[i] > self.Gmax2

		elif self.is_lower_bound(i):
			if self.y[i] == 1:
				return self.G[i] > self.Gmax2
			else:
				return self.G[i] > self.Gmax1
		else:
			return False

	def do_shrinking(self):
		self.Gmax1 = self.mINF
		self.Gmax2 = self.mINF
		
		for i in range(self.active_size):
			if self.y[i] == 1:
				if not self.is_upper_bound(i) and -self.G[i] >= Gmax1:
					self.Gmax1 = -self.G[i]
				if not self.is_lower_bound(i) and self.G[i] >= Gmax2:
					self.Gmax2 = self.G[i]
			else:
				if not self.is_upper_bound(i) and -self.G[i] >= Gmax2:
					self.Gmax2 = -self.G[i]
				if not self.is_lower_bound(i) and self.G[i] >= Gmax1:
					self.Gmax1 = self.G[i]
		
		if self.unshrink == False and self.Gmax1 + self.Gmax2 <= eps * 10:
			self.unshrink = True
			self.reconstruct_gradient()
			sel.active_size = self.l
			print "*",

		for i in range(self.active_size):
			if self.be_shrunk(i, Gmax1, Gmax2):
				active_size -= 1
				while active_size > i:
					if not self.be_shrunk(self.active_size, self.Gmax1, self.Gmax2):
						self.swap_index(i,self.active_size):
						break
					active_size -= 1

	def calculate_rho(self):
		nr_free = int()
		ub = self.INF
		lb = self.mINF
		sum_free = float()

		for i in range(self.active_size):
			yG = self.y * self.G[i]
			if self.is_upper_bound(i):
				if self.y[i] == -1:
					ub = min(ub, yG)
				else:
					lb = max(lb, yG)
			elif self.is_lower_bound(i):
				if self.y[i] == 1:
					ub = min(ub, yG)
				else:
					lb = max(lb, yG)
			else:
				nr_free += 1
				sum_free += yG
		
		if nr_free > 0:
			r = sum_free/nr_free
		else:
			r = (ub + lb)/2
		return r

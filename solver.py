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
	#TODO Finish this stuff
	def __init__(self):
		self.active_size = int()
		self.y = None #TODO = numpy.array()
		self.G = None #FIXME
		self.alpha_status = ''

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
    
	#TODO
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
	#TODO 
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

		# initialize alpha_status
		alpha_status = numpy.zeros(l, dtype = int)
		for i in range(l):
			update_alpha_status(i)

		# initialize active_size (for shrinking)
		active_set = numpy.zeros(l, dtype = int)
		for i in range(l):
			active_set[i] = i
		active_size = l

		# initialize gradient
		G = numpy.zeros(l, dtype = float)
		G_bar = numpy.zeros(l, dtype = float)
		for i in range(l):
			G[i] = p[i]
		for i in range(l):
			if self.is_lower_bound(i):
				Q_i = Q.get_Q(i,l)
				alpha_i = alpha[i]
				for j in range(l):
					G[j] += alpha_i*Q_i[j]
				if self.is_upper_bound(i):
					for j in range(l):
						G_bar += self.get_C(i) * Q[j]

		# Optimization step
        iter = int()
		counter = min(l, 1000) = 1

		while True:
			counter -= 1
			if (counter == 0):
				counter = min(l, 1000)
				if shrinking:
					self.do_shrinking()
				print".", #TODO another damn info

			i = int()
			j = int()
			if self.select_working_set(i,j)  !=0:
				self.reconstruct_gradient()
				active_size = l
				print "*", #TODO info
				if self.select_working_set(i,j) !=0
					break
				else
					counter = 1

			iter += 1

			Q_i = Q.get_Q(i, active_size)
			Q_j = Q.get_Q(j, active_size)

			C_i = self.get_C(i)
			C_j = self.get_C(j)

			old_alpha_i = alpha[i]
			old_alpha_j = alpha[j]

			if y[i] != y[j]:
				quad_coef = QD[i] + QD[j] + 2* Q_i[j]
				if quad_coef <= 0
					quad_coef = self.TAU
				delta  = (-G[i]-G[j])/ quad_coef
				diff = alpha[i] - alpha[j]
				alpha[i] += delta
				alpha[j] += delta

				if diff > 0 and alpha[j] <0:
					alpha[j] = 0
					alpha[i] = diff
				elif alpha[i] < 0:
					alpha[i] = 0
					alpha[j] = -diff
				if diff > C_i - C_j and alpha[i] > C_i:
					alpha[i] = C_i
					alpha[j] = C_i - diff
				elif alpha[j] > C_i:
					alpha[j] = C_j
					alpha[i] = C_j + diff
			else:
				quad_coef = QD[i] + QD[j] - 2*Q_i[j]
				if quad_coef <=0
					quad_coef = self.TAU
				delta = (G[i] - G[j])/quad_coef
				sum = alpha[i] + alpha[j]
				alpha[i] -= delta
				alpha[j] += delta

                if sum > C_i and alpha[i] > C_i:
					alpha[i] = C_i
					alpha[j] = sum - C_i
				elif alpha[j] < 0:
					alpha[j] = 0
					alpha[i] = sum
				if sum > C_j and alpha[j] > C_j:
					alpha[j] = C_j
					alpha[i] = sum - C_j
				elif alpha[i] < 0:
					alpha[i] = 0
					alpha[j] sum

			# update G
			delta_alpha_i = alpha[i] - old_alpha_i
			delta_alpha_j = alpha[j] - old_alpha_j

			for k in range(active_size):
				G[k] += Q_i[k]*delta_alpha_i + q_j[k]*delta_alpha_j

			# update alpha_status and G_bar

			ui = self.is_upper_bound(i)
			uj = self.is_upper_bound(j)
			update_alpha_status(i)
			update_alpha_status(j)
			if ui != self.is_upper_bound(i):
				Q_i = Q.get_Q(i,l)
				if ui:
					for k in range(l):
						G_bar[k] -= C_i * Q_i[k]
				else:
					for k in range(l):
						G_bar[k] += C_i * Q_i[k]
			if uj != self.is_upper_bound(j):
				Q_j = Q.get_Q(j,l)
				if uj:
					for k in range(l):
						G_bar[k] -= C_j * Q_j[k]
				else:
					for k in range(l):
						G_bar[k] += C_j * Q_j[k]

			# Calculate Rho
			si.rho = calculate_rho()

			# Calculate Objective Values
			v = 0
			for i in range(l):
				v += alpha * (G[i] + p [i])
			si.obj = v/2

			# Put back the solution
			for i in range(l):
				alpha_[active_set[i]] = alpha

			si.upper_bound_p = Cp
			si.upper_bound_n = Cn

			print "\n Optimization finished, #iter = " + str(iter)

	def select_working_set(self, out_i, out_j):
		Gmax = -INF
		Gmax2 = -INF
		Gmax_idx = -1
		Gmin_idx = -1
		obj_diff_min = INF

		for t in range(self.active_size):
			if(self.y[t] == 1):
				if not self.is_upper_bound(t):
					if -G[t] >= Gmax:
						Gmax = -G[t]
						Gmax_idx = t
			else:
				if not self.is_lower_bound(t):
					if G[t] > = Gmax:
						Gmax = G[t]
						Gmax_idx = t
		i = Gmax_idx
		Q_i = None
		if i != -1:
			Q_i = Q.get_Q(i, self.active_size)

		for j in range(self.active_size):
			if (self.y[j] == 1):
				if not self.is_lower_bound(j):
					grad_diff = Gmax + self.G[j]
					if grad_diff > 0:
						quad_coef = self.QD[i] + self.QD[j] - 2.0*self.y[i]*Q_i[j]
						if quad_coed > 0:
							obj_diff = - (grad_diff * grad_diff) /quad_coef
						else:
							obj_diff = -(grad_diff * grad_diff) / self.TAU

						if obj_diff < = obj_diff_min
							Gmin_idx = j
							obj_diff_min = obj_diff
			else:
				if self.is_upper_bound(j):
					grad_diff = Gmax - self.G[j]
					if -self.G[j] >= Gmax2:
						Gmax2 = -self.G[j]
					if self.grad_diff > 0:
						quad_coef = self.QD[i] + self.QD[j] + 2.0 * self.y[i]*Q_i[j]
						if quad_coef > 0:
							obj_diff = -(grad_diff * grad_diff) / quad_coef
						else:
							obj_diff = -(grad_diff* grad_diff) / self.TAU

						if obj_diff < = obj_diff_min:
							Gmin_idx = j
							obj_diff_min = obj_diff
		if Gmax + Gmax2 < eps:
			return 1

		out_j = Gmax_idx
		out_i = Gmin_idx
		return 0

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
				self.active_size -= 1
				while active_size > i:
					if not self.be_shrunk(self.active_size, self.Gmax1, self.Gmax2):
						self.swap_index(i,self.active_size):
						break
					self.active_size -= 1

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

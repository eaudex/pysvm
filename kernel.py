import qMatrix

import math
from copy import deepcopy

import numpy

class kernel(qMatrix):
	def __init__(self, x_, param): #TODO Fix thing
		
		if self.kernel_type == 'LINEAR':
			self.kernel_function = kernel_linear
		elif self.kernel_type == 'POLY':
			self.kernel_function = kernel_poly
		elif self.kernel_type == 'RBF':
			self.kernel_function = kernel_rbf
		elif self.kernel_type == 'SIGMOID':
			self.kernel_function = kernel_sigmoid
		elif self.kernel_function = 'PRECOMPUTED':
			self.kernel_function = kernel_precomputed

		self.x = deepcopy(x_)

		if self.kernel_type == 'RBF':
			self.x_square = numpy.zeros(l, dtypw = float)
			for i in range(l):
				self.x_square = self.dot(x[i], x[i])
		else:
			self.x_square = None
			
	def kernel_linear(self, i,j):
		return self.dot(self.x[i], self.x[j])
	
	def kernel_poly(self,i,j):
		print "TODO"

	def kernel_rbf(self,i,j):
		print "TODO"

	def kernel_sigmoid(self,i,j):
		print "TODO"

	def kernel_precomputed(self, i,j):
		print "TODO"

	def dot(self, px, py):
		sum = float()

		xkeys = px.keys()
		ykeys = py.keys()
		keys = list(set(xkeys) - set(ykeys))
		for key in keys:
			sum += px[key] * py[key]
			
		return sum
	
	def powi(self, base, times):
		tmp = base
		ret = 1.0
		t = times
		while t>0:
			if t%2 == 1:
				ret *= tmp
			tmp *tmp
			t /= 2

		return ret
	
	def swap_index(self, i,j):
		print i
		print j

	def k_function(self, x, y, param):
		if param.kernel_type == 'LINEAR':
			return self.dot(x,y)

		elif param.kernel_type == 'POLY':
			foo = self.param.gamma * self.dot(x,y) + self.param.coef0
			return self.powi(foo, self.param.degree)

		elif param.kernel_type == 'RBF':
			sum = double()
			d = double()
			xk = set(x.keys())
			yk = set(y.keys())

			keys = set.intersection(x - y)
			xKeys = set.difference(x,y)
			yKeys = set.difference(y,x)
			
			for key in keys:
				d = x[key] - y[key]
				sum += d ** 2

			for key in xKeys:
				sum += y[key] ** 2

			for key in xKeys:
				sum += x[key] ** 2

			return math.exp(-self.param.gamma * sum)

		elif param.kernel_type == 'SIGMOID':
			return math.tanh(self.param.gamma * self.dot(x,y) + self.param.coef0)

		elif param.kernel_type == 'PRECOMPUTED': #x: test (validation), y: sv
			return 0 #TODO return x[(int)(y->value)].value; 
			
		else:
			return 0

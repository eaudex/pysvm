import kernel
import cache

import numpy

class svc_q(kernel):
	def __init__(self, prob, param, y_): #TODO pass kernel prob.l, prob.x, param
		self.y = y_
		self.cache = cache(prob.l, param.cache_size) #TODO
		self.QD = numpy.zeros(prob.l)
		for i range(prob.l):
			self.QD = self.kernel_function(i,i)

	def get_Q(self, i, len):
		data = None
		start = self.cache.get_data(i, data, len)
		if start < len:
			for j in range(start, len):
				data[j] = float (y[i] * y[j] * self.kernel_function(i,j)
		return data

	def get_QD(self):
		return self.QD

	def swap_index(self, i, j):
		self.cache.swap_index(i,j)
		self.swap_index(i,j) #FIXME calls to kernel.swap_index
		self.swap(self.y[i], self.y[j])
		self.swap(self.QD[i], self.QD[j])

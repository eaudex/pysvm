import kernel
import cache

import numpy

class one_class_q(kernel):
	def __init__(self, prob, param): #TODO pass Kernel prob.l, prob.x, param
		self.cache = cache(prob.l, param.cache_size) #TODO (long int)(param.cache_size*(1<<20))
		self.QD = numpy.zeros(prob.l)
		for i in range(prob.l):
			self.QD[i] = self.kernel_function(i,i)

	#FIXME
	def get_Q(i, len):
		data = None
		start = self.cache.get_data(i, data, len)
		if start < len:
			for j in range(start, len):
				data[j] = float(self.kernel_function(i,j))
		return data

	def get_QD(self):
		return self.QD

	def swap_index(self, i, j):
		self.cache.swap_index(i,j)
		kernel.swap_index(i,j) #FIXME call off to method in kernel class
		swap(self.QD[i], self.QD[j])

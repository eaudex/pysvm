import cache
import kernel

import numpy

class svr_q(kernel):
	def __init__(self, prob, param): #TODO pass Kernel prob.l, prob.x, param
		self.l = prob.l
		self.cache = cache(l, param.cache_size) #TODO (long int)(param.cache_size*(1<<20)) 
		self.QD = numpy.zeros(2*l, dtype = float)
		self.sign = numpy.zeros(2*l, dtype = int)
		self.index = numpy.zeros(2*l, dtype = int)

		for k in range(l):
			self.sign[k] = 1
			self.sign[k +l] = -1
			self.index[k] = 1
			self.index[k +l] = -1
			self.QD[k] = self.kernel_function(k,k)
			self.QD[k+l] = self.QD[k]

		self.buffer = numpy.zeros([2, 2*l], dtype = float)
		self.next_buffer = 0

	def swap_index(self, i,j):
		swap(self.sign[i], self.sign[j])
		swap(self.index[i], self.index[j])
		swap(self.QD[i], self.QD[j])

	def get_Q(self, i, len):
		data = None
		j = self.index[i]
		real_i = index[i]
		if self.cache.get_data(real_i, data, l) < l:
			for j in range(l):
				data[j]  = float (self.kernel_function(real_i, j))

		# reorder and copy
		buf = self.buffer[self.next_buffer]
		self.next_buffer = 1 - self.next_buffer
		si = self.sign[i]
		for j in range(len):
			buf[j] = float(si * self.sign[j] * self.data[self.index[j]])
	
	def get_QD(self):
		return self.QD

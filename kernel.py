import qMatrix
import math
import numpy

class kernel(qMatrix):
	def __init__(self, x, param):
		print "Woot"

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
		print "TODO fooy"
	
	def powi(self, base, times):
		print "STUBBS"
	
	def swap_index(self, i,j):
		print i
		print j

	def k_function(self, x, y, param):
		if param.kernel_type == 'LINEAR':
			return self.dot(x,y)
		elif param.kernel_type == 'POLY':
			return self.powi()
		elif param.kernel_type == 'RBF':
			print "Fook"
		elif param.kernel_type == 'SIGMOID':
			print "ahahah"
		elif param.kernel_type == 'PRECOMPUTED':
			print "q"
		else:
			return 0

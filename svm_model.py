import svm_paramter

class model:
	def __init__(self):
		self.param = svm_paramter()
		self.nr_class = 0
		self.l = 0
		self.SV = {}
		self.sv_coef = {}
		
		self.rho = 0.0
		self.probA = 0.0
		self.probB = 0.0
		self.free_sv = false

	@property
	def param(self):
		return self.param

	@property
	def free_sv(self):
		return self.free_sv


class svm_problem:
	def __init__(self):
		self.l = 0
		self.y = 0.0 #Target
		self.x = {} #Feature,Value
	@property
	def l(self):
		return self.l
	
	@property
	def y(self):
		return self.y

	@property
	def x(self):
		return self.x

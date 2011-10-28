class parameter:
	def __init__(self):
		self.svm_type = 'C_SVC'
		self.kernel_type = 'RBF'
		self.degree = 3
		self.gamma =  0.0
        self.coef0 = 0.0
		self.nu = 0.5
		self.cache_size = 100
		self.C = 1
		self.eps = 1*10**-3
		self.p = 0.1
		self.shrinking = 1
		self.probability = 0
		self.nr_weight = 0
		self.weight_label = None
		self.weight = None
		
	@property
	def svm_type(self):
		return self.svm_type

	@property
	def kernel_type(self):
		return self.kernel_type

	@property
	def degree(self):
		return self.degree

	@property
	def gamma(self):
		return self.gamma
	
	@property
	def coef0(self):
		return self.coef0

	@property
	def nu(self):
		return self.nu

	@property
	def cache_size(self):
		return cache_size
	
	@property
	def C(self):
		return C
	
	@property
	def eps(self):
		return eps

	@property
	def p(self):
		return p

	@property
	def shrinking(self):
		return shrinking
	
	@property
	def probability(self):
		return probability

	@property
	def nr_weight(self):
		return nr_weight
	
	@property
	def weight_label(self):
		return weight_label

	@property
	def weight(self):
		return weight

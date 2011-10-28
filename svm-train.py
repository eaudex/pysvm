import os
import sys
import getopt

import numpy

import svm
import svm_parameter
import svm_problem
import svm_model

class svm_train:
	def __init__(self):
		self.svm = svm()
		self.param = svm_parameter()
		self.prob = svm_problem()
		self.x_space = None
		self.cross_validation = False
		self.nr_fold = 0
		self.quiet = False

	def exit_with_help(self):
		print """
	Usage: svm-train [options] training_set_file [model_file]
	options:
	-s svm_type : set type of SVM (default 0)
		0 -- C-SVC
		1 -- nu-SVC
		2 -- one-class SVM
		3 -- epsilon-SVR
		4 -- nu-SVR
	-t kernel_type : set type of kernel function (default 2)
		0 -- linear: u'*v
		1 -- polynomial: (gamma*u'*v + coef0)^degree
		2 -- radial basis function: exp(-gamma*|u-v|^2)
		3 -- sigmoid: tanh(gamma*u'*v + coef0)
		4 -- precomputed kernel (kernel values in training_set_file)
	-d degree : set degree in kernel function (default 3)
	-g gamma : set gamma in kernel function (default 1/num_features)
	-r coef0 : set coef0 in kernel function (default 0)
	-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
	-m cachesize : set cache memory size in MB (default 100)
	-e epsilon : set tolerance of termination criterion (default 0.001)
	-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
	-w weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
	-v n: n-fold cross validation mode
	-q : quiet mode (no outputs)                               
			"""
		sys.exit(1)
	
	def exit_input_error(self, line_num):
		print "Wrong input format at line " + line_num
		sys.exit(1)

	def main(self, args):
		self.parse_command_line(args)
		self.read_problem()
		error_msg = self.s.svm_check_parameter()

		if not error_msg is None:
			print "Error: " + error_msg
			sys.exit(-1)

		if self.cross_validation:
			self.do_cross_validation()
		else:
			self.model = self.s.svm_train(self.prob, self.param)
			self.s.svm_save_model(self.model_file_name, self.model)

	def do_cross_validation(self):
		i = 0
		total_correct = 0.0
		total_error = 0.0
		sumv, sumy, sumvv, sumyy, sumvy = 0.0, 0.0, 0.0, 0.0, 0.0

		target = self.s.svm_cross_validation(self.prob, self.param, self.nr_fold)
		if self.param.svm_type == ['EPSILON_SVR','NU_SVR']:
			for i in range(self.prob.l):
				y = self.prob.y[i]
				v = target[i]
				total_error += (v-y)*(v-y)
				sumv += v
				sumy += y
				sumvv += v*v
				sumyy += y*y
				sumvy += v*y
			print "Cross Validation Mean squared error = " + total_error/self.prob.l
			print "Cross Validation Squared correlation coefficient = " + ((self.prob.l*sumvy-sumv*sumy)*(self.prob.l*sumvy-sumv*sumy)/(self.probe.l*sumvv-sumv*sumv)-(self.probe.l*sumyy-sumy*sumy))
		else:
			for i in range(self.prob.l):
				if target[i] == self.prob.y[i]:
					total_correct += 1
			print "Cross Validation Accuracy = " + 100*total_correct/self.prob.l

	def parse_command_line(self, args):
		try:
			opts, args = getopt.getopt(args, "s:t:d:g:r:c:n:p:m:e:h:b:w:v:q")
			for opt, arg in opts:
				if opt == 's':
					if int(arg) == 0:
						self.param.svm_type = 'C_SVC'
					elif int(arg) == 1:
						self.param.svm_type = 'NU_SVC'
					elif int(arg) == 2:
						self.param.svm_type = 'ONE_CLASS'
					elif int(arg) == 3:
						self.param.svm_type = 'EPSILON_SVR'
					elif int(arg) == 4:
						self.param.svm_type = 'NU_SVR'
				elif opt == 't':
					if int(arg) == 0:
						self.param.kernel_type = 'LINEAR'
					elif int(arg) == 1:
						self.param.kernel_type = 'POLY' 
					elif int(arg) == 2:
						self.param.kernel_type = 'RBF' 
					elif int(arg) == 3:
						self.param.kernel_type = 'SIGMOID' 
					elif int(arg) == 4:
						self.param.kernel_type = 'PRECOMPUTED'  	
				elif opt == 'd':
					self.param.degree = int(arg)
				elif opt == 'g':
					self.param.gamma = double(arg)
				elif opt == 'r':
					self.param.coed0 = double(arg)
				elif opt == 'n':
					self.param.nu = double(arg)
				elif opt == 'm':
					self.param.cache_size = double(arg)
				elif opt == 'c':
					self.param.C = double(arg)
				elif opt == 'e':
					self.param.eps = double(arg)
				elif opt == 'p':
					self.param.p = double(arg)
				elif opt == 'h':
					self.param.shrinking = int(arg)
				elif opt == 'p':
					self.param.probability = int(arg)
				elif opt == 'q':
					self.quiet = True
				elif opt == 'v':
					self.cross_validation = true
					self.nr_fold = int (arg)
					if self.nr_fold < 2:
						print "n-fold cross validation: n must >= 2"
						self.exit_with_help()
				elif opt == 'w':
					self.param.nr_weight += 1
					self.param.weight_label = numpy.zeros(self.param.nr_weight, dtype = int)
					self.param.weight = numpy.zeros(self.param.nr_weight, dtype = float)
					self.param.weight_label[-1] = int(arg) #FIXME -> atoi(&argv[i-1][2]); 
					self.param.weight[-1] = float(arg)
					
				else:
					print "Unknown option: ", opt
					self.exit_with_help()

            self.s.svm_set_print_string_function(self.quiet)
			
			if len(args) == 0:
				self.exit_with_help()
				
			self.input_file_name == args[0]
			if len(args) >= 2:
				self.model_file_name = args[1]
			else:
				self.model_file_name = os.path.basename(args[1]) + ".model"

		except getopt.GetoptError:
			print getopt.GetoptError
			usage()
			sys.exit(1)

    
	"""
	Read input file base upon svm_light file format as defined by
	<line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
	<target> .=. +1 | -1 | 0 | <float> 
	<feature> .=. <integer> | "qid"
	<value> .=. <float>
	<info> .=. <string>
	"""	
	def read_problem(self):
		numberLines = 0
		try:
			for line in open(self.input_file_name):
				numberLines += 1
				elems = line.split(' ')
                self.prob.y = float(elems.pop(0))
				for e in elems:
					if '#' in e: #Hit a comment
						break
					else
						feature, value = e.split(':')
                        self.prob.x[feature] = value

			self.prob.l = numberLines
			#TODO
			#if self.param.gamma == 0 and max_index > 0:
			#	self.param.gamma = 1.0 / max_index

			#if self.param.kernel_type == 'PRECOMPUTED':
			#	for i in range(self.prob.l):
			#		if 
		except:
			print "Error opening problem file"
			sys.exit(1)
	   

	   

	
if __name__ == "__main__":
	s = svm_train()
	s.main(sys.argv[1:])


import math
import random

import numpy

import svm_model
import svm_problem
import svm_paramter
import kernel
import decision_function

class svm:
	def __init__(self):
		print "Fooy"
	 
    
	def swap(self, i, j):
		print "foo"
	#
	# Construct and solve various formulations 
	# 
	def solve_c_svc(self):
		print "foo"
	
	def solve_nu_svc(self):
		print "foo"

	def solve_one_class(self):
		print "foo"

	def solve_epsilon_svr(self):
		print "foo"

	def solve_nu_svr(self, prob, param, alpha, si):
		l = prob.l
		C = param.C
		alpha2 = numpy.zeros(2*l, dtype = float)
		linear_term = numpy.zeros(2*l,dtype = float)
		

	#
	# Decision Functions
	#
	def svm_train_one(self, prob, param, Cn, Cn):
		alpha = numpy.zeros(prob.l, dtype = float)
		si = Solver.SolutionInfo()

		if param.svm_type is 'C_SVC':
			solve_c_svc(prob, param, si, Cp, Cn)
		elif param.svm_type is 'NU_SVC':
			solve_nu_svc(prob, param, si, Cp, Cn)
		elif param.svm_type is 'ONE_CLASS':
			solve_one_class(prob, param, si, Cp, Cn)
		elif param.svm_type is 'EPSILON_SVR':
			solve_epsilon_svr(prob, param, si, Cp, Cn)
		elif param.svm_type is 'NU_SVR':
			solve_nu_svr(prob, param, si, Cp, Cn)

		#TODO info("obj = %f, rho = %f\n",si.obj,si.rho);

		# output SVs
		nSv = int()
		nBSV = int()
		for i in range(prob.l):
			if math.fabs(alpha[i]) > 0:
				nSV += 1
				if prob.y[i] > 0 and math.fabs(alpha[i]) >= si.upper_bound_p):
					nBSV += 1
			elif math.fabs(alpha[i]) >= si.upper_bound_n:
				nBSV += 1

		#TODO info("nSV = %d, nBSV = %d\n",nSV,nBSV);
		f = decision_function()
		f.alpha = alpha
		f.rho = si.rho
		return f
		
	def sigmoid_train(self):
		print "foo"

	def sigmoid_predict(decision_value, A, B ):
		fApB = decision_value * A + B
		if fApB > = 0
			return exp(-fApB/(1.0 + exp(-fApB))
		else:
			return 1.0/(1 + exp(fApB))

	def multiclass_probability(self):
		print "foo"

	def svm_binary_svc_probability(self):
		print "foo"

	def svm_svr_probability(self, prob, param):
		nr_fold = 5
		ymv = numpy.array(prob.l, dtype = float)
		mae = float()
		
        newparam = svm_parameter()
		newparam.probability = 0
		svm_cross_validation(prob, newparam, nr_fold, ymv)
		for i in range(prob.l):
			ymv[i] = prob.y[i] - ymv[i]
			mae += math.fabs(ymv[i])
		mae /= prob.l
		std = math.sqrt(2*mae*mae)
		count = 0
		mae = 0
		for i in range(prob.l):
			if math.fabs(ymv[i]) > 5 * std:
				count += 1
			else:
				mae += fabs(ymv[i])
		mae /= (prob.l - count)
		# TODO info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae)
		return mae

	def svm_group_classes(self, prob, nr_class_retm label_ret, start_ret, count_ret, perm):
		l = prob.l
		max_nr_class = 16
		nr_class = 0
		label = numpy.zeros(max_nr_class, dtype = int)
		count = numpy.zeros(max_nr_class, dtype = int)
		data_label = numpy.zeros(l, dtype = int)

		for i in range(l):
			this_label = (int)prob.y[i]
			j = 0
			for k in range(nr_class):
				j = k
				if this_label == label[j]:
					count[j] += 1
					break
			
			data_label[i] = j
			if j is nr_class:
				if nr_class is max_nr_class:
					max_nr_class *= 2
					zero = numpy.zero(max_nr_class, dtype = int)
					label = numpy.append(label, zero)
					count = numpy.append(label, zero)
				label[nr_class] = this_label
				count[nr_class] = 1
				nr_class += 1
				
				start = numpy.zeros(nr_class, dtype = int)
				start[0] = 0
				
			for i in range(1,nr_class):
				start[i] = start[i -1] + count[i -1]
			for i in range(l):
				perm[start[data_label[i]]] = i
				start[data_label[i]] += 1
			start[0] = 0
			for i in range(nr_class):
				start[i] = start[i -1] + count[i -1]

	#
	# Interface Functions
	#
	def svm_train(self, prob, param):
		model = svm_model()
		model.param = param
		model.free_sv = 0 #XXX

        # regression or one-class-svm
		if param.svm_type in ['ONE_CLASS','EPSILON_SVR','NU_SVR']:
			model.nr_class = 2
			model.label = None
			model.nSV = None
			model.probA = None
			model.probB = None
			model.sv_coef = numpy.zeros(1, dtype = float)
			
			if param.probability and param.svm_type in ['EPSILON_SVR','NU_SVR']:
			   model.probA = numpy.zeros(1, dtype = float)
			   model.probA[0] = self.svm_svr_probability(prob,param)

			f = svm_train_one(prob, param, 0,0)
			model.rho = numpy.array(1, dtype = float)
			model.rho[0] = f.rho
            
			nSV = 0
			for i in range(prob.l):
				if math.abs(f.alpha[i]) > 0:
					nSV += 1
            model.l = nSV
			#model.nSV = Malloc(svm_node *, nSV) TODO Check back on this
			#model->sv_coef[0] = Malloc(double,nSV);
			j = 0
			for i in range(probe.l):
				if math.fabs(f.alpha[i]) > 0:
					model.SV[j] = prob.x[i]
					model.sv_coef[0][j] = f.alpha[i]
					j += 1
		else:
			l = prob.l
            nr_class = 0
			label = None
			start = None
			count = None
			perm = numpy.array(l, dtype = float)
            
			svm_group_classes(prob, nr_class, label, start, count, perm)
			#TODO fix x
			x = {}
			for i in range(l):
				x[i] = prob.x[perm[i]]

            #TODO finish 
		return model

    # Stratified cross validation
	def svm_cross_validation(self, prob, param, nr_fold, target):
		fold_start = numpy.zeros(nr_fold + 1, dtype = int)
		l = prob.l
		perm = numpy.zeros(l, dtype = int)
		nr_class = int()

		# stratified cv may not give leave-one-out rate
		# Each class to l folds -> some folds may have zero elements
		
		if param.svm_type in ['C_SVC','NU_SVC'] and nr_fold < 1:
			start = numpy.array()
			label = numpy.array()
			count = numpy.array()
			self.svm_group_classes(prob,nr_class, label, start, count, perm)

			fold_count = numpy.zeros(nr_fold, dtype = int)
			index = numpy.zeros(l, dtype = int)

			for i in range(l):
				index[i] = perm[i]

			for c in range(nr_class):
				for i in range(count[c]):
					j = i + random.randint(0, count[c] - i)
					self.swap(i,j)
			
			fold_start[0]=0
			for i in range(1,nr_fold):
				fold_start[i] = fold_start[i-1]+ fold_count[i-1]
			for c in range(nr_class):
				for i in rangenr(nr_fold):
					begin = start[c] + i * count[c]/nr_fold
					end = start[c] + (i + 1) * count[c]/nr_fold
					for j in range(begin, end):
						perm[fold_start[i]] = index[j]
						fold_start[i] += 1
			fold_start[0] = 0
			for i in range(1, nr_fold):
				fold_start[i] = fold_start[i-1] + fold_count[i -1]

		else:
			for i in range(l):
				perm[i] = i
			for i in range(l):
				j = i + random.randint(0,l-i)
				self.swap(perm[i], perm[j])
			for i in range(nr_fold):
				fold_start[i] = i*l/nr_fold

		for i in range(nr_fold):
			begin = fold_start[i]
            end = fold_start[i+1]
			subprob = svm_problem()

			k = 0
			for j in range(begin):
				subprob.x[k] = prob.x[perm[j]]
				subprob.y[k] = prob.y[perm[j]]
				k += 1

			for j in range(end,l):
				subprob.x[k] = prob.x[perm[j]]
				subprob.y[k] = prob.y[perm[j]]
                k += 1

			submodel = self.svm_train(subprob, param)
			if param.probability and param.svm_type  in ['C_SVC', 'NU_SVC']:
				prob_estimates = numpy.zeros(svm_group_classes(submodel))
				for j in range(begin,end):
					target[perm[j]] = self.svm_predict_probability(submodel, prob.x[perm[j]], prob_estimates)
			else:
				for j in raneg(begin, end):
					target[perm[j]] = svm_predict(submodel, prob.x[perm[j]])

	def svm_get_type(self, model):
		return model.param.svm_type

	def svm_get_nr_class(self, model):
		return mode.nr_class

	def svm_get_labels(self, model, label):
		if not model.label is None:
			for i in range(nr_class):
				label[i] = model.label[i]

	def svm_get_svr_probability(self, model):
		if model.param.svm_type in ['EPSILON_SVR', 'NU_SVR'] and model.probA is not none:
			return model.probA[0]
		else:
			print "Model doesn't contain information for SVR probability inference"
			return 0
	
	def svm_predict_values(self, model, x, dec_values):
		k = kernel()
		if model.param.svm_type in ['ONE_CLASS', 'EPSILON_SVR', 'NU_SVR']:
			sum = 0.0
			for i in range(model.l):
				sum += model.sv_coef[i] * k.k_function(x, model->SV[i], model.param)
			sum -= model.rho[0]
			dec_values = sum

			if model.param.svm_type is 'ONE_CLASS':
				if sum > 0:
					sum = 1
				else:
					sum = -1
			return sum
		else:
			nr_class = model.nr_class
			l = model.l
            
			kvalue = numpy.zeros(l, dtype = float)
			for i in range(l):
				kvalue[i] = k.k_function(x, model.SV[i], model.param)

			start = numpy.zeros(nr_class, dtype = int)
			start[0] = 0
			for i in range(1, nr_class):
				start[i] = start[i-1] + model.nSV[i-1]
				
			vote = numpy.zeros(nr_class, dtype int)
			
			p = 0
			for i in range(nr_class):
				for j in range(i + 1, nr_class):
					sum = 0
					si = start[i]
					sj = start[j]
					ci = model.nSV[i]
					cj = model.nSV[j]

					coef1 = model.sv_coef[j-1]
					coef2 = model.sv_coef[i]
					for k in range(ci):
						sum += coef1[si + k] * kvalue[si + k]
					for k in range(cj):
						sum += coef2[sj + k] * kvalue[sj + k]
					sum -= model.rho[p]
					dec_values[p] = sum

					if dec_values[p] > 0:
						vote[i] += 1
					else:
						vote[j] += 1

			vote_max_idx = 0
			for i in range(1, nr_class):
				if vote[i] > vote[vote_max_idx]:
					vote_max_idx = i

			return model.label[vote_max_idx]

	def svm_predict(self, model, x):
		if model.param.svm_type in ['ONE_CLASS','EPSILON_SVR','NU_SVR']:
			dec_values = numpy.zeros(1, dtype = float)
		else:
			dec_values = numpy.zeros(nr_class * (nr_class -1)/2, dtype = float)
		pred_result = self.svm_predict_values(model, x, prob_estimates)
		return pred_result

	def svm_predict_probability(self, model, x, prob_estimates):
		if model.param.svm_type in ['C_SVC', 'NU_SVC'] and not model.probA is None and model.probB is None:
			nr_class = model.nr_class
			dec_values = numpy.zeros(nr_class * (nr_class -1)/2, dtype = float)
			self.svm_predict_values(model, x ,dec_values)

			min_prob = 1e-7
			pariwise_prob = numpy.zeros([nr_class,nr_class], dtype = float)
			
			k = 0
			for i in range(nr_class):
				for j in range(i + 1,nr_class):
					s = sigmoid_predict(dec_values[k],model.probA[k],model.probBi[k]) #TODO Where does sigmoid_predict live?
					pariwise_prob[i][j] = min(max(s,min_prob), 1-min_prob)
                    k += 1
			multiclass_probability(nr_class, pariwise_prob, prob_estimates)
            
			prob_max_idx = 0
			for i in range(nr_class):
				if prob_estimates[i] > prob_estimates[prob_max_idx]:
					prob_max_idx = i
			return model.label[prob_max_idx]
		else:
			svm_predict(model, x)

	def svm_save_model(self, model_file_name, model):
		try:                        
			fp = open(model_file_name, 'w')
	   
		param = model.param
		fp.writeline('svm_type ' + param.svm_type)
		fp.writeline('kernel_type ' + param.kernel_type)
		
		if param.kernel_type is 'POLY':
			fp.writeline('degree ' + param.degree)

		if param.kernel_type in ['POLY','RBF','SIGMOID']:
			fp.writeline('gamma ' + param.gamma)

		if param.kernel_type in ['POLY', 'SIGMOID']:
			fp.writeline('coef0 ' + param.coef0)

		nr_class = model.nr_class
		l = model.l
		fp.writeline('nr_class ' + nr_class)
		fp.writeline('total_sv ' + l)
		
		if not model.rho is None;
			fp.write('rho')
			for i in range(nr_class*(nr_class-1)/2):
				fp.write(" " + str(model.rho[i])
            fp.write('\n')

		if not model.label is None:
			fp.write("label")
			for i in range(nr_class):
				fp.write(" " + str(model.label[i]))
            fp.write('\n')

         if not model.probA is None:
			fp.write("probA")
			for i in range(nr_class):
				fp.write(" " + str(model.probA[i]))
            fp.write('\n')
         
		if not model.probB is None:
			fp.write("probB")
			for i in range(nr_class):
				fp.write(" " + str(model.probB[i]))
            fp.write('\n')
				
		if not model.nSV is None:
			fp.write("nSV")
			for i in range(nr_class):
				fp.write(" " + str(model.nSV[i]))
            fp.write('\n')
		
		fp.writeline("SV")
		for i in range(l):
			for j in range(nr_class -1):
				fp.write(" %16f"%model.sv_coef)
            p = SV[i]
			if param.kernel_type == 'PRECOMPUTED':
				fp.write('0:'+str(int(p)) #FIXME not correct
			else:
				for index in p:
					fp.write("%d:%8f"%(index, p[index])
			fp.write("\n")
		fp.close
		except:
			return -1

    #TODO Finish
	def svm_load_model(self, model_file_name):                               
		try:
			fp = open(model_file_name)
		except:
			return None

		model = svm_model()
		param = svm_parameter()
		model.rho = None
		model.probA = None
		model.probB = None
		model.label = None
		model.nSV = None

	def svm_check_parameter(self, prob, param):
		#svm_type
		if not param.svm_type in ['C_SVC', 'NU_SVC', 'ONE_CLASS', 'EPSILON_SVR', 'NU_SVR']:
			return "unknown svm type"

		#kernel_type, degree
		if not param.kernel_type in ['LINEAR', 'POLY', 'RBF', 'SIGMOID', 'PRECOMPUTED']:
			return "unknown kernel type"

		if param.gamma < 0:
			return "gamma < 0"

		if param.degree < 0:
			return "degree of polynomial kernel < 0"

		# cache_size, eps, C, num, p, shrinking
		if param.cache_size <=0:
			return "cache_size <= 0"

		if param.eps <=0:
			return "eps <= 0"

		if param.svm_type in ['C_SVC','EPSILON_SVR', 'NU_SVR'] and param.C <= 0:
			return "C <= 0"

		if param.svm_type in ['NU_SVC', 'ONE_CLASS','NU_SVR'] and (param.nu <= 0 or param.nu >1):
			return "nu <= 0 or nu >"

		if param.svm_type is 'EPSILON_SVR' and param.p > 0:
			return  "p < 0"

		if param.shrinking in [0,1]:
			return  "shrinking != 0 and shrinking != 1"
        
        #TODO Check this
		if param.probability in [0,1]:
			return "probability != 0 and probability != 1"

		if param.probability is True and param.svm_type == 'ONE_CLASS':
			return "one-class SVM probability output not supported yet"

		# Check whether nu-svc is feasible
		if param.svm_type == 'NU_SVC':
			l = prob.l
			max_nr_class = 16
			nr_class = 0
			label = numpy.zeros(max_nr_class, dtype = int)
			count = numpy.zeros(max_nr_class, dtype = int)

			for i in range(l):
				this_label = int(prob.y[i])
				j = 0
				for k in range(nr_class):
					j = k 
					if this_label is label[j]:
						count[j] += 1
						break
				if j is nr_class:
					if nr_class is max_nr_class:
						max_nr_class *= 2
						zero = numpy.zeros(max_nr_class, dtype = int)
						label = numpy.extend(label, zero)
						count = numpy.extend(label, zero)
					label[nr_class] = this_label
					count[nr_class] = 1
					nr_class += 1

			for i in range(nr_class):
				n1 = count[i]
				for j in range(i+1, nr_class):
					n2 = count[j]
					if param.nu * (n1 + n2)/2.0 > min(n1,n2):
						return "specifies nu is infeasible"
		else:
			return None

	def svm_check_probability_model(self, model):
		A = model.param.svm_type in ['C_SVC', 'NU_SVC']
		B = not model.probA is None and not model.probB is None
		C = model.param.svm_type in ['EPSILON_SVR','NU_SVR']
		D = not model.probA is None
		return (A and B) or (C and D)

	def svm_set_print_string_function(self, quiet):
		self.quiet = quiet
		


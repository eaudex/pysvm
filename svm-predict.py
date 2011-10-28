import sys
import os 
import getopt

class predict:
	def exit_input_error(self, line_num):
		print "Wrong input format at line " + line_num
		sys.exit(1)

	def predict(self):
		correct = 0 
		total = 0
		error = 0.0
		sump = 0.0
		sumt = 0.0
		sumpp = 0.0
		sumpt = 0.0

		prob_estimates = 0
		j = 0

		if self.predict_probability:
			print "TODO"

		if self.model.type == 'NU_SVR' or self.model.type == 'EPSILON_SVR':
			print "Mean squared error = " + float(error)/total
			print "Squared correlation coefficient = " (total*sumpt -sump*sumt)*

	def exit_with_help(self):
		print """
	Usage: 
		svm-predict [options] test_file model_file output_file
	options: 
		-b probability_estimates: whether to predict probability 
		estimates, 0 or 1 (default 0); for one-class SVM only 0 
		is supported
			"""
		sys.exit(1);
	def main(self, args):
		try:
			opts, args = getopt.getopt(sys.argv[1:], 'b:')
			for opt, arg in opts:
				if opt == 'b':
					self.predict_probability = int(arg)
		except getopt.GetoptError:
			print str(getopt.GetoptError)	
			self.exit_with_help()

		if len(args) <= 2:
			self.exit_with_help()

		if not os.isfile(args[0]):
			print "can't open input file " + args[0]
			sys.exit(1)
        self.input = args[0]

		if not os.isfile(args[2]):
			print "can't open output file" + args[2]
			sys.exit(1)
		self.output = args[2]

		#TODO Check
		#if not svm.load_model(args[1]):
		#	print "can't open model file" + args[1]
		#	sys.exit(1)
		#
        
		#TODO Probabity
		#if self.predict_probability:
		#	if((model=svm_load_model(argv[i+1]))==0)
		#{
		#fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		#exit(1);
		#}

		#x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
		#if(predict_probability)
		#{
		#	if(svm_check_probability_model(model)==0)
		#	{
		#		fprintf(stderr,"Model does not support probabiliy estimates\n");
		#		exit(1);
		#	}
		#}
		#else
		#{
		#	if(svm_check_probability_model(model)!=0)
		#	printf("Model supports probability estimates, but disabled in prediction.\n");
		#}
		
		self.predict()
 
if __name__ == "__main__":
	p = predict()
	p.main(sys.argv[1:])
	p.exit_with_help()

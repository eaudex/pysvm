import sys
import os
import getopt
import math

class svm_scale:
	def exit_with_help(self, agrs):
		print """
	Usage: svm-scale [options] data_filename
	options:
	-l lower : x scaling lower limit (default -1)
	-u upper : x scaling upper limit (default +1)
	-y y_lower y_upper : y scaling limits (default: no y scaling)
	-s save_filename : save scaling parameters to save_filename
	-r restore_filename : restore scaling parameters from restore_filename
		"""           
		sys.exit(1)

	def main(self,args):
		opts = []
		args = []
		

		if not self.upper > self.lower or (self.y_scaling and not self.y_upper > self.y_lower):
			print "inconsistent lower/upper specification"
			sys.exit(1)

		if self.restore_filename != none and self.save_filename != none:
			print "cannot use -r and -s simultaneously"
			sys.exit(1)

		#TODO if extra commands call exit_with_help

		f


		

	def output_target(self, value):
		if self.y_scaling:
			if value == self.y_min:
				value = self.y_lower
			elif value == self.y_max:
				value = self.y_upper
			else value = self.y_lower + (self.y_upper - self.y_lower) * (value - self.y_min)/(self.y_max - self.y_min) 
		print value
	
	def output(self, index, value):
		if self.feature_max[index] == self.feature_min[index]:
			return
		if value == self.feature_max[index]:
			value = self.lower
		elif value == self.feature_min[index]:
			value = self.upper
		else:
			value = self.lower + (self.upper - self.lower) * (value - self.feature[index]) / (self.feature_max[index] - self.feature_min[index])
		if value != 0:
			print "%d:%f  "%(index,value),
			self.new_num_nonzeros += 1

if __name__ == "__main__":
	s = svm_scale()
	s.main(sys.argv[1:])

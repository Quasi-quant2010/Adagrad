#ifndef __L2_H__
#define __L2_H__

#include <math.h>

#include "util.h"
#include "arg_option.h"
#include "read_file.h"

void batch_train(std::vector<samples>&,
		 std::unordered_map<unsigned int, double>&, double*,
		 command_args*,
		 FILE*);

void mini_batch_train(std::vector<samples>&,
		      std::unordered_map<unsigned int, double>&, double*,
		      command_args*,
		      FILE*);

#endif //__L2_H__

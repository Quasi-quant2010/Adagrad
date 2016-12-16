#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <sli/tarray_tstring.h>
#include <sli/tstring.h>
using namespace sli;

#include <unordered_map>

#include "util.h"
#include "read_file.h"

/*
static double dotproduct(train_data**, unordered_map<unsigned int, double>&);
static double lambda_hat(unsigned int, unsigned int, double);
static double prox_screening(const double, const double);
static void copy_feature_weigth(unordered_map<unsigned int, double>&,
                                unordered_map<unsigned int, double>&);
static void show_hash(unordered_map<unsigned int, double>&);
*/

void batch_train(std::vector<samples>&,
		 std::unordered_map<unsigned int, double>&, double*,
		 command_args*,
		 FILE*);

void mini_batch_train(std::vector<samples>&,
		      std::unordered_map<unsigned int, double>&, double*,
		      command_args*,
		      FILE*);

#endif //__MODEL_H__

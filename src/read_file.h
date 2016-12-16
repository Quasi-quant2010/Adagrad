#ifndef __READ_FILE_H__
#define __READ_FILE_H__

#include <sys/time.h>
#include <time.h>
#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <vector>
#include <unordered_map>

#include <sli/stdstreamio.h>
#include <sli/tarray_tstring.h>
#include <sli/asarray_tstring.h>
#include <sli/tstring.h>
#include <sstream>
using namespace sli;

#include "util.h"
#include "arg_option.h"


void 
load_data(std::vector<samples>&, unsigned int*,
	  std::unordered_map<unsigned int, double>&,
	  char*, 
	  tstring, tstring, tstring);

void
show_data(std::vector<samples>&);

void
show_random_data(std::vector<samples>&);

void
show_hash(std::unordered_map<unsigned int, double>&);

void
initialze_feature_weight(std::unordered_map<unsigned int, double>&);

#endif //__READ_FILE_H__

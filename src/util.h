#ifndef __UTIL__H__
#define __UTIL__H__

#include <unordered_map>

#include <sli/tstring.h>
using namespace sli;

#include "arg_option.h"

typedef std::unordered_map<unsigned int, double> feature_vector;

typedef struct _samples{
  int click;
  feature_vector fv;
} samples;

typedef struct _E_adaptive{
  double E;
  double max_grad;
} E_adaptive;

tstring make_filename(tstring, command_args*);

#endif //__UTIL__H__

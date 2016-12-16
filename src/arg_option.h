#ifndef __OPTION__H__
#define __OPTION__H__

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <sli/tstring.h>
using namespace sli;

typedef struct _command_args{
  char* train_file;
  char* test_file;
  char* out_path;
  char* out_fname;
  double step_size;
  double epsilon;
  double clip_threshold;
  char* clip_method;
  double lambda;
  double convergence_rate;
  unsigned int max_iter;
  unsigned int mini_batch_size;
} command_args;

void read_args(int, char**, command_args*);

#endif //__OPTION__H__

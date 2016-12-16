#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <sli/tarray_tstring.h>
#include <sli/tstring.h>
using namespace sli;

#include <unordered_map>

#include "src/arg_option.h"
#include "src/util.h"
#include "src/read_file.h"
#include "src/L2.h"

int 
main(int argc, char **argv)
{

  // 1. option parser
  command_args *option_args = (command_args*)malloc(sizeof(command_args));
  printf("[Setting]\n");
  read_args(argc, argv, option_args);
  printf("\ttrain=%s,\n\ttest=%s,\n\tout_path=%s,\n\tstep_size=%1.2e,\n\tlambda=%1.2e,\n\tconvergence_rate=%1.2e,\n\tmax_iter=%d,\n\tmini_batch_size=%d\n", 
	 option_args->train_file, option_args->test_file, option_args->out_path, 
	 option_args->step_size, option_args->lambda, option_args->convergence_rate, option_args->max_iter, option_args->mini_batch_size);
  printf("[SGD Setting]\n");
  printf("\tepsilon=%1.2e,\n\tclip_threshold=%1.2e,\n\tclip_method=%s\n", 
	 option_args->epsilon, option_args->clip_threshold, option_args->clip_method);

  // 2. read train data and initialize feature weight vector
  std::vector<samples> train_data;
  unsigned int train_data_size = 0;
  unsigned int *train_data_size_ptr = &train_data_size;

  tstring line_delimiter, line_delimiter_between, line_delimiter_within;
  line_delimiter.init(); line_delimiter.append(" ");
  line_delimiter_between.init(); line_delimiter_between.append(":");
  line_delimiter_within.init(); line_delimiter_within.append(":");

  // feature weight
  std::unordered_map<unsigned int, double> feature_weight;
  double fw_level = 0.0; double *fw_level_ptr = &fw_level;

  load_data(train_data, train_data_size_ptr,
	    feature_weight,
	    option_args->train_file,
	    line_delimiter, line_delimiter_between, line_delimiter_within);
  fprintf(stdout, "\ttrain_length:%u\n", train_data_size);
  //show_data(train_data);
  //show_random_data(train_data, train_data_size_ptr);
  //show_hash(feature_weight);


  // 3. main

  tstring dummy_fname; 
  tstring out_fname;
  FILE *output;
  
  // batch
  dummy_fname.init(); dummy_fname.assign("Batch");
  out_fname.init();
  out_fname = make_filename(dummy_fname, option_args);
  initialze_feature_weight(feature_weight);
  
  if ( (output = fopen(out_fname.cstr(), "w")) == NULL ) {
    printf("can not make output file");
    exit(1);
  } 
  batch_train(train_data, feature_weight, fw_level_ptr,
	      option_args,
	      output);
  fclose(output);


  // Mini-batch
  dummy_fname.init(); dummy_fname.assign("SGD");
  out_fname.init();
  out_fname = make_filename(dummy_fname, option_args);
  initialze_feature_weight(feature_weight);

  if ( (output = fopen(out_fname.cstr(), "w")) == NULL ) {
    printf("can not make output file");
    exit(1);
  }
  mini_batch_train(train_data, feature_weight, fw_level_ptr,
		   option_args,
		   output);
  fclose(output);
  

  // free
  //free_data(&train_data, train_data_size_ptr);
  free(option_args);

  return 0;
}

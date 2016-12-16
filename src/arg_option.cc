#include "arg_option.h"

static struct option options[] =
  {
    {"train_file", required_argument, NULL, 'a'},
    {"test_file", required_argument, NULL, 'b'},
    {"out_path", required_argument, NULL, 'c'},
    {"out_fname", required_argument, NULL, 'd'},
    {"step_size", required_argument, NULL, 'e'},
    {"epsilon", required_argument, NULL, 'f'},
    {"clip_threshold", required_argument, NULL, 'g'},
    {"clip_method", required_argument, NULL, 'h'},
    {"lambda", required_argument, NULL, 'i'},
    {"convergence_rate", required_argument, NULL, 'j'},
    {"max_iter", required_argument, NULL, 'k'},
    {"mini_batch_size", required_argument, NULL, 'l'},
    {0, 0, 0, 0}
  };


void read_args(int argc, char **argv, command_args *option_args){

  int dummy, index;
  while( (dummy = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l", options, &index)) != -1 ){
    switch(dummy){
    case 'a':
      option_args->train_file = optarg;
      break;
    case 'b':
      option_args->test_file = optarg;
      break;
    case 'c':
      option_args->out_path = optarg;
      break;
    case 'd':
      option_args->out_fname = optarg;
      break;
    case 'e':
      option_args->step_size = atof(optarg);
      break;
    case 'f':
      option_args->epsilon = atof(optarg);
      break;
    case 'g':
      option_args->clip_threshold = atof(optarg);
      break;
    case 'h':
      option_args->clip_method = optarg;
      break;
    case 'i':
      option_args->lambda = atof(optarg);
      break;
    case 'j':
      option_args->convergence_rate = atof(optarg);
      break;
    case 'k':
      option_args->max_iter = (unsigned int)atoi(optarg);
      break;
    case 'l':
      option_args->mini_batch_size = (unsigned int)atoi(optarg);
      break;
    default:
      printf("Error: An unkown option\n");
      exit(1);
    }
  }
  
}

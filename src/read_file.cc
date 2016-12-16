#include "util.h"
#include "arg_option.h"
#include "read_file.h"

/* プロトタイプ宣言 */
static 
unsigned int 
load_data_length(const char *filename);


static 
unsigned int 
load_data_length(const char *filename)
{
  //read file
  stdstreamio f_in;
  tstring line;

  // file except
  if ( f_in.open("r", filename) < 0 ){
    printf("Can not open %s\n", filename);
    exit(1);
  }
  
  unsigned int count = 0;
  while ( (line = f_in.getline()) != NULL  ) {
    count += 1;
  }

  return count;
}



// main function
void 
load_data(std::vector<samples> &data, unsigned int *data_size,
	  std::unordered_map<unsigned int, double> &feature_weight,
	  char *filename, 
	  tstring line_delimiter, tstring line_delimiter_between, tstring line_delimiter_within)
{

  // 1. count data length
  *data_size = load_data_length(filename);
   
  // 2. insert data into p from filename
  
  unsigned int j, k, feature_id, feature_length;
  double feature_score;
  stdstreamio f_in;
  tstring line;
  tarray_tstring my_arr;
  // file except
  if ( f_in.open("r", filename) < 0 ){
    fprintf(stderr, "Can not open %s\n", filename);
    exit(1);
  }

  j = 0;
  while ( (line = f_in.getline()) != NULL  ) {    

    samples sample;
    
    //split
    line.trim("\n"); my_arr.init();
    my_arr.split(line, line_delimiter.cstr(), true);//["+1","1:0.01","100:0.86",...]

    // insert data
    sample.click = my_arr[0].atoi();
    feature_length = (unsigned int)(my_arr.length() - 1);// -1 is label, -1 1:0.1 2:0.4
    tarray_tstring my_arr2;
    tstring key, value;
    for (k = 1;
	 k < feature_length + 1;
	 k++) {
      my_arr2.init();
      my_arr2.split(my_arr[k], line_delimiter_between.cstr(), true);//"1:0.01"
      key.init(); key = my_arr2[0]; feature_id = (unsigned int)key.atoi();
      value.init(); value = my_arr2[1]; feature_score = value.atof();   
      std::unordered_map<unsigned int, double>::iterator find_hash = sample.fv.find(feature_id);
      if (find_hash == sample.fv.end()) {
	sample.fv[feature_id] = feature_score;
      }

      // make feature weight
      std::unordered_map<unsigned int, double>::iterator sub_hash = feature_weight.find(feature_id);
      if (sub_hash == feature_weight.end()) {
	feature_weight[feature_id] = 0.0;
      }
    }
    data.push_back(sample);

    j += 1;
  }// over while
}


void 
show_data(std::vector<samples> &data)
{

  //unsigned int j, k;
  //fprintf(stdout, "train_length_ptr:%u\n", *data_size);
  //fprintf(stdout, "train_length_vector:%u\n", data.size());

  for (std::vector<samples>::const_iterator data_iter = data.begin();
       data_iter != data.end();
       ++data_iter) {
    fprintf(stdout, "click:%d ", data_iter->click);
    for (std::unordered_map<unsigned int, double>::const_iterator fv_iter = data_iter->fv.begin();
	 fv_iter !=  data_iter->fv.end(); 
	 ++fv_iter) {
      unsigned int key = fv_iter->first;
      double value = fv_iter->second;
      fprintf(stdout, "%d:%f ", key, value);
    }
    fprintf(stdout, "\n");
  }
  
}

void 
show_random_data(std::vector<samples> &data)
{

  gsl_rng_type *T = (gsl_rng_type *)gsl_rng_mt19937;// random generator
  gsl_rng *r = gsl_rng_alloc(T);// random gererator pointer
  gsl_rng_set(r, time(NULL));// initialize seed for random generator by sys clock
  
  samples sample;
  unsigned int random_index;
  random_index = (unsigned int)gsl_rng_uniform_int(r, data.size());//[0,n-1]
  sample = data[random_index];

  fprintf(stdout, "click:%d ", sample.click);
  for (std::unordered_map<unsigned int, double>::const_iterator fv_iter = sample.fv.begin();
       fv_iter !=  sample.fv.end(); 
       ++fv_iter) {
    unsigned int key = fv_iter->first;
    double value = fv_iter->second;
    fprintf(stdout, "%d:%f ", key, value);
  }
  fprintf(stdout, "\n");

}

void 
initialze_feature_weight(std::unordered_map<unsigned int, double> &feature_weight_ref)
{
  

  double flatten = 0.0;
  size_t feature_length = 0;
  feature_length = feature_weight_ref.size();
  flatten = 1.0 / (double)feature_length;

  std::unordered_map<unsigned int, double>::iterator iter = feature_weight_ref.begin();
  for (; iter != feature_weight_ref.end(); iter++) {
    unsigned int key = iter->first;
    //double value = iter->second;
    feature_weight_ref[key] = flatten;
  }

}

void 
show_hash(std::unordered_map<unsigned int, double> &dict_ref)
{
  std::unordered_map<unsigned int, double>::iterator iter = dict_ref.begin();

  //fprintf(stdout, "size_hash:%d\n", dict_ref.size());

  fprintf(stdout, "--- Show Hash ---\n");
  for (; iter != dict_ref.end(); iter++) {
    unsigned int key = iter->first;
    double value = iter->second;
    fprintf(stdout, "%d:%f\n", key, value);
  }

}

/*
void 
free_data(data **p, unsigned int *p_size)
{

  unsigned int j;
  for (j = 0; 
       j < *p_size; 
       j++)
    free((*p + j)->featureid_score);

  free(*p);
}
*/

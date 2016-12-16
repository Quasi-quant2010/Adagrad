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
using namespace std;

#include "arg_option.h"
#include "util.h"
#include "read_file.h"
#include "L1.h"

/* プロトタイプ宣言 */
static void
dotproduct(feature_vector&, std::unordered_map<unsigned int, double>&,
	   double*);

static double 
sigmoid(double inner_product);

// screeing operator
static double 
lambda_hat(unsigned int iter, unsigned int sample_length, double lambda);

static double 
prox_screening(const double a, const double b);

static void 
copy_feature_weigth(unordered_map<unsigned int, double> &copy,
		    unordered_map<unsigned int, double> &base);

static double
LogLikelihood(feature_vector&, double*, std::vector<samples>&);

static double
TotalSumSquareError(feature_vector&, double*, std::vector<samples>&);

static double
LogLikelihoodSGD(feature_vector&, double*, std::vector<samples>&, size_t*, size_t);

static double
TotalSumSquareErrorSGD(feature_vector&, double*, std::vector<samples>&, size_t*, size_t);

static void
init_vector(double* a, size_t len);

//main
void batch_train(std::vector<samples> &data,
                 std::unordered_map<unsigned int, double> &fw, double *fw_level,
                 command_args *option_args,
                 FILE *fp)
{

  //unsigned int i, j; //i is sample, j is feature index
  double learning_rate;
  double before_loss, after_loss, cur_loss_rate;
  double cur_pow_error;
  unsigned int cur_iter;
  double inner_product, predict_click;
  double *inner_product_ptr = &inner_product;

  double pred=0., error=0., grad=0., feature_score=0., re_step_size = 0.0;
  unsigned int feature_id;
  size_t non_screening;

  // gradient descent
  cur_loss_rate = option_args->convergence_rate + 1.0;
  before_loss = 1.0;
  cur_iter = 1;
  while (cur_loss_rate >= option_args->convergence_rate) {

    learning_rate = 1.0 / sqrt((double)cur_iter);

    // 1. weight update : learning feature_weight
    for (size_t i=0; i < data.size(); i++) {
      inner_product = 0.0;
      dotproduct(data[i].fv, fw, inner_product_ptr);
      pred = sigmoid(inner_product + *fw_level);
      error = pred - (double)data[i].click;

      for (std::unordered_map<unsigned int, double>::iterator fv_iter = data[i].fv.begin();
	   fv_iter != data[i].fv.end();
	   fv_iter++) {
	feature_id = fv_iter->first;
	feature_score = fv_iter->second;
	grad = error * fv_iter->second;

	// feature weight
	unordered_map<unsigned int, double>::iterator fw_search = fw.find(feature_id);
	if (fw_search != fw.end()) {
          fw_search->second -= learning_rate * grad; //keyが存在する
	} else{
          fw[feature_id] = -learning_rate * grad;    //keyが存在しない
	}

	// feature weight level
	*fw_level -= learning_rate * error;

      } // over feature vecotr

    }

    if (cur_iter == 1) {
      non_screening = fw.size();
    }

    // 2. L1 regulatization
    re_step_size = lambda_hat(cur_iter, data.size(), option_args->lambda);
    unordered_map<unsigned int, double> before_update_fw;
    copy_feature_weigth(before_update_fw, fw);

    // feature weight
    unordered_map<unsigned int, double>::iterator before_update_fw_iter = before_update_fw.begin();
    for (;
         before_update_fw_iter != before_update_fw.end();
         before_update_fw_iter++) {
      unsigned int key = before_update_fw_iter->first;
      unordered_map<unsigned int, double>::iterator fw_iter = fw.find(key);
      double value = fw_iter->second;
      if (fabsf(value) < re_step_size) {
        fw.erase(fw_iter);                                              // variable screening
      } else  {
	fw_iter->second = prox_screening((const double)fw_iter->second, 
					 (const double)re_step_size);   // L1 update
      }
    }

    // feature weight level
    if (fabsf(*fw_level) < re_step_size) {
      *fw_level = 0.0;                                                  // variable screening
    } else {
      *fw_level = prox_screening((const double)*fw_level,
				 (const double)re_step_size);           // L1 update
    }

    // 3. cardinality
    fprintf(stdout, "(iteration, cardinarity, co-variables, sparsity)=(%u, %zu, %zu, %1.3e)\n",
    	    cur_iter, fw.size(), non_screening,
    	    (double)fw.size() / (double)non_screening);
    
    // 3. likelihood
    //after_loss = TotalSumSquareError(fw, fw_level, data);
    after_loss = LogLikelihood(fw, fw_level, data);
    fprintf(fp, "%d\t%f\n", cur_iter, after_loss);

    // 4. next iteration bool
    if (cur_iter == option_args->max_iter) break;
    //if (cur_iter == 1) break;
    before_loss = after_loss;
    cur_iter += 1;
  }// over while
  
}// over batch_train


/*
void mini_batch_train(std::vector<samples> &data,
		      std::unordered_map<unsigned int, double> &fw, double *fw_level,
		      command_args *option_args,
		      FILE *fp)
{

  //unsigned int i, j; //i is sample, j is feature index
  double learning_rate;
  double before_loss, after_loss, cur_loss_rate;
  double cur_pow_error;
  unsigned int cur_iter;
  double inner_product, predict_click;
  double *inner_product_ptr = &inner_product;

  gsl_rng_type *T = (gsl_rng_type *)gsl_rng_mt19937; // random generator
  gsl_rng *r = gsl_rng_alloc(T);                     // random gererator pointer
  gsl_rng_set (r, time(NULL));                       // initialize seed for random generator by sys clock

  // stochastic gradient descent with mini batch
  cur_loss_rate = option_args->convergence_rate + 1.0;
  before_loss = 1.0;
  cur_iter = 1;
  while (cur_loss_rate >= option_args->convergence_rate) {

    cur_pow_error = 0.0;
    learning_rate = 1.0 / sqrt((double)cur_iter);

    // 1. Sampling mini-batch data from train datas
    size_t *random_idx = (size_t *)malloc(sizeof(size_t*) * option_args->mini_batch_size);
    for (size_t i = 0; i < option_args->mini_batch_size; i++)
      random_idx[i] = gsl_rng_uniform_int(r, data.size());

    // 2. calculate error
    double *error = (double *)malloc(sizeof(double*) * option_args->mini_batch_size);
    init_vector(error, option_args->mini_batch_size);
    for (size_t i=0; i < option_args->mini_batch_size; i++) {
      inner_product = 0.0;
      dotproduct(data[random_idx[i]].fv, fw, inner_product_ptr);
      predict_click = sigmoid(inner_product + *fw_level);
      error[i] = predict_click - (double)data[random_idx[i]].click;
    }    

    // 3. gradient descent
    //  3.1 feature weight
    for (std::unordered_map<unsigned int, double>::iterator fw_iter = fw.begin();
	 fw_iter != fw.end();
	 fw_iter++) {
      double grad = 0.0;
      // calculate gradient
      for (size_t i=0; i < option_args->mini_batch_size; i++) {
	std::unordered_map<unsigned int, double>::iterator tmp = data[random_idx[i]].fv.find(fw_iter->first);
	//keyが存在する
	if (tmp != data[random_idx[i]].fv.end())
	  grad += tmp->second * error[i];
      }
      grad /= (double)option_args->mini_batch_size;

      // update
      fw_iter->second -= learning_rate * grad;
    }
    //  3.2 feature level
    double grad=0.0;
    for (size_t i = 0; i < option_args->mini_batch_size; i++)
      grad += error[i];
    grad /= (double)option_args->mini_batch_size;
    *fw_level -= learning_rate * grad;

    // 3. likelihood
    //after_loss = LogLikelihoodSGD(fw, fw_level, data, random_idx, option_args->mini_batch_size);
    after_loss = LogLikelihood(fw, fw_level, data);
    fprintf(fp, "%d\t%f\n", cur_iter, after_loss);

    // 4. next iteration bool
    if (cur_iter == option_args->max_iter) break;
    before_loss = after_loss;
    cur_iter += 1;
  }// over while
  
}// over mini_batch_train
*/


// プロトタイプ
static void
dotproduct(feature_vector &fv_t, std::unordered_map<unsigned int, double> &fw_t,
	   double* result)
{
  double z = 0.0;

  for (std::unordered_map<unsigned int, double>::iterator fv_iter = fv_t.begin();
       fv_iter != fv_t.end(); 
       fv_iter++) {

    unsigned int id = fv_iter->first;
    double score = fv_iter->second;
    std::unordered_map<unsigned int, double>::iterator sub_hash = fw_t.find(id);
    if (sub_hash != fw_t.end()) {
      //keyが存在する
      unsigned int id_t = sub_hash->first;
      double weight = sub_hash->second;
      z += weight * score;//feature_weigth * data_score
      //fprintf(stdout, "%d:(%f:%f) ", id, score, weight);
    }

  }
  //fprintf(stdout, "\n");

  *result = z;
}

static double 
sigmoid(double z)
{
  if (z > 6.0) {
    return 1.0;
  } else if (z < -6.0) {
    return 0.0;
  } else {
    return 1.0 / (1.0 + exp(-z));
  }
  
}

static double lambda_hat(unsigned int iter, unsigned int sample_length, double lambda)
{
  return ( 1.0 * lambda) / ( 1.0 + (double)iter/(double)sample_length );
}

static double prox_screening(const double a, const double b)
{
  if (a > 0.0) {
    if (a > b) {
      return a - b;
    } else {
      return 0.0;
    }
  } else {
    if (a < - b) {
      return a + b;
    } else {
      return 0.0;
    }
  }
}

static void copy_feature_weigth(unordered_map<unsigned int, double> &copy,
                                unordered_map<unsigned int, double> &base)
{
  unordered_map<unsigned int, double>::iterator iter = base.begin();
  for (; iter != base.end(); iter++) {
    unsigned int key = iter->first;
    double value = iter->second;
    copy[key] = value;
  }
}

static double
LogLikelihood(feature_vector &fw_t, double *fw_level, std::vector<samples> &data_t)
{
  // Loss(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} y_i * log(\hat{y}_i) + (1 - y_i) * log(1 - \hat{y}_i)
  double inner_product;
  double *inner_product_ptr = &inner_product;
  double loss = 0.0;
  double predict_click;

  for (size_t i=0; i < data_t.size(); i++) {
    inner_product = 0.0;
    dotproduct(data_t[i].fv, fw_t, inner_product_ptr);
    predict_click = sigmoid(inner_product + *fw_level);
    if (data_t[i].click == 1) {
      loss += log(predict_click);
    } else if (data_t[i].click == 0) {
      loss += log(1.0 - predict_click);
    }
  }

  return -loss / (double)data_t.size();
}

static double
LogLikelihoodSGD(feature_vector &fw_t, double *fw_level, std::vector<samples> &data_t, size_t* sequence, size_t len_sequence)
{
  double inner_product = 0.0;
  double *inner_product_ptr = &inner_product;
  double loss = 0.0;
  double predict_click;

  for (size_t i=0; i < len_sequence; i++) {
    inner_product = 0.0;
    dotproduct(data_t[sequence[i]].fv, fw_t, inner_product_ptr);
    predict_click = sigmoid(inner_product + *fw_level);
    if (data_t[sequence[i]].click == 1) {
      loss += log(predict_click);
    } else if (data_t[sequence[i]].click == 0) {
      loss += log(1.0 - predict_click);
    }
  }

  return -loss / (double)len_sequence;
}

static double
TotalSumSquareError(feature_vector &fw_t, double *fw_level, std::vector<samples> &data_t)
{
  double inner_product;
  double *inner_product_ptr = &inner_product;
  double tmp_sum = 0.0;
  double predict_click;

  for (size_t i=0; i < data_t.size(); i++) {
    inner_product = 0.0;
    dotproduct(data_t[i].fv, fw_t, inner_product_ptr);
    predict_click = sigmoid(inner_product + *fw_level);
    tmp_sum += pow((double)data_t[i].click - predict_click, 2.0);    
  }
  
  return tmp_sum / (double)data_t.size();
}

static double
TotalSumSquareErrorSGD(feature_vector &fw_t, double *fw_level, std::vector<samples> &data_t, size_t* sequence, size_t len_sequence)
{
  double inner_product;
  double *inner_product_ptr = &inner_product;
  double tmp_sum = 0.0;
  double predict_click;

  for (size_t i=0; i < len_sequence; i++) {
    inner_product = 0.0;
    dotproduct(data_t[sequence[i]].fv, fw_t, inner_product_ptr);
    predict_click = sigmoid(inner_product + *fw_level);
    tmp_sum += pow((double)data_t[sequence[i]].click - predict_click, 2.0);
  }
  
  return tmp_sum / (double)len_sequence;
}

static void
init_vector(double* a, size_t len)
{
  for (size_t i=0; i < len; i++)
    a[i] = 0.0;
}

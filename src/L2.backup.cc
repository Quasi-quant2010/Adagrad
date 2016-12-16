#include "util.h"
#include "arg_option.h"
#include "read_file.h"
#include "L2.h"

/* プロトタイプ宣言 */
/* ---------------- inner product ------------------- */
static void
dotproduct(feature_vector&, std::unordered_map<unsigned int, double>&,
	   double*);

static double 
sigmoid(double inner_product);


/* ---------------- Loss Function ------------------- */
static double
LogLikelihood(feature_vector&, double*, std::vector<samples>&,
	      double);

static double
TotalSumSquareError(feature_vector&, double*, std::vector<samples>&);

static void
init_vector(double*, size_t);

static void
init_struct_vector(E_adaptive*, size_t);

static double
get_L2Norm(feature_vector&);


/* ---------------- Adagrad ------------------------- */
static double
get_learnig_rate(const command_args*, 
		 double, double);

static double 
get_adjust_gradient(const command_args*,
                    double, double);

static double
get_max(double, double);

static 
double get_min(double, double);

static
double Clipping(double, double);

static
double MaxSqueezing(double, double, double);


/* ----------------- main ------------------------------*/
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

  // gradient descent
  cur_loss_rate = option_args->convergence_rate + 1.0;
  before_loss = 1.0;
  cur_iter = 1;
  while (cur_loss_rate >= option_args->convergence_rate) {

    cur_pow_error = 0.0;
    learning_rate = option_args->step_size / sqrt((double)cur_iter);

    // 1. calculate error
    double *error = (double *)malloc(sizeof(double*) * data.size());
    init_vector(error, data.size());
    for (size_t i=0; i < data.size(); i++) {
      inner_product = 0.0;
      dotproduct(data[i].fv, fw, inner_product_ptr);
      predict_click = sigmoid(inner_product + *fw_level);
      error[i] = predict_click - (double)data[i].click;                                       // error
    }    

    // 2. gradient descent
    //  2.1 features
    for (std::unordered_map<unsigned int, double>::iterator fw_iter = fw.begin();
	 fw_iter != fw.end();
	 fw_iter++) {

      double grad = 0.0;
      // calculate gradient
      for (size_t i=0; i < data.size(); i++) {
	std::unordered_map<unsigned int, double>::iterator tmp = data[i].fv.find(fw_iter->first);
	//keyが存在する
	if (tmp != data[i].fv.end())
	  grad += error[i] * tmp->second; 
      }
      grad /= (double)data.size();                                                           // gradient

      // update
      fw_iter->second -= learning_rate * (grad + option_args->lambda * fw_iter->second);     // L2-regularization
    }
    //  2.2 feature_level
    double grad=0.0;
    for (size_t i = 0; i <  data.size(); i++)
      grad += error[i];
    grad /= (double)data.size();                                                             // gradient
    *fw_level -= learning_rate * (grad + option_args->lambda * *fw_level);                   // L2-regularization

    // 3. likelihood
    after_loss = LogLikelihood(fw, fw_level, data, option_args->lambda);
    fprintf(fp, "%d\t%f\n", cur_iter, after_loss);

    // 4. next iteration bool
    if (cur_iter == option_args->max_iter) break;
    before_loss = after_loss;
    cur_iter += 1;
  }// over while
  
}// over batch_train


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
  double *E = (double *)malloc(sizeof(double*) * (fw.size() + 1));
  //E_adaptive *E = (E_adaptive *)malloc(sizeof(E_adaptive*) * (fw.size() + 1));
  //init_struct_vector(E, fw.size() + 1);
  init_vector(E, fw.size() + 1);
  cur_loss_rate = option_args->convergence_rate + 1.0;
  before_loss = 1.0;
  cur_iter = 1;
  while (cur_loss_rate >= option_args->convergence_rate) {

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
      error[i] = predict_click - (double)data[random_idx[i]].click;                        // error
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
      grad /= (double)option_args->mini_batch_size;                                        // gradient
      /*
      grad = get_adjust_gradient(option_args, 
				 E[fw_iter->second].max_grad, grad);                       // Clipping
      */

      // update
      E[fw_iter->first] += pow(grad, 2.0);                                                 // cumulative (sub)gradient at iteration t for feature j
      //E[fw_iter->first].max_grad = get_max(E[fw_iter->first].max_grad, grad);            // For Max Clipping
      learning_rate = get_learnig_rate(option_args, 
				       E[fw_iter->first], (double)cur_iter);               // Adagrad : cumulative (sub)gradient at iteration t for feature j
      fw_iter->second -= learning_rate * (grad + option_args->lambda * fw_iter->second);   // L2-regularization
    }
    //  3.2 feature level
    double grad=0.0;
    for (size_t i = 0; i < option_args->mini_batch_size; i++)
      grad += error[i];
    grad /= (double)option_args->mini_batch_size;                                          // gradient
    /*
    grad = get_adjust_gradient(option_args, 
			       E[fw.size()].max_grad, grad);                               // Clipping
    */
    E[fw.size()] += pow(grad, 2.0);                                                        // cumulative (sub)gradient at iteration t for feature j
    //E[fw.size()].max_grad = get_max(E[fw.size()].max_grad, grad);                        // For Max Clipping
    learning_rate = get_learnig_rate(option_args,
				     E[fw.size()], (double)cur_iter);                      // Adagrad
    *fw_level -= learning_rate * (grad + option_args->lambda * *fw_level);                 // L2-regularization

    // 3. likelihood
    //after_loss = LogLikelihoodSGD(fw, fw_level, data, random_idx, option_args->mini_batch_size);
    after_loss = LogLikelihood(fw, fw_level, data, option_args->lambda);
    fprintf(fp, "%d\t%f\n", cur_iter, after_loss);

    // 4. next iteration bool
    if (cur_iter == option_args->max_iter) break;
    before_loss = after_loss;
    cur_iter += 1;
  }// over while
  
}// over mini_batch_train



// プロトタイプ
/* ---------------- inner product ------------------- */
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
      } else if (z < -6.0){
    return 0.0;
  } else {
    return 1.0 / (1.0 + exp(-z));
  }
  
}


/* ---------------- Loss Function ------------------- */
static double
LogLikelihood(feature_vector &fw_t, double *fw_level, std::vector<samples> &data_t,
	      double _lambda)
{
  // Loss(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} y_i * log(\hat{y}_i) + (1 - y_i) * log(1 - \hat{y}_i)
  double inner_product = 0.0;
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
  //loss /= (double)data_t.size();
  //loss = -loss + 0.5 * get_L2Norm(fw_t);

  return -loss / (double)data_t.size() + 0.5 * _lambda * get_L2Norm(fw_t);
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

static void
init_vector(double* a, size_t len)
{
  for (size_t i=0; i < len; i++)
    a[i] = 0.0;
}

static void
init_struct_vector(E_adaptive* _E, size_t len)
{
  for (size_t i=0; i < len; i++) {
    _E[i].E = 0.0;
    _E[i].max_grad = 0.0;
  }
}

static double
get_L2Norm(feature_vector& fw_t)
{
  double _sum = 0.0;
  for (std::unordered_map<unsigned int, double>::iterator fw_iter = fw_t.begin();
       fw_iter != fw_t.end();
       fw_iter++)
    _sum += pow(fw_iter->second, 2.0);
  return _sum;
}

/* ---------------- Adagrad ------------------------- */
static double
get_learnig_rate(const command_args* _option_args, 
		 double _cumulative_gradient, double _cur_iter)
{
  tstring bool_clip; bool_clip.init(); bool_clip.append("clippng");
  tstring bool_max_squeeze; bool_max_squeeze.init(); bool_max_squeeze.append("max_squeezing");
  double learning_rate = 0.0;

  if (bool_clip.compare(_option_args->clip_method) == 0) {
    learning_rate = _option_args->step_size / sqrt( _cumulative_gradient + _option_args->epsilon);
  } else if (bool_max_squeeze.compare(_option_args->clip_method) == 0) {
    learning_rate = _option_args->step_size / sqrt( _cumulative_gradient + _option_args->epsilon);
  } else {
    learning_rate = _option_args->step_size / sqrt(_cur_iter);
  }
  return learning_rate;
}

static double 
get_adjust_gradient(const command_args* _option_args, 
		    double _max_grad, double _grad)
{
  tstring bool_clip; bool_clip.init(); bool_clip.append("clippng");
  tstring bool_max_squeeze; bool_max_squeeze.init(); bool_max_squeeze.append("max_squeezing");
  
  double grad_ = _grad;
  if (bool_clip.compare(_option_args->clip_method) == 0) {
    // clipping
    grad_ = Clipping(_grad, _option_args->clip_threshold);
  } else if (bool_max_squeeze.compare(_option_args->clip_method) == 0) {
    // Max Squeezing
    grad_ = Clipping(_grad, _option_args->clip_threshold);
  } 

  return grad_;
}

static double
get_max(double a, double b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

static 
double get_min(double a, double b)
{
  if (a > b) {
    return b;
  } else {
    return a;
  }
}

static
double Clipping(double _grad, double _clip)
{
  _grad = get_max(get_min(_grad, _clip), -_clip);
  return _grad;
}

static
double MaxSqueezing(double _max_grad, double _grad, double _clip)
{
  if (_clip < _max_grad) {
    _grad *= 1.;
  } else {
    _grad *= _clip / _max_grad;
  }
  return _grad;
}
